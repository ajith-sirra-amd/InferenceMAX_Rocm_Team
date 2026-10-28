"""GPU checks for native vLLM FP8 weight conversion and ATOM projection dispatch."""
import json
from types import SimpleNamespace

import torch

from inferencex_atom_attention.fp8_up_projection import (
    project, projection_weight, reference_projection, validate_projection,
)
from preflight import make_backend
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
from vllm.model_executor.kernels.linear.scaled_mm.aiter import (
    AiterHipbMMPerTokenFp8ScaledMMLinearKernel,
    AiterPerTokenFp8ScaledMMLinearKernel,
    AiterPreshuffledPerTokenFp8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.pytorch import RowWiseTorchFP8ScaledMMLinearKernel
from vllm.model_executor.layers.quantization.online.fp8 import Fp8PtpcOnlineLinearMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTokenSym, kFp8StaticChannelSym,
)


def quantize_layer(weight, packing_kernel=None):
    # Supply an already-loaded layer; execute native online quantization and packing.
    # No Kimi model is constructed for this isolated test.
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
    layer.tp_size = 1
    layer.input_size_per_partition = weight.shape[1]
    layer.bias = None
    method = Fp8PtpcOnlineLinearMethod.__new__(Fp8PtpcOnlineLinearMethod)
    method.requantization_source = None
    method.requantization_source_parameters = {}
    method.input_dtype = method.out_dtype = torch.bfloat16
    if packing_kernel is None:
        with set_current_vllm_config(VllmConfig()):
            method.fp8_linear = init_fp8_linear_kernel(
                activation_quant_key=kFp8DynamicTokenSym,
                weight_quant_key=kFp8StaticChannelSym, weight_shape=weight.shape,
                input_dtype=torch.bfloat16, out_dtype=torch.bfloat16,
                module_name="FP8ProjectionPreflight",
            )
    else:
        # Exercise each native packing implementation, including layouts that the
        # current tuning table does not select. No GEMM is invoked on these objects.
        kernel = packing_kernel.__new__(packing_kernel)
        kernel.layer_param_names = ("weight", "weight_scale", "input_scale", "input_scale_ub")
        kernel.config = SimpleNamespace(activation_quant_key=kFp8DynamicTokenSym)
        method.fp8_linear = kernel
    layer.quant_method = method
    method.process_weights_after_loading(layer)
    return layer


def layout_checks():
    heads = 12
    rows = heads * 256
    channels = (torch.arange(rows, device="cuda") * 17) % 512
    amplitudes = (448.0 / (2.0 ** (torch.arange(rows, device="cuda") % 3)))
    amplitudes *= torch.where(torch.arange(rows, device="cuda") % 2 == 0, 1, -1)
    weight = torch.zeros((rows, 512), device="cuda", dtype=torch.bfloat16)
    weight[torch.arange(rows, device="cuda"), channels] = amplitudes.to(torch.bfloat16)
    cache = (((torch.arange(9 * 576, device="cuda").reshape(9, 576) % 9) - 4) / 4
             ).to(torch.float8_e4m3fn)
    indices = torch.tensor([8, 2, 8, 0, 5], device="cuda", dtype=torch.int32)
    cache_scale = torch.tensor([0.125], device="cuda")
    raw = cache.double()[indices.long()] * 0.125
    expanded = (raw[:, channels] * amplitudes.double()).view(5, heads, 256)
    expected = (torch.cat((expanded[:, :, :128], raw[:, None, 512:].expand(-1, heads, -1)), -1),
                expanded[:, :, 128:])
    scales = torch.tensor([1.0, 1.0], device="cuda")
    for cls in (AiterHipbMMPerTokenFp8ScaledMMLinearKernel,
                AiterPreshuffledPerTokenFp8ScaledMMLinearKernel,
                AiterPerTokenFp8ScaledMMLinearKernel, RowWiseTorchFP8ScaledMMLinearKernel):
        layer = quantize_layer(weight, cls)
        params = projection_weight(layer, heads)
        for got, want in zip(reference_projection(cache, cache_scale, indices, params, heads), expected):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        kv8 = project(cache, cache_scale, indices, params, heads, scales)
        validate_projection(kv8, expected)
        bf16 = project(cache, cache_scale, indices, params, heads)
        for got, want in zip(bf16, expected):
            torch.testing.assert_close(got.double(), want, rtol=0, atol=0)
        print("FP8_NATIVE_LAYOUT_PASS", cls.__name__, flush=True)
    # The adapter must reject a scale with the wrong number of channels.
    layer.weight_scale = torch.nn.Parameter(torch.ones(1, device="cuda"), requires_grad=False)
    try:
        projection_weight(layer, heads)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid projection scale accepted")


def chain_checks():
    torch.manual_seed(109)
    heads, nk, nq = 12, 8192, 512
    weight = (torch.randn((heads * 256, 512), device="cuda") / 32).to(torch.bfloat16)
    layer = quantize_layer(weight)
    from inferencex_atom_attention.backend import AtomFlyDSLMLAImpl
    impl = AtomFlyDSLMLAImpl.__new__(AtomFlyDSLMLAImpl)
    impl.kv_b_proj, impl.num_heads = layer, heads
    impl.process_weights_after_loading(torch.bfloat16)
    params = impl._atom_fp8_projection
    if params is None:
        raise AssertionError("FP8 projection was not prepared by the actual backend")
    cache = torch.randn((nk, 576), device="cuda").to(torch.float8_e4m3fn)
    indices = torch.arange(nk - 1, -1, -1, device="cuda", dtype=torch.int32)
    cache_scale = torch.tensor([0.137], device="cuda")
    new_latent = (torch.randn((nq, 512), device="cuda") * 0.137).to(torch.bfloat16)
    new_rope = (torch.randn((nq, 1, 64), device="cuda") * 0.137).to(torch.bfloat16)
    q = torch.randn((nq, heads, 384), device="cuda", dtype=torch.bfloat16)[:, :, ::2]
    cq = torch.tensor([0, nq], device="cuda", dtype=torch.int32)
    ck = torch.tensor([0, nk], device="cuda", dtype=torch.int32)
    chunk = SimpleNamespace(query_start_loc=cq, cu_seq_lens=ck,
                            max_query_len=nq, max_seq_len=nk)
    backend = make_backend(heads)
    backend.prepare_metadata(SimpleNamespace(query_start_loc=cq, cu_seq_lens=cq,
                                             max_query_len=nq, max_seq_len=nq))

    def invoke():
        expanded_new = layer.quant_method.apply(layer, new_latent).view(nq, heads, 256)
        knew = torch.cat((expanded_new[:, :, :128], new_rope.expand(-1, heads, -1)), -1)
        vnew = expanded_new[:, :, 128:]
        backend.begin_forward(direct_kv=True)
        try:
            backend.run_prefill_new_tokens(q, knew, vnew, True)
            kv8 = project(cache, cache_scale, indices, params, heads, backend._gather_scales)
            out, lse = backend.run_prefill_projected_fp8(chunk, q, kv8)
        finally:
            stats = backend.end_forward()
        return kv8, out, lse, stats

    actual = invoke()
    expected = reference_projection(cache, cache_scale, indices, params, heads)
    validate_projection(actual[0], expected)
    k, v = (t.to(torch.bfloat16) for t in expected)
    backend.validate_projected_fp8(chunk, q, k, v, *actual[:3],
                                    exact_bf16_projection=False)
    assert actual[3] == {"q_quantizations": 1, "q_reuses": 1}, actual[3]
    for _ in range(3):
        invoke()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = invoke()
    q.mul_(0.5)
    cache.copy_((cache.float() * 0.5).to(cache.dtype))
    cache_scale.mul_(0.75)
    new_latent.mul_(0.5)
    new_rope.mul_(0.5)
    graph.replay()
    torch.cuda.synchronize()
    fresh = invoke()
    for a, b in zip(captured[0], fresh[0]):
        torch.testing.assert_close(a.float(), b.float(), rtol=0, atol=0)
    torch.testing.assert_close(captured[1], fresh[1], rtol=0, atol=0)
    torch.testing.assert_close(captured[2], fresh[2], rtol=0, atol=0)
    validate_projection(captured[0], reference_projection(cache, cache_scale, indices, params, heads))
    assert backend._q_cache is None and backend._gather_scales is None
    print("FP8_NATIVE_PROJECTION_CHAIN_PASS", json.dumps({
        "native_linear_kernel": params.linear_kernel, "graph_changed_inputs": True,
        "projection": "FP8xFP8", "output": "FP8", "fmha": "FP8",
    }), flush=True)


def run():
    layout_checks()
    chain_checks()


if __name__ == "__main__":
    run()
