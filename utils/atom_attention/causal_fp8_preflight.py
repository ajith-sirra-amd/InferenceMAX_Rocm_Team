"""Check causal FP8 reuse, masking, scales and changed-input graph replay."""
import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import torch

import inferencex_atom_attention.backend as backend_module
from preflight import check_output, make_backend, reference


def run():
    if os.environ.get("ATOM_USE_FLYDSL_FP8_CAUSAL_ATTN") != "1":
        raise RuntimeError("Causal preflight must explicitly enable the candidate")
    torch.manual_seed(37)
    for lengths in ([1], [17, 9], [257, 33], [512, 127]):
        total = sum(lengths)
        # Exercise noncontiguous inputs as produced by the native MLA split.
        packed = torch.randn(total, 12, 512, device="cuda", dtype=torch.bfloat16)
        q, k, v = packed[..., :192], packed[..., 192:384], packed[..., 384:]
        cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()],
                          device="cuda", dtype=torch.int32)
        metadata = SimpleNamespace(query_start_loc=cu, max_query_len=max(lengths),
                                   cu_seq_lens=cu, max_seq_len=max(lengths))
        backend = make_backend(12)
        backend.prepare_metadata(metadata)
        assert backend._causal_fp8_enabled
        quantizations = []
        real_quant = backend_module.fused_qkv_per_tensor_quant

        def quantize(*args, **kwargs):
            result = real_quant(*args, **kwargs)
            quantizations.append(result)
            return result

        def invoke(return_lse=True, out=None):
            backend.begin_forward(direct_kv=True)
            try:
                result = backend.run_prefill_new_tokens(q, k, v, return_lse, out=out)
                scales = backend._gather_scales.clone()
            finally:
                stats = backend.end_forward()
            return result, scales, stats

        with patch.object(backend_module, "fused_qkv_per_tensor_quant", quantize):
            (actual, lse), gather_scales, stats = invoke()
        assert len(quantizations) == 1, "Causal attention added a second quantization"
        assert stats["q_quantizations"] == 1
        assert backend._q_cache is None and backend._gather_scales is None
        quant = quantizations[0]
        torch.testing.assert_close(gather_scales, torch.cat(quant[6:8]), rtol=0, atol=0)
        torch.testing.assert_close(gather_scales, 2 * torch.cat(quant[4:6]), rtol=0, atol=0)
        operands = [quant[i].float() * quant[i+3] for i in range(3)]
        expected, expected_lse = reference(*operands, lengths, lengths, True)
        check_output(actual, expected, lse, expected_lse, f"causal_same_fp8_{lengths}")
        bf16_expected, bf16_lse = reference(q, k, v, lengths, lengths, True)
        check_output(actual, bf16_expected, lse, bf16_lse, f"causal_bf16_{lengths}")
        output = torch.empty_like(actual)
        no_lse, _, _ = invoke(False, output)
        assert isinstance(no_lse, torch.Tensor) and no_lse.data_ptr() == output.data_ptr()
        torch.testing.assert_close(no_lse, actual, rtol=0, atol=0)

        # Hold FP8 scales fixed: modifying future raw values could legitimately
        # change earlier quantized values through the per-tensor amax.
        if lengths[0] > 1:
            changed = list(quant)
            for index in (1, 2):
                changed[index] = quant[index].clone()
                tail = lengths[0] - 1
                changed[index][tail:].copy_((-quant[index][tail:].float()).to(quant[index].dtype))
            masked, masked_lse = backend._run(
                q, k, v, cu_q=cu, cu_k=cu, max_q=max(lengths), max_k=max(lengths),
                causal=True, return_lse=True, stage="causal_mask_test", quantized=changed,
            )
            torch.testing.assert_close(masked[:tail], actual[:tail], rtol=0, atol=0)
            torch.testing.assert_close(masked_lse[:, :tail], lse[:, :tail], rtol=0, atol=0)

        # Reusing a graph must recompute descales and consume changed inputs.
        for _ in range(3):
            invoke()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured, captured_scales, _ = invoke()
        q.mul_(0.5)
        k.mul_(0.75)
        v.mul_(1.5)
        graph.replay()
        eager, eager_scales, _ = invoke()
        torch.testing.assert_close(captured[0], eager[0], rtol=0, atol=0)
        torch.testing.assert_close(captured[1], eager[1], rtol=0, atol=0)
        torch.testing.assert_close(captured_scales, eager_scales, rtol=0, atol=0)
        print("CAUSAL_FP8_CASE_PASS", json.dumps({"lengths": lengths,
              "quantizations": 1, "graph_changed_inputs": True}), flush=True)
    print("CAUSAL_FP8_PREFLIGHT_PASS", flush=True)


if __name__ == "__main__":
    run()
