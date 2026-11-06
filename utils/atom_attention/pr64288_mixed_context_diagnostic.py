"""Bounded real-layer DCP routing/merge diagnostic; not model accuracy."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import sys
import time

import torch
import torch.multiprocessing as mp

FLAG = "VLLM_ROCM_USE_AITER_FLYDSL_FP8_PREFILL_ATTN"
WORLD = 8
HEADS = 12


def source_versions():
    import aiter
    import vllm
    roots = {"aiter": Path(aiter.__file__).parent, "vllm": Path(vllm.__file__).parent}
    result = {"torch": torch.__version__, "hip": torch.version.hip}
    for package in ("aiter", "flydsl", "vllm"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not recorded by package metadata"
    result["source_files"] = {}
    for package, relative in (
        ("aiter", "ops/flydsl/gather_kv_b_proj.py"),
        ("vllm", "v1/attention/ops/rocm_aiter_mla_fp8_prefill.py"),
        ("vllm", "model_executor/layers/attention/mla_attention.py"),
    ):
        path = roots[package] / relative
        result["source_files"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


class Calls:
    """Observe actual function inputs without replacing any production function."""
    def __init__(self):
        self.records = []
        self.native_gathers = 0
        self.context_result = None

    def trace(self, frame, event, result):
        name = frame.f_code.co_name
        module = frame.f_globals.get("__name__", "")
        args = frame.f_locals
        if event == "call":
            if name == "_context_parallel_gather_prefill_kv":
                self.native_gathers += 1
            if module.endswith("rocm_aiter_mla_fp8_prefill") and name == "run":
                q = args["q"]
                key = (q.data_ptr(), tuple(q.shape), tuple(q.stride()))
                q8, qs = args["self"].queries[key]
                k, v, ks, vs = args["kv8"]
                self.records.append(("fp8", args["chunk"], q8, k, v, qs, ks, vs))
            if module.endswith("prefill.aiter_flash_attn"):
                if name == "run_prefill_context_chunk":
                    self.records.append(("native", args["chunk"], args["q"],
                                         args["k"], args["v"], None, None, None))
                elif name == "run_prefill_new_tokens":
                    self.records.append(("suffix", args["self"]._prefill_metadata,
                                         args["q"], args["k"], args["v"],
                                         None, None, None))
        elif event == "return" and name in (
            "compute_context", "_context_parallel_compute_prefill_context"
        ) and isinstance(result, tuple):
            self.context_result = result

    def reference(self, tokens, device):
        # FP64 CPU oracle uses the actual quantized inputs and natural-log LSE.
        output = torch.zeros(tokens, HEADS, 128, dtype=torch.float64)
        lse = torch.full((HEADS, tokens), -torch.inf, dtype=torch.float64)
        context_output = output.clone()
        context_lse = lse.clone()
        for kind, chunk, q, k, v, qs, ks, vs in self.records:
            tensors = []
            for value, descale in ((q, qs), (k, ks), (v, vs)):
                value = value.cpu().double()
                if descale is not None:
                    value = value * descale.cpu().double()
                tensors.append(value)
            q, k, v = tensors
            cq = chunk.query_start_loc.cpu().tolist()
            ck = cq if kind == "suffix" else chunk.cu_seq_lens.cpu().tolist()
            offset = 0 if kind == "suffix" else chunk.token_slice.start or 0
            for request, (qa, qb) in enumerate(zip(cq, cq[1:])):
                ka, kb = ck[request:request + 2]
                if qa == qb or ka == kb:
                    continue
                sl = slice(offset + qa, offset + qb)
                for head in range(HEADS):
                    logits = (q[qa:qb, head] @ k[ka:kb, head].T) * 192**-0.5
                    if kind == "suffix":
                        mask = torch.ones(qb-qa, kb-ka, dtype=torch.bool).tril()
                        logits.masked_fill_(~mask, -torch.inf)
                    local_lse = logits.logsumexp(-1)
                    local_out = logits.softmax(-1) @ v[ka:kb, head]
                    targets = [(output, lse)]
                    if kind != "suffix":
                        targets.append((context_output, context_lse))
                    for acc, acc_lse in targets:
                        merged = torch.logaddexp(acc_lse[head, sl], local_lse)
                        acc[sl, head] = (
                            acc[sl, head] * (acc_lse[head, sl] - merged).exp()[:, None]
                            + local_out * (local_lse - merged).exp()[:, None]
                        )
                        acc_lse[head, sl] = merged
        return output, context_output, context_lse


@torch.inference_mode()
def worker(rank, output_dir, init_method):
    from tests.v1.attention.utils import (
        BatchSpec, create_common_attn_metadata, create_vllm_config,
    )
    from vllm.config import set_current_vllm_config
    from vllm.config.quantization import QuantizationConfigArgs
    from vllm.distributed import (
        cleanup_dist_env_and_memory, init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention
    from vllm.model_executor.layers.linear import ColumnParallelLinear
    from vllm.model_executor.layers.quantization.online.base import OnlineQuantizationConfig
    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm.v1.attention.backends.mla.prefill.aiter_flash_attn import AiterFlashAttnPrefillBackend
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLABackend, AiterMLAMetadataBuilder
    from vllm.v1.attention.ops.rocm_aiter_mla_fp8_prefill import use_fp8_context
    from vllm.v1.kv_cache_interface import MLAAttentionSpec
    from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

    path = Path(output_dir) / f"mixed-rank-{rank}.json"
    report = {"rank": rank, "status": "running", "cases": [],
              "native_sha": os.environ["NATIVE_SHA"]}
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    device = torch.device("cuda", rank)
    config = create_vllm_config(
        model_name="deepseek-ai/DeepSeek-V2-Lite", dtype=torch.bfloat16,
        tensor_parallel_size=WORLD, max_model_len=32768, block_size=64,
        max_num_seqs=4, max_num_batched_tokens=2048,
    )
    config.cache_config.cache_dtype = "fp8"
    config.parallel_config.decode_context_parallel_size = WORLD
    config.parallel_config.cp_kv_cache_interleave_size = 64
    config.attention_config.use_prefill_query_quantization = False
    config.kernel_config.enable_jit_warmup = False
    init_workspace_manager(device)
    try:
        with set_current_vllm_config(config), set_default_torch_dtype(torch.bfloat16):
            init_distributed_environment(
                world_size=WORLD, rank=rank, local_rank=rank,
                distributed_init_method=init_method,
            )
            initialize_model_parallel(
                tensor_model_parallel_size=WORLD, decode_context_model_parallel_size=WORLD,
            )
            quant = OnlineQuantizationConfig(QuantizationConfigArgs(
                targets={"*.self_attn.kv_b_proj": "fp8_per_channel"}
            ))
            projection = ColumnParallelLinear(
                512, HEADS * WORLD * 256, bias=False, params_dtype=torch.bfloat16,
                quant_config=quant, prefix="model.layers.0.self_attn.kv_b_proj",
            ).to(device)
            torch.manual_seed(101 + rank)
            projection.weight.normal_(mean=0, std=0.02)
            projection.quant_method.process_weights_after_loading(projection)
            layers = {}
            for arm in ("off", "on"):
                os.environ[FLAG] = str(int(arm == "on"))
                layer = MLAAttention(
                    num_heads=HEADS, scale=192**-0.5, qk_nope_head_dim=128,
                    qk_rope_head_dim=64, v_head_dim=128, q_lora_rank=None,
                    kv_lora_rank=512, kv_b_proj=projection, cache_config=config.cache_config,
                    prefix=f"model.layers.{0 if arm == 'off' else 1}.self_attn.attn",
                    attn_backend=AiterMLABackend, prefill_backend_cls=AiterFlashAttnPrefillBackend,
                )
                layer.impl.process_weights_after_loading(torch.bfloat16)
                assert layer.impl._use_flydsl_fp8_prefill == (arm == "on")
                layers[arm] = layer
            report["versions"] = source_versions()
            report["projection_kernel"] = type(projection.quant_method.fp8_linear).__name__
            for name, query_lens in (("mixed", [512, 3, 7]), ("all_short", [7, 3, 5])):
                contexts = [8704, 0, 128]
                torch.manual_seed(42)
                q = torch.randn(sum(query_lens), HEADS, 192, device=device) * 0.5
                suffix = torch.randn(sum(query_lens), 576, device=device)
                cache_scale = torch.tensor([0.07], dtype=torch.float32, device=device)
                metadata = {}
                for arm, layer in layers.items():
                    spec = MLAAttentionSpec(
                        block_size=64, num_kv_heads=1, head_size=576,
                        dtype=torch.float8_e4m3fn, cache_dtype_str="fp8",
                    )
                    builder = AiterMLAMetadataBuilder(spec, [layer.layer_name], config, device)
                    builder.chunked_prefill_workspace_size = 8192
                    builder.chunked_prefill_workspace = torch.empty(
                        (8192 + 8192 // WORLD, 576), dtype=torch.bfloat16, device=device,
                    )
                    layer.dcp_manager.init_kv_gather(builder.chunked_prefill_workspace, 8192)
                    common = create_common_attn_metadata(
                        BatchSpec(seq_lens=[c+ql for c, ql in zip(contexts, query_lens)],
                                  query_lens=query_lens),
                        64, device, arange_block_indices=True,
                    )
                    metadata[arm] = builder.build(0, common)
                blocks_per_req = common.block_table_tensor.shape[1]
                cache = torch.zeros(
                    len(contexts), blocks_per_req * 64, 576,
                    dtype=torch.bfloat16, device=device,
                ).to(torch.float8_e4m3fn)
                for request, length in enumerate(contexts):
                    latent = torch.randn(length, 576, dtype=torch.bfloat16, device=device)
                    quantized = (latent.float() / cache_scale).to(cache.dtype).view(torch.uint8)
                    pos = torch.arange(length, device=device)
                    shard = quantized[(pos // 64) % WORLD == rank]
                    cache.view(torch.uint8)[request, :len(shard)].copy_(shard)
                cache = cache.view(-1, 64, 576)
                outputs = {
                    arm: torch.empty(sum(query_lens), HEADS * 128,
                                     dtype=torch.bfloat16, device=device)
                    for arm in layers
                }

                def forward(arm):
                    layers[arm].impl.forward_mha(
                        q, suffix[:, :512], suffix[:, None, 512:], cache,
                        metadata[arm], cache_scale, outputs[arm],
                    )

                case = {"name": name, "arms": {}, "timing_scope": "layer forward, not serving"}
                report["cases"].append(case)
                for arm in ("off", "on"):
                    chunked = metadata[arm].prefill.chunked_context
                    assert chunked is not None and chunked.empty_token_slices
                    chunks = chunked.chunks
                    eligible = sum(use_fp8_context(c.max_query_len, c.max_seq_len) for c in chunks)
                    if name == "mixed":
                        assert 0 < eligible < len(chunks), "Mixed routing was not constructed"
                        assert any(c.is_continuation for c in chunks)
                    else:
                        assert eligible == 0
                    calls = Calls()
                    sys.setprofile(calls.trace)
                    try:
                        forward(arm)
                        torch.cuda.synchronize()
                    finally:
                        sys.setprofile(None)
                    kinds = [r[0] for r in calls.records]
                    expected_fp8 = eligible if arm == "on" else 0
                    for kind, _, qi, ki, vi, *_ in calls.records:
                        expected_dtype = torch.float8_e4m3fn if kind == "fp8" else torch.bfloat16
                        assert qi.dtype == ki.dtype == vi.dtype == expected_dtype
                    assert kinds.count("fp8") == expected_fp8, (arm, kinds, expected_fp8)
                    assert kinds.count("native") == len(chunks) - expected_fp8
                    assert calls.native_gathers == len(chunks) - expected_fp8
                    assert layers[arm].prefill_backend.fp8_context is None
                    ref_out, ref_context, ref_lse = calls.reference(sum(query_lens), device)
                    actual = outputs[arm].view(-1, HEADS, 128).cpu().double()
                    torch.testing.assert_close(actual, ref_out, atol=2e-3, rtol=2e-3)
                    assert calls.context_result is not None
                    co, cl = calls.context_result
                    torch.testing.assert_close(co.cpu().double(), ref_context, atol=2e-3, rtol=2e-3)
                    torch.testing.assert_close(cl.cpu().double(), ref_lse, atol=1e-3, rtol=0)
                    case["arms"][arm] = {
                        "fp8_chunks": expected_fp8, "native_chunks": calls.native_gathers,
                        "output_max_absolute": (actual-ref_out).abs().max().item(),
                        "context_lse_max_absolute": (
                            cl.cpu().double()-ref_lse
                        )[torch.isfinite(ref_lse)].abs().max().item(),
                        "chunks": [
                            {"query": c.max_query_len, "context": c.max_seq_len,
                             "continuation": c.is_continuation} for c in chunks
                        ],
                        "numerical_status": "passed",
                    }
                    print("MIXED_ROUTING", rank, name, arm, json.dumps(case["arms"][arm]), flush=True)
                    del calls, ref_out, ref_context, ref_lse
                if name == "all_short":
                    torch.testing.assert_close(outputs["on"], outputs["off"], atol=0, rtol=0)
                    case["native_only_exact_match"] = True
                # Warm every arm/shape; tracing and oracle work are excluded.
                for arm in ("off", "on"):
                    for _ in range(5):
                        forward(arm)
                torch.cuda.synchronize()
                timings = {"off": [], "on": []}
                for arm in ("off", "on", "on", "off"):
                    torch.distributed.barrier()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    host_start = time.perf_counter()
                    start.record()
                    for _ in range(20):
                        forward(arm)
                    end.record()
                    end.synchronize()
                    timings[arm].append({
                        "gpu_ms_per_forward": start.elapsed_time(end) / 20,
                        "host_ms_per_forward": (time.perf_counter()-host_start)*1000/20,
                    })
                case["timing_samples"] = timings
                print("MIXED_TIMING", rank, name, json.dumps(timings), flush=True)
                path.write_text(json.dumps(report, indent=2) + "\n")
            report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        path.write_text(json.dumps(report, indent=2) + "\n")
        reset_workspace_manager()
        cleanup_dist_env_and_memory()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.device_count() == WORLD
    mp.spawn(worker, args=(str(args.output), (args.output / "dist-init").resolve().as_uri()),
             nprocs=WORLD, join=True)
    reports = [json.loads((args.output/f"mixed-rank-{rank}.json").read_text()) for rank in range(WORLD)]
    assert all(r["status"] == "passed" for r in reports)
    summary = {"status": "passed", "ranks": WORLD, "cases_per_rank": 2,
               "scope": "Random native FP8 weights; routing and FP64 input-oracle checks. Not Kimi accuracy, graph replay, or serving performance.",
               "native_sha": os.environ["NATIVE_SHA"]}
    (args.output / "mixed-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("MIXED_SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
