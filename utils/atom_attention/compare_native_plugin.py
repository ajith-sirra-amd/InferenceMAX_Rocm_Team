"""Compare native and plugin prefill on the same ROCm GPU and inputs.

Run in the patched vLLM environment. This checks kernel parity, not complete
serving, DCP communication, model accuracy, or end-to-end performance.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys

import torch

from vllm.model_executor.layers.attention.mla_attention import MLACommonPrefillMetadata
from vllm.v1.attention.backends.mla.rocm_aiter_flydsl_mla import (
    AiterFlyDSLPrefillBackend,
)
from vllm.v1.attention.ops.atom_per_tensor_fp8_quant import (
    fused_qkv_per_tensor_quant as native_quant,
)

FLAG = "ATOM_USE_FLYDSL_FP8_PREFILL_ATTN"


def exact(actual, expected):
    if actual.dtype == torch.float8_e4m3fn:
        actual, expected = actual.view(torch.uint8), expected.view(torch.uint8)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=False)


def make_backend(cls):
    return cls(
        num_heads=12, scale=192 ** -0.5, kv_lora_rank=512,
        qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
        vllm_config=None,
    )


def compare_case(plugin_cls, plugin_quant, flag, nq, nk):
    os.environ[FLAG] = flag
    backends = [make_backend(plugin_cls), make_backend(AiterFlyDSLPrefillBackend)]
    # Strided Q and V reproduce layouts accepted by the serving adapter.
    q = torch.randn(12, nq, 192, device="cuda").bfloat16().transpose(0, 1)
    k = torch.randn(nk, 12, 192, device="cuda").bfloat16()
    v = torch.randn(nk, 12, 256, device="cuda").bfloat16()[..., 128:]
    new_k = torch.randn(nq, 12, 192, device="cuda").bfloat16()
    new_v = torch.randn(nq, 12, 256, device="cuda").bfloat16()[..., 128:]
    cu_q = torch.tensor([0, nq], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, nk], device="cuda", dtype=torch.int32)
    chunk = MLACommonPrefillMetadata.ContextChunk(
        index=0, request_slice=slice(0, 1), token_slice=slice(0, nq),
        continuation_token_end=nq, is_continuation=False,
        num_context_tokens=nk, query_start_loc=cu_q, max_query_len=nq,
        cu_seq_lens=cu_k, starts=torch.zeros(1, device="cuda", dtype=torch.int32),
        max_seq_len=nk, seq_lens=torch.tensor([nk], device="cuda", dtype=torch.int32),
        token_to_seq=torch.zeros(nk, device="cuda", dtype=torch.int32),
        all_rows_active=True,
    )
    for backend in backends:
        metadata = MLACommonPrefillMetadata(
            block_table=torch.zeros((1, 1), device="cuda", dtype=torch.int32),
            query_start_loc=cu_q, max_query_len=nq, prefill_backend=backend,
        )
        backend.prepare_metadata(metadata)

    for step in range(2):
        if step:
            # Reuse storage and metadata while changing Q between forwards.
            q.mul_(-0.7).add_(0.3)
        if flag == "1":
            for got, expected in zip(
                native_quant(q, k, v)[:6], plugin_quant(q, k, v)[:6]
            ):
                exact(got, expected)
        results = []
        for backend in backends:
            backend.begin_forward()
            try:
                causal = backend.run_prefill_new_tokens(q, new_k, new_v, True)
                first = backend.run_prefill_context_chunk(chunk, q, k, v)
                second = backend.run_prefill_context_chunk(chunk, q, k, v)
            finally:
                stats = backend.end_forward()
            results.append((causal, first, second, stats))
        for native_pair, plugin_pair in zip(results[1][:3], results[0][:3]):
            for got, expected in zip(native_pair, plugin_pair):
                exact(got, expected)
        eligible = flag == "1" and nq >= 512 and nk >= 8192
        expected_stats = {
            "q_quantizations": 1 if eligible else 0,
            "q_reuses": 1 if eligible else 0,
        }
        for result in results:
            if result[3] != expected_stats:
                raise AssertionError((flag, nq, nk, step, result[3], expected_stats))
    torch.cuda.synchronize()
    return {
        "flag": flag, "query_tokens": nq, "context_tokens": nk,
        "forwards": 2, "cached_calls_per_forward": 2,
        "quantized_inputs_and_descales_exact": flag == "1",
        "outputs_and_lse_exact": True,
        "expected_cached_fmha": "FP8" if eligible else "BF16",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.plugin_root.resolve()))
    plugin_cls = importlib.import_module(
        "inferencex_atom_attention.backend"
    ).AtomFlyDSLPrefillBackend
    plugin_quant = importlib.import_module(
        "inferencex_atom_attention.quantize"
    ).fused_qkv_per_tensor_quant
    previous_flag = os.environ.get(FLAG)
    torch.manual_seed(0)
    try:
        cases = [
            compare_case(plugin_cls, plugin_quant, flag, nq, nk)
            for flag in ("0", "1")
            for nq, nk in ((511, 8192), (512, 8191), (512, 8192), (1024, 16384))
        ]
    finally:
        if previous_flag is None:
            os.environ.pop(FLAG, None)
        else:
            os.environ[FLAG] = previous_flag
    proof = {
        "status": "passed", "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(), "cases": cases,
        "scope": "Quantizer and prefill backend parity only; no full serving validation.",
    }
    args.output.write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(proof))


if __name__ == "__main__":
    main()
