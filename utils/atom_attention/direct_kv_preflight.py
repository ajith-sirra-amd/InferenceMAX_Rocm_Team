"""Exercise the opt-in backend's scale lifetime, Q reuse, and graph replay."""
import os
from types import SimpleNamespace

import torch

from inferencex_atom_attention.projection_fp8 import expand_context_fp8
from preflight import make_backend
from projection_experiment import fixture, native


def run():
    torch.manual_seed(73)
    q = torch.randn((512, 12, 384), device="cuda", dtype=torch.bfloat16)[:, :, ::2]
    data = fixture(8192)
    new = fixture(512)
    new = (*new[:4], data[4], 12)
    backend = make_backend(12)
    if not backend._direct_kv_enabled:
        raise RuntimeError("ATOM_USE_FUSED_KV_FP8_OUTPUT=1 is required")
    cq = torch.tensor([0, 512], device="cuda", dtype=torch.int32)
    ck = torch.tensor([0, 8192], device="cuda", dtype=torch.int32)
    chunk = SimpleNamespace(query_start_loc=cq, max_query_len=512,
                            cu_seq_lens=ck, max_seq_len=8192)
    new_meta = SimpleNamespace(query_start_loc=cq, max_query_len=512,
                              cu_seq_lens=cq, max_seq_len=512)
    backend.prepare_metadata(new_meta)

    def invoke():
        knew, vnew = native(new)
        backend.begin_forward(direct_kv=True)
        try:
            backend.run_prefill_new_tokens(q, knew, vnew, True)
            scales = backend._gather_scales
            kv8 = expand_context_fp8(
                data[0], data[1], data[2], data[4], 12,
                mode="supplied_scales", output_scales=scales,
            )
            out, lse = backend.run_prefill_projected_fp8(chunk, q, kv8)
            out2, lse2 = backend.run_prefill_projected_fp8(chunk, q, kv8)
        finally:
            stats = backend.end_forward()
        return kv8, out, lse, out2, lse2, stats

    kv8, out, lse, out2, lse2, stats = invoke()
    assert stats == {"q_quantizations": 1, "q_reuses": 2}, stats
    assert backend._q_cache is None and backend._gather_scales is None
    k, v = native(data)
    backend.validate_projected_fp8(chunk, q, k, v, kv8, out, lse)
    torch.testing.assert_close(out2, out, rtol=0, atol=0)
    torch.testing.assert_close(lse2, lse, rtol=0, atol=0)
    for _ in range(3):
        invoke()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = invoke()
    q.mul_(0.5)
    new[0].copy_((new[0].float() * 0.75).to(new[0].dtype))
    data[0].copy_((data[0].float() * 0.5).to(data[0].dtype))
    graph.replay()
    torch.cuda.synchronize()
    kv8, out, lse, out2, lse2, stats = captured
    k, v = native(data)
    backend.validate_projected_fp8(chunk, q, k, v, kv8, out, lse)
    fresh = invoke()
    for captured_tensor, fresh_tensor in zip(captured[0], fresh[0]):
        torch.testing.assert_close(captured_tensor.float(), fresh_tensor.float(), rtol=0, atol=0)
    torch.testing.assert_close(out, fresh[1], rtol=0, atol=0)
    torch.testing.assert_close(lse, fresh[2], rtol=0, atol=0)
    print("DIRECT_KV_BACKEND_NUMERICS_PASS", stats, flush=True)


if __name__ == "__main__":
    run()
