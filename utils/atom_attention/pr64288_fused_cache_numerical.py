"""Numerical gate for fused cached scales, direct projection and actual FP8 FMHA."""
import json
import sys
from pathlib import Path

import torch
from aiter.ops.flydsl import flydsl_flash_attn_fp8_func
from aiter.ops.shuffle import shuffle_weight
from pr64288_cache_bound_diagnostics import scales_from_cache
from pr64288_fused_cache_preparation import fused_prepare
from vllm.v1.attention.ops.rocm_aiter_mla_fp8_prefill import ProjectionWeight, project


def metrics(value, reference):
    err = value - reference
    return {
        "max_absolute": err.abs().max().item(),
        "relative_rms": (err.square().mean() / reference.square().mean()).sqrt().item(),
        "normalized_peak": (err.abs().max() / reference.abs().max()).item(),
    }


def run(report):
    torch.set_num_threads(8)
    torch.manual_seed(7)
    heads = 12
    softmax_scale = 192 ** -0.5
    weight = torch.randn(heads * 256, 512, device="cuda").to(torch.float8_e4m3fn)
    wscale = torch.linspace(0.005, 0.02, heads * 256, device="cuda")
    reference_weight = weight.cpu().double() * wscale.cpu().double()[:, None]
    for queries, rows in ((20, 386), (512, 8192)):
        raw = torch.randn(rows + 37, 576, device="cuda")
        cache = raw.to(torch.float8_e4m3fn)
        indices = torch.randperm(rows + 37, device="cuda")[:rows].to(torch.int32)
        cache_scale = torch.tensor([0.25], device="cuda")
        q = torch.randn(queries, heads, 192, device="cuda", dtype=torch.bfloat16) * 2
        cu_q = torch.tensor([0, queries], device="cuda", dtype=torch.int32)
        cu_k = torch.tensor([0, rows], device="cuda", dtype=torch.int32)
        original_q, original_indices = q.clone(), indices.clone()
        for shuffled in (False, True):
            q.copy_(original_q)
            indices.copy_(original_indices)
            cache_scale.fill_(0.25)
            params = ProjectionWeight(
                shuffle_weight(weight, (16, 16)) if shuffled else weight,
                wscale, shuffled,
            )
            _, norms = scales_from_cache(cache, cache_scale, indices, params, heads)

            def forward():
                q8, qs, scales = fused_prepare(q, cache, cache_scale, indices, norms)
                kv8 = project(cache, cache_scale, indices, params, heads, scales)
                k8, v8, ks, vs = kv8
                output, lse = flydsl_flash_attn_fp8_func(
                    q8, k8, v8, softmax_scale=softmax_scale,
                    cu_seqlens_q=cu_q, cu_seqlens_kv=cu_k,
                    max_seqlen_q=queries, max_seqlen_kv=rows,
                    cross_seqlen=True, causal=False, return_lse=True,
                    q_descale=qs, k_descale=ks, v_descale=vs,
                    out=None, stream=None,
                )
                return output, lse, q8, qs, kv8

            def check(result, phase):
                output, lse, q8, qs, (k8, v8, ks, vs) = result
                latent = cache.cpu().double()[indices.cpu().long()] * cache_scale.cpu().double()
                expanded = (latent[:, :512] @ reference_weight.t()).view(rows, heads, 256)
                k_ref = torch.cat((expanded[..., :128], latent[:, None, 512:].expand(-1, heads, -1)), -1)
                v_ref = expanded[..., 128:]
                record = {
                    "query_rows": queries, "cached_rows": rows,
                    "preshuffled": shuffled, "phase": phase, "projection": [],
                }
                report["cases"].append(record)
                for value, ref, scale in ((k8, k_ref, ks), (v8, v_ref, vs)):
                    reconstructed = value.cpu().double() * scale.cpu().double()
                    m = metrics(reconstructed, ref)
                    m["clipped"] = int((ref.abs() > 448 * scale.cpu().double()).sum())
                    record["projection"].append(m)
                    assert m["clipped"] == 0
                    assert m["relative_rms"] < 0.04
                    assert m["normalized_peak"] < 0.07
                qf = q8.cpu().double() * qs.cpu().double()
                kf = k8.cpu().double() * ks.cpu().double()
                vf = v8.cpu().double() * vs.cpu().double()
                logits = torch.einsum("qhd,khd->hqk", qf, kf) * softmax_scale
                expected = torch.einsum("hqk,khd->qhd", logits.softmax(-1), vf)
                ref_lse = logits.logsumexp(-1)
                record["fmha_output"] = metrics(output.cpu().double(), expected)
                record["fmha_lse_max_absolute"] = (lse.cpu().double() - ref_lse).abs().max().item()
                torch.testing.assert_close(output.cpu().double(), expected, atol=2e-3, rtol=2e-3)
                torch.testing.assert_close(lse.cpu().double(), ref_lse, atol=1e-3, rtol=0)
                # Report the total input-quantization effect separately; do not
                # confuse an FP8-input oracle with BF16/model accuracy.
                del logits, expected
                bf_logits = torch.einsum("qhd,khd->hqk", q.cpu().double(), k_ref) * softmax_scale
                bf_output = torch.einsum("hqk,khd->qhd", bf_logits.softmax(-1), v_ref)
                record["versus_prequantization_inputs"] = metrics(output.cpu().double(), bf_output)
                record["prequantization_lse_max_absolute"] = (
                    lse.cpu().double() - bf_logits.logsumexp(-1)
                ).abs().max().item()
                record["passed"] = True
                print("FUSED_NUMERICAL", json.dumps(record), flush=True)

            check(forward(), "eager")
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = forward()
            q.mul_(0.5)
            cache_scale.mul_(0.75)
            indices.copy_(indices.flip(0))
            graph.replay()
            check(captured, "changed_input_graph_replay")
    report["status"] = "passed"


if __name__ == "__main__":
    report = {
        "status": "running", "cases": [],
        "native_source": "e2393d702a3736dffdfbfc7eb25abbabe484a1df",
        "scope": "Experimental fused scale preparation plus actual projection and FP8 FMHA. Existing projection and FP8-input attention gates unchanged. Not serving/model validation.",
    }
    try:
        run(report)
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        Path(sys.argv[1]).write_text(json.dumps(report, indent=2))
