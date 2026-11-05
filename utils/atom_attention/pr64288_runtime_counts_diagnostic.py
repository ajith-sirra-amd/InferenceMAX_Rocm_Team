"""Compare the isolated runtime-count change with its parent on one GPU.

This is a kernel diagnostic, not serving, model-accuracy, or PR-head evidence.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

import torch

from vllm.v1.attention.ops import rocm_aiter_mla_fp8_quant as candidate

PARENT = "a03796b8e3bfe536e22f9db741109dbf80170079"
CANDIDATE = "768a5da2a8200b84c28b7d675414c95dfbc3ca55"
SOURCE = "vllm/v1/attention/ops/rocm_aiter_mla_fp8_quant.py"


def main(report):
    root = Path("/work/vllm")
    assert os.environ["NATIVE_SHA"] == CANDIDATE
    head = subprocess.check_output(
        ["git", "-c", "safe.directory=/work/vllm", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    assert head == CANDIDATE
    source = subprocess.check_output(
        ["git", "-c", "safe.directory=/work/vllm", "show", PARENT + ":" + SOURCE],
        cwd=root, text=True,
    )
    path = Path("/work/runtime_counts_parent.py")
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("runtime_counts_parent", path)
    parent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parent)
    torch.set_num_threads(8)
    torch.manual_seed(42)
    cache = torch.randn(131109, 576, device="cuda").to(torch.float8_e4m3fn)
    cache_scale = torch.tensor([0.25], device="cuda")
    norms = torch.tensor([0.2, 0.3], device="cuda")
    all_indices = torch.randperm(cache.shape[0], device="cuda").to(torch.int32)
    # Include empty input, partial tiles and both sides of power-of-two boundaries.
    cases = [(512, n) for n in (0, 1, 15, 16, 17, 255, 256, 257, 386, 4095, 4096, 4097, 8192)]
    cases += [(8192, n) for n in (8193, 12288, 16384, 24576, 32768, 49152, 98304, 131072)]
    for case_id, (query_rows, cache_rows) in enumerate(cases):
        q = torch.randn(query_rows, 12, 192, device="cuda", dtype=torch.bfloat16)
        indices = all_indices[:cache_rows]
        outputs = {}
        first_call_ms = {}
        order = (("parent", parent), ("candidate", candidate))
        if case_id % 2:
            order = tuple(reversed(order))
        for name, module in order:
            torch.cuda.synchronize()
            started = time.perf_counter()
            prepared = module.prepare_fp8_context(q, cache, cache_scale, indices, norms)
            scales = module.cached_kv_scales(cache, cache_scale, indices, norms)
            torch.cuda.synchronize()
            first_call_ms[name] = (time.perf_counter() - started) * 1000
            outputs[name] = (*prepared, scales)
        for actual, expected in zip(outputs["candidate"], outputs["parent"]):
            # CPU conversion supports exact comparison of E4M3 storage values.
            torch.testing.assert_close(
                actual.float().cpu(), expected.float().cpu(), atol=0, rtol=0
            )
        steady = {}
        if cache_rows in (8192, 16384, 32768, 131072):
            for name, module in order:
                def invoke():
                    return module.prepare_fp8_context(q, cache, cache_scale, indices, norms)
                for _ in range(10):
                    invoke()
                torch.cuda.synchronize()
                # Three blocks; measure host+device launch cost too.
                samples = []
                for _ in range(3):
                    started = time.perf_counter()
                    for _ in range(30):
                        invoke()
                    torch.cuda.synchronize()
                    samples.append((time.perf_counter() - started) * 1000 / 30)
                steady[name] = samples
        record = {
            "query_rows": query_rows, "cache_rows": cache_rows,
            "exact_parent_match": True, "first_call_ms": first_call_ms,
            "steady_prepare_ms": steady,
        }
        report["cases"].append(record)
        print("RUNTIME_COUNTS_CASE", json.dumps(record), flush=True)
    report["status"] = "passed"


if __name__ == "__main__":
    report = {
        "status": "running", "candidate": CANDIDATE, "parent": PARENT, "cases": [],
        "scope": "Single-GPU exact-output regression and exploratory kernel timing. First-call timings include compilation/allocation; not serving or model-accuracy evidence.",
    }
    try:
        main(report)
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        Path("/work/evidence/runtime-counts-diagnostic.json").write_text(
            json.dumps(report, indent=2)
        )
