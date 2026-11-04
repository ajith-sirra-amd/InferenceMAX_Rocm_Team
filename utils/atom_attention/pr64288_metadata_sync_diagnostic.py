"""Diagnose the isolated metadata copy changes using real metadata objects."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
from vllm.model_executor.layers.attention.mla_attention import MLACommonPrefillMetadata
from vllm.utils.torch_utils import async_tensor_h2d

BEFORE = "44d9f92e05f45ef45ec11d762011a2179be9d9cc"
AFTER = "c894899149c43bae0873ec8e3b8b76213d267f8f"
ROOT = Path("/work/vllm")


def load_source(revision, evidence):
    path = "vllm/v1/attention/ops/rocm_aiter_mla_prefill.py"
    source = subprocess.check_output([
        "git", "-c", "safe.directory=/work/vllm", "-C", str(ROOT),
        "show", revision + ":" + path,
    ], text=True)
    target = evidence / ("context_" + revision[:8] + ".py")
    target.write_text(source)
    spec = importlib.util.spec_from_file_location("context_" + revision[:8], target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def probe(function):
    torch.cuda.set_sync_debug_mode("error")
    try:
        result = function()
    except RuntimeError as error:
        return {"sync_debug_error": str(error)}
    finally:
        torch.cuda.set_sync_debug_mode("default")
    return {"values": result.cpu().tolist(), "sync_debug_error": None}


def main(report, evidence):
    chunk = MLACommonPrefillMetadata.ContextChunk(
        index=0, request_slice=slice(0, 2), token_slice=slice(0, 2),
        continuation_token_end=0, is_continuation=False, num_context_tokens=10,
        query_start_loc=torch.tensor([0, 1, 2], device="cuda", dtype=torch.int32),
        max_query_len=1,
        cu_seq_lens=torch.tensor([0, 6, 10], device="cuda", dtype=torch.int32),
        starts=torch.zeros(2, device="cuda", dtype=torch.int32),
        max_seq_len=6,
        seq_lens=torch.tensor([6, 4], device="cuda", dtype=torch.int32),
        token_to_seq=torch.tensor([0]*6+[1]*4, device="cuda", dtype=torch.int32),
        all_rows_active=False,
        padded_local_seq_lens=[4, 4],
        local_context_lens_allranks=[[4, 2], [3, 1]],
        num_local_context_tokens=8, local_starts=[0, 0],
    )
    expected = [0, 1, 2, 3, 8, 9, 4, 5, 6, 12]
    for label, revision in (("before", BEFORE), ("after", AFTER)):
        module = load_source(revision, evidence)
        function = lambda: module.context_row_indices(chunk, torch.device("cuda"))
        torch.testing.assert_close(
            function().cpu(), torch.tensor(expected, dtype=torch.int32),
            rtol=0, atol=0,
        )
        entry = {"revision": revision, "exact_rows": True, "sync_debug": probe(function)}
        # Profile the actual metadata function, keeping profiler teardown outside
        # the recorded function scope. This is diagnostic, not a timing threshold.
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            with torch.profiler.record_function("metadata_copy_under_test"):
                result = function()
        relevant = []
        for event in profile.events():
            parent = event.cpu_parent
            inside = False
            while parent is not None:
                if parent.name == "metadata_copy_under_test":
                    inside = True
                    break
                parent = parent.cpu_parent
            if inside and any(word in event.name.lower() for word in ("sync", "memcpy", "copy", "pin")):
                relevant.append({"name": event.name, "cpu_us": event.cpu_time_total})
        entry["copy_events"] = relevant
        report["context_rows"][label] = entry
    # Isolate the underlying pointer-list transfer without pretending this is a
    # DSpark model-entry regression test.
    caches = [torch.empty(16, device="cuda") for _ in range(4)]
    values = [x.data_ptr() for x in caches]
    report["pointer_transfer_primitive"] = {
        "before": probe(lambda: torch.tensor(values, dtype=torch.int64, device="cuda")),
        "after": probe(lambda: async_tensor_h2d(values, dtype=torch.int64, device="cuda")),
        "scope": "Torch transfer primitive used by DSpark; model-entry validation still required",
    }
    assert report["pointer_transfer_primitive"]["after"]["values"] == values
    report["cached_models"] = [
        {"name": p.name, "is_directory": p.is_dir(), "has_config": (p/"config.json").is_file()}
        for p in Path("/models").iterdir()
        if "kimi" in p.name.lower() or "dspark" in p.name.lower()
    ]
    report["status"] = "passed"


if __name__ == "__main__":
    evidence = Path(os.environ["REVIEW_EVIDENCE"])
    report = {
        "status": "running", "context_rows": {},
        "scope": "Real context metadata and isolated Torch pointer transfer diagnostics; not serving validation.",
    }
    try:
        main(report, evidence)
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        (evidence/"metadata-sync-diagnostic.json").write_text(json.dumps(report, indent=2))
        print("METADATA_SYNC_DIAGNOSTIC", json.dumps(report), flush=True)
