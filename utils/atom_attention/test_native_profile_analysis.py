import gzip
import json

import pytest

import native_profile_analysis as analysis


def test_compressed_trace_analysis_counts_overlap_and_writes_rank_evidence(tmp_path):
    traces = tmp_path / "traces"
    traces.mkdir()
    (traces / "profile-control.json").write_text(json.dumps({"status": "captured"}))
    proof = tmp_path / "proof.json"
    proof.write_text(json.dumps({
        "native_sha": "abc",
        "phase": "replay",
        "workers": [{"rank": 0, "kinds": ["projection", "cached_fmha"]}],
    }))
    events = [
        {"ph": "X", "cat": "kernel", "name": "A", "ts": 0, "dur": 4,
         "args": {"device": 0}},
        {"ph": "X", "cat": "kernel", "name": "B", "ts": 2, "dur": 4,
         "args": {"device": 0}},
        {"ph": "X", "cat": "kernel", "name": "A", "ts": 8, "dur": 2,
         "args": {"device": 0}},
        {"ph": "X", "cat": "cpu_op", "name": "cpu", "ts": 0, "dur": 100},
    ]
    with gzip.open(traces / "native-steady_tp0_rank0.123.pt.trace.json.gz", "wt") as f:
        json.dump({"traceEvents": events}, f)
    report = analysis.analyze(traces, proof, "abc", 1, tmp_path / "output")
    worker = report["workers"][0]
    assert worker["kernel_count"] == 3
    assert worker["kernel_duration_us"] == 10
    assert worker["devices"]["0"]["kernel_busy_us"] == 8
    assert worker["devices"]["0"]["observed_kernel_span_us"] == 10
    assert report["top_kernels"][0] == {
        "kernel": "A", "calls": 2, "total_gpu_ms": 0.006,
        "mean_us": 3.0, "share_of_summed_kernel_time_pct": 60.0,
    }
    saved = json.loads((tmp_path / "output/profile-analysis.json").read_text())
    assert saved["workers"]["0"]["kernel_count"] == 3


def test_cpu_only_trace_cannot_be_reported_as_gpu_evidence():
    with pytest.raises(ValueError, match="no positive-duration GPU kernels"):
        analysis.summarize([
            {"ph": "X", "cat": "cpu_op", "name": "cpu", "ts": 0, "dur": 100}
        ])
