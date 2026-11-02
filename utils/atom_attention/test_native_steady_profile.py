import gzip
import json
import pytest

import native_steady_profile as profile


class Clock:
    def __init__(self):
        self.value = 0.0

    def monotonic(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


@pytest.fixture
def environment(tmp_path, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(profile.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(profile.time, "sleep", clock.sleep)
    monkeypatch.setattr(profile.time, "time", clock.monotonic)
    monkeypatch.setattr(profile.os, "kill", lambda pid, signal: None)
    log = tmp_path / "benchmark.log"
    log.write_text("Phase profiling (profiling) started\n")
    calls = []
    output = tmp_path / "profiles"

    def request(port, path, body=None):
        calls.append((path, clock.value))
        if path == "/metrics":
            return f"vllm:generation_tokens_total {100 + 20 * clock.value}\n"
        if path == "/stop_profile":
            with gzip.open(output / "worker.pt.trace.json.gz", "wt") as stream:
                json.dump({"traceEvents": []}, stream)
        return ""

    monkeypatch.setattr(profile, "request", request)
    args = dict(
        repository=tmp_path,
        output=output,
        benchmark_log=log,
        port=8897,
        server_pid=123,
        settle_seconds=3,
        capture_seconds=2,
        timeout_seconds=10,
        poll_seconds=1,
        prefix="native-steady",
    )
    return args, clock, calls


def test_capture_waits_for_warmup_and_active_workload(environment):
    args, clock, calls = environment
    report = profile.capture(**args)
    assert calls == [
        ("/metrics", 0),
        ("/metrics", 3),
        ("/health", 3),
        ("/start_profile", 3),
        ("/stop_profile", 5),
    ]
    assert report["generation_tokens_during_settle"] == 60
    assert report["capture_started_unix"] == 3
    assert report["capture_stopped_unix"] == 5
    assert report["status"] == "captured"
    assert report["analysis_pending"] is True
    saved = json.loads((args["output"] / "profile-control.json").read_text())
    assert saved["traces"][0]["name"] == "worker.pt.trace.json.gz"
    assert saved["traces"][0]["bytes"] > 0


def test_warmup_without_measured_phase_never_starts_profiler(environment):
    args, clock, calls = environment
    args["benchmark_log"].write_text("Phase warmup (warmup) complete\n")
    with pytest.raises(TimeoutError, match="Measured agentic phase"):
        profile.capture(**args)
    assert calls == []
    saved = json.loads((args["output"] / "profile-control.json").read_text())
    assert saved["status"] == "failed"


def test_stalled_generation_does_not_produce_a_profile(environment, monkeypatch):
    args, clock, calls = environment
    monkeypatch.setattr(
        profile,
        "request",
        lambda port, path, body=None: "vllm:generation_tokens_total 100\n",
    )
    with pytest.raises(RuntimeError, match="No generation progress"):
        profile.capture(**args)
    assert not list(args["output"].glob("*.trace.json*"))


def test_server_failure_still_stops_profiling(environment, monkeypatch):
    args, clock, calls = environment

    def alive(pid, signal):
        if clock.value >= 4:
            raise ProcessLookupError("server exited")

    monkeypatch.setattr(profile.os, "kill", alive)
    with pytest.raises(ProcessLookupError):
        profile.capture(**args)
    assert calls[-2:] == [("/start_profile", 3), ("/stop_profile", 4)]
    saved = json.loads((args["output"] / "profile-control.json").read_text())
    assert saved["status"] == "failed"
    assert "server exited" in saved["error"]


def test_trace_output_cannot_escape_repository(environment):
    args, clock, calls = environment
    args["output"] = args["repository"].parent / "outside-profile"
    with pytest.raises(ValueError, match="inside the allowed repository"):
        profile.capture(**args)
    assert not args["output"].exists()
    assert calls == []


def test_generation_counter_sums_labels_and_rejects_missing_values():
    assert profile.generation_tokens(
        '# HELP vllm:generation_tokens_total Generated tokens\n'
        'vllm:generation_tokens_total{engine="0"} 100\n'
        'vllm:generation_tokens_total{engine="1"} 37\n'
        'other_counter 999\n'
    ) == 137
    with pytest.raises(ValueError, match="absent"):
        profile.generation_tokens("other_counter 999\n")

def test_partial_phase_marker_waits_for_the_remaining_log_bytes(environment, monkeypatch):
    args, clock, calls = environment
    args["benchmark_log"].write_text("Phase profil")

    def append_after_wait(seconds):
        clock.sleep(seconds)
        if clock.value == 1:
            with args["benchmark_log"].open("a") as stream:
                stream.write("ing (profiling) started\n")

    monkeypatch.setattr(profile.time, "sleep", append_after_wait)
    profile.capture(**args)
    assert calls[-2:] == [("/start_profile", 4), ("/stop_profile", 6)]
