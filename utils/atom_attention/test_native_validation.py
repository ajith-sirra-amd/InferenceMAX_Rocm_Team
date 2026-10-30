"""Check native validation setup and execution-evidence rejection paths."""

import io
import json
from unittest.mock import patch
from urllib.error import HTTPError

import pytest

from utils.atom_attention.native_validation_plugin.native_fp8_validation import (
    validate_execution_log,
)
from utils.atom_attention.native_vllm_gpu_validate import resolve_exact_rocm_wheel

SHA = "a" * 40


@pytest.fixture
def runtime():
    return {
        "hip": "10.0.0",
        "torch": "2.13.0+rocm10.0.0",
        "triton": "3.8.0+git456",
        "tags": ["cp312-cp312-manylinux_2_39_x86_64"],
    }


def wheel_response(url, timeout):
    package = url.split("/")[-2]
    versions = {"torch": "2.13.0", "triton": "3.8.0+git456", "vllm": "0.33.0"}
    wheel_version = "2.13.0+rocm10.0.0" if package == "torch" else versions[package]
    filename = f"{package}-{wheel_version}-cp312-cp312-manylinux_2_39_x86_64.whl"
    return io.BytesIO(json.dumps([{
        "package_name": package,
        "variant": "rocm100",
        "version": versions[package],
        "python_tag": "cp312",
        "abi_tag": "cp312",
        "platform_tag": "manylinux_2_39_x86_64",
        "path": f"../../../{SHA}/rocm100-wheels/{filename.replace('+', '%2B')}",
    }]).encode())


def test_exact_wheel_resolution(runtime):
    with patch("urllib.request.urlopen", side_effect=wheel_response):
        url, packages = resolve_exact_rocm_wheel(SHA, runtime)
    assert url == (
        f"https://wheels.vllm.ai/rocm/{SHA}/rocm100-wheels/"
        "vllm-0.33.0-cp312-cp312-manylinux_2_39_x86_64.whl"
    )
    assert packages["torch"]["version"] == "2.13.0"
    assert packages["torch"]["wheel_version"] == "2.13.0+rocm10.0.0"


def test_rocm_release_is_not_the_hip_component_version(runtime):
    """ROCm 10 ships HIP 7.15; selecting rocm715 returns a real-world 404."""
    def response(url, timeout):
        if "/rocm100/" not in url:
            raise HTTPError(url, 404, "Not Found", {}, None)
        return wheel_response(url, timeout)

    with patch("urllib.request.urlopen", side_effect=response):
        url, _ = resolve_exact_rocm_wheel(SHA, runtime | {"hip": "7.15.26333"})
    assert "/rocm100-wheels/" in url


@pytest.mark.parametrize("failure", ["metadata_version", "filename_abi"])
def test_wheel_filename_must_match_metadata_and_runtime(runtime, failure):
    def response(url, timeout):
        entries = json.load(wheel_response(url, timeout))
        if entries[0]["package_name"] == "torch":
            if failure == "metadata_version":
                entries[0]["version"] = "2.12.0"
            else:
                entries[0]["path"] = entries[0]["path"].replace("cp312-cp312", "cp311-cp311")
        return io.BytesIO(json.dumps(entries).encode())

    with patch("urllib.request.urlopen", side_effect=response):
        with pytest.raises(RuntimeError):
            resolve_exact_rocm_wheel(SHA, runtime)


@pytest.mark.parametrize("change", [
    {"torch": "2.12.0"},
    {"torch": "2.13.0+rocm10.0.1"},
    {"triton": "3.7.0"},
    {"triton": "3.8.0"},  # Dropping the wheel's git revision must still fail.
    {"tags": ["cp311-cp311-manylinux_2_39_x86_64"]},
    {"hip": None},
])
def test_incompatible_runtime_is_rejected(runtime, change):
    with patch("urllib.request.urlopen", side_effect=wheel_response):
        with pytest.raises(RuntimeError):
            resolve_exact_rocm_wheel(SHA, runtime | change)


def test_missing_build_is_not_replaced(runtime):
    missing = HTTPError("https://wheels.vllm.ai/missing", 404, "missing", {}, None)
    with patch("urllib.request.urlopen", side_effect=missing):
        with pytest.raises(HTTPError) as error:
            resolve_exact_rocm_wheel(SHA, runtime)
    assert error.value.code == 404


@pytest.fixture
def events():
    return [{
        "native_sha": SHA, "phase": "eval", "rank": rank, "pid": 1000 + rank,
        "kind": kind, "dtype": "torch.float8_e4m3fn", "nonempty": True,
        "graph_capture": False, "accuracy_stress_routing": True,
    } for rank in range(8) for kind in ("projection", "cached_fmha", "causal_fmha")]


def log_text(events):
    return "\n".join("Worker NATIVE_FP8_EXECUTION " + json.dumps(e) for e in events)


def test_eight_workers_execute_every_required_operation(events):
    result = validate_execution_log(
        log_text(events), 8, SHA, require_causal=True, phase="eval",
    )
    assert result["accuracy_stress_routing"] is True
    assert [worker["rank"] for worker in result["workers"]] == list(range(8))
    assert result["workers"][0]["kinds"] == ["cached_fmha", "causal_fmha", "projection"]


@pytest.mark.parametrize("failure", [
    "missing_worker", "no_causal", "warmup_only", "wrong_revision",
    "bf16", "capture_only", "restart", "stress_replay",
])
def test_incomplete_or_inapplicable_execution_is_rejected(events, failure):
    phase = "eval"
    if failure == "missing_worker":
        events = events[:-3]
    elif failure == "no_causal":
        events = [e for e in events if e["kind"] != "causal_fmha"]
    elif failure == "warmup_only":
        events = [dict(e, phase="startup") for e in events]
    elif failure == "wrong_revision":
        events[0]["native_sha"] = "b" * 40
    elif failure == "bf16":
        events[0]["dtype"] = "torch.bfloat16"
    elif failure == "capture_only":
        events[0]["graph_capture"] = True
    elif failure == "restart":
        events.append(dict(events[0], pid=9999))
    else:
        phase = "replay"
        events = [dict(e, phase=phase) for e in events]
    with pytest.raises(ValueError):
        validate_execution_log(log_text(events), 8, SHA, require_causal=True, phase=phase)
