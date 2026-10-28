"""Accuracy-only FP8 routing and evidence for real evaluation calls."""
import json
import os

_STRESS = "ATOM_FP8_ACCURACY_STRESS"
_MARKER = "[ATOM-FP8-ACCURACY-CALL]"
_reported_pids = set()


def use_fp8_context(max_query_len, max_seq_len):
    if max_query_len <= 0 or max_seq_len <= 0:
        return False
    return os.environ.get(_STRESS, "0") == "1" or (
        max_query_len >= 512 and max_seq_len >= 8192
    )


def record_accuracy_call(q8, kv8, *, fp8_projection):
    if os.environ.get(_STRESS, "0") != "1":
        return
    pid = os.getpid()
    if pid in _reported_pids:
        return
    phase_file = os.environ.get("ATOM_FP8_ACCURACY_ACTIVE_FILE", "")
    if not phase_file or not os.path.isfile(phase_file):
        return
    payload = {
        "pid": pid, "fp8_projection": fp8_projection,
        "q_dtype": str(q8.dtype), "k_dtype": str(kv8[0].dtype), "v_dtype": str(kv8[1].dtype),
        "q_tokens": q8.shape[0], "kv_tokens": kv8[0].shape[0],
        "heads": q8.shape[1], "phase": "gsm8k_evaluation",
    }
    _validate_record(payload)
    print(_MARKER + " " + json.dumps(payload), flush=True)
    _reported_pids.add(pid)


def _validate_record(payload):
    if payload.get("fp8_projection") is not True:
        raise RuntimeError("GSM8K FP8 proof requires actual native FP8 projection")
    if any(payload.get(name) != "torch.float8_e4m3fn" for name in ("q_dtype", "k_dtype", "v_dtype")):
        raise RuntimeError("GSM8K FP8 proof requires E4M3 Q/K/V")
    if min(payload.get("q_tokens", 0), payload.get("kv_tokens", 0), payload.get("heads", 0)) <= 0:
        raise RuntimeError("Empty attention call is not FP8 execution proof")
    if payload.get("phase") != "gsm8k_evaluation":
        raise RuntimeError("FP8 proof was not recorded during GSM8K")


def validate_accuracy_log(log, workers):
    records = {}
    decoder = json.JSONDecoder()
    for line in log.splitlines():
        if _MARKER not in line:
            continue
        payload, _ = decoder.raw_decode(line.split(_MARKER, 1)[1].lstrip())
        _validate_record(payload)
        records[payload["pid"]] = payload
    if len(records) != workers:
        raise RuntimeError(f"Expected {workers} FP8 workers during GSM8K, observed {len(records)}")
    return {"workers": workers, "records": list(records.values()),
            "scope": "native FP8 projection and FP8 FMHA invoked during evaluation; not a numerical threshold result"}
