"""Execution evidence for native FP8 validation; not an attention backend."""

import json
import os
from pathlib import Path
import re

MARKER = "NATIVE_FP8_EXECUTION "
KINDS = {"projection", "cached_fmha", "causal_fmha"}
_REGISTERED = False


def validate_execution_log(text, workers, native_sha, *, require_causal, phase):
    if workers <= 0 or not re.fullmatch(r"[0-9a-f]{40}", native_sha):
        raise ValueError("Expected positive worker count and exact native revision")
    if phase not in {"eval", "replay"}:
        raise ValueError("Expected eval or replay execution phase")
    stress_modes = set()
    required = {"projection", "cached_fmha"}
    if require_causal:
        required.add("causal_fmha")
    by_rank = {}
    for line in text.splitlines():
        if MARKER not in line:
            continue
        event = json.loads(line.split(MARKER, 1)[1])
        if event.get("phase") != phase:
            continue
        if event.get("native_sha") != native_sha:
            raise ValueError("Execution evidence contains a different native revision")
        stress = event.get("accuracy_stress_routing")
        if type(stress) is not bool or (phase == "replay" and stress):
            raise ValueError("Invalid accuracy-stress routing for this phase")
        stress_modes.add(stress)
        if len(stress_modes) > 1:
            raise ValueError("Execution evidence mixes routing configurations")
        kind = event.get("kind")
        if kind not in KINDS:
            raise ValueError("Unrecognized native FP8 execution event")
        if event.get("dtype") != "torch.float8_e4m3fn" or event.get("nonempty") is not True:
            raise ValueError("Execution evidence is not nonempty OCP FP8")
        if event.get("graph_capture") is not False:
            raise ValueError("Graph capture alone is not evaluation execution")
        rank, pid = event.get("rank"), event.get("pid")
        if type(rank) is not int or type(pid) is not int or pid <= 0:
            raise ValueError("Execution evidence is missing worker identity")
        record = by_rank.setdefault(rank, {"pid": pid, "kinds": set()})
        if record["pid"] != pid:
            raise ValueError("Worker restarted during the measured phase")
        record["kinds"].add(kind)
    if set(by_rank) != set(range(workers)):
        raise ValueError("Not every expected worker executed native FP8")
    if len({record["pid"] for record in by_rank.values()}) != workers:
        raise ValueError("Worker execution identities are not distinct")
    if any(not required.issubset(record["kinds"]) for record in by_rank.values()):
        raise ValueError("A worker did not execute every required native FP8 operation")
    return {
        "native_sha": native_sha,
        "phase": phase,
        "accuracy_stress_routing": next(iter(stress_modes)),
        "workers": [
            {"rank": rank, "pid": record["pid"], "kinds": sorted(record["kinds"])}
            for rank, record in sorted(by_rank.items())
        ],
    }


def register():
    """Observe native calls, with an explicitly selected evaluation routing override."""
    global _REGISTERED
    if _REGISTERED:
        return

    import functools
    import hashlib

    import torch
    import torch.distributed as distributed
    import vllm
    from aiter.ops import flydsl
    from vllm.v1.attention.ops import rocm_aiter_mla_fp8_prefill as native

    manifest = json.loads(Path(os.environ["INFERENCEX_NATIVE_MANIFEST"]).read_text())
    expected_sha = os.environ["INFERENCEX_NATIVE_VLLM_SHA"]
    if not re.fullmatch(r"[0-9a-f]{40}", expected_sha):
        raise ValueError("Native validation requires an exact source revision")
    if manifest["native_sha"] != expected_sha:
        raise ValueError("Native installation evidence has a different source revision")
    root = Path(vllm.__file__).resolve().parent.parent
    if str(root) != manifest["source_root"]:
        raise ValueError("Validation did not import the requested native installation")
    for relative, expected_hash in manifest["source_hashes"].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Native source changed after installation: {relative}")

    stress = os.environ["INFERENCEX_NATIVE_ACCURACY_STRESS"]
    if stress not in {"0", "1"}:
        raise ValueError("Native accuracy stress must be explicitly 0 or 1")
    if stress == "1":
        native.use_fp8_context = lambda query, context: query > 0 and context > 0
    phase_file = Path(os.environ["INFERENCEX_NATIVE_PHASE_FILE"])
    emitted = set()

    def record(kind, tensors):
        if kind in emitted or not phase_file.is_file():
            return
        if torch.cuda.is_current_stream_capturing():
            return
        if not all(t.dtype == torch.float8_e4m3fn and t.numel() for t in tensors):
            raise RuntimeError("Native FP8 execution observer received other dtypes")
        event = {
            "native_sha": expected_sha,
            "kind": kind,
            "phase": phase_file.read_text().strip(),
            "rank": distributed.get_rank(),
            "pid": os.getpid(),
            "dtype": str(tensors[0].dtype),
            "nonempty": True,
            "graph_capture": False,
            "accuracy_stress_routing": stress == "1",
        }
        print(MARKER + json.dumps(event, sort_keys=True), flush=True)
        emitted.add(kind)

    project = native.project

    @functools.wraps(project)
    def observed_project(*args, **kwargs):
        result = project(*args, **kwargs)
        if result[0].dtype == torch.float8_e4m3fn:
            record("projection", (args[0], args[3].weight, result[0], result[1]))
        return result

    attention = flydsl.flydsl_flash_attn_fp8_func

    @functools.wraps(attention)
    def observed_attention(q, k, v, *args, **kwargs):
        result = attention(q, k, v, *args, **kwargs)
        record("causal_fmha" if kwargs.get("causal") else "cached_fmha", (q, k, v))
        return result

    native.project = observed_project
    flydsl.flydsl_flash_attn_fp8_func = observed_attention
    _REGISTERED = True
