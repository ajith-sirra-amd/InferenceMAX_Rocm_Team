"""Validate actual runtime ASM evidence from every tensor-parallel worker."""
import json
import pathlib
import re
import sys


def verify(text, tp):
    if tp <= 0:
        raise ValueError("TP must be positive")
    expected = set(range(tp))
    dispatch = set()
    completed = set()
    compared = set()
    accuracy_failures = 0
    for line in text.splitlines():
        if "context-chunk fp8 FAILED" in line or "[FP8-VALIDATION-FAILED]" in line:
            raise ValueError("ASM runtime failure recorded")
        accuracy_failures += "[FP8-ACCURACY-FAILED]" in line
        worker = re.search(r"Worker_TP(\d+)_DCP\d+", line)
        if not worker:
            continue
        rank = int(worker.group(1))
        if "[FP8-ASM-DISPATCH]" in line:
            required = ("mode=asm", "optimized=True", "causal=False", "need_lse=True",
                        "kernel=mla_prefill_ps_asm_fwd", "reducer=mla_reduce_v1")
            if all(token in line for token in required) and all(
                re.search(rf"\b{operand}=\([^)]*\)/torch\.float8\w*", line)
                for operand in ("q", "k", "v")
            ):
                dispatch.add(rank)
        if "context-chunk fp8 OK" in line:
            completed.add(rank)
        if "[FP8-VALIDATION]" in line and "optimized_matches_current=True" in line:
            compared.add(rank)
    for name, observed in (("FP8 ASM dispatch", dispatch), ("successful context calls", completed),
                           ("optimized/ordinary ASM agreement", compared)):
        if observed != expected:
            raise ValueError(f"{name}: observed={sorted(observed)}, expected={sorted(expected)}")
    return {"fp8_asm_verified": True, "workers": sorted(dispatch),
            "accuracy_qualified": False, "accuracy_failure_events": accuracy_failures}


if __name__ == "__main__":
    try:
        result = verify(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace"),
                        int(sys.argv[2]))
    except ValueError as error:
        print(json.dumps({"fp8_asm_verified": False, "error": str(error)}))
        raise SystemExit(1)
    print(json.dumps(result, indent=2))
