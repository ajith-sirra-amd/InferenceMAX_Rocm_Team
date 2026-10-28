"""Validate native optimized execution evidence after evaluation or replay."""

import argparse
import json
from pathlib import Path

from . import validate_execution_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", required=True, type=int)
    parser.add_argument("--native-sha", required=True)
    parser.add_argument("--phase", required=True, choices=("eval", "replay"))
    parser.add_argument("--require-causal", action="store_true")
    args = parser.parse_args()
    proof = validate_execution_log(
        args.server_log.read_text(errors="replace"),
        args.workers,
        args.native_sha,
        require_causal=args.require_causal,
        phase=args.phase,
    )
    args.output.write_text(json.dumps(proof, indent=2) + "\n")
    print("NATIVE_FP8_EXECUTION_PROOF_PASS", json.dumps(proof), flush=True)


if __name__ == "__main__":
    main()
