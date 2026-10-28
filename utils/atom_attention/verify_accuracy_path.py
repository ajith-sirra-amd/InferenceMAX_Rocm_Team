"""Require actual FP8 projection/FMHA calls during GSM8K on every worker."""
import argparse
import json
from pathlib import Path

from inferencex_atom_attention.accuracy import validate_accuracy_log

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-log", required=True)
    parser.add_argument("--workers", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    proof = validate_accuracy_log(Path(args.server_log).read_text(errors="replace"), args.workers)
    Path(args.output).write_text(json.dumps(proof, indent=2) + "\n")
    print("GSM8K_FP8_EXECUTION_PROOF_PASS", json.dumps(proof), flush=True)
