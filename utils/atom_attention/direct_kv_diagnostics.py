"""Run isolated chain checks or replay a captured Kimi diagnostic."""
import os
from pathlib import Path
import subprocess
import sys

if __name__ == "__main__":
    fixture_path = os.environ.get("FAILURE_FIXTURE_PATH", "")
    if fixture_path:
        from rope_balanced_diagnostics import run
        run(fixture_path)
    else:
        root = Path(__file__).resolve().parents[2]
        plugin = root / "upstream/InferenceX/benchmarks/single_node/agentic/atom_attention"
        subprocess.run(["bash", "-n", str(plugin.parent / "kimik3_fp4_mi355x_mtp.sh")], check=True)
        for name in ("test_prefill_flag.py", "test_accuracy_evidence.py"):
            subprocess.run([sys.executable, str(plugin / "tests" / name)], check=True)
        from fp8_weight_preflight import run as run_fp8_weights
        run_fp8_weights()
        if os.environ.get("ATOM_USE_FLYDSL_FP8_CAUSAL_ATTN", "0") == "1":
            from causal_fp8_preflight import run as run_causal
            run_causal()
        print("FP8_ACCURACY_PATH_PREFLIGHT_PASS", flush=True)
