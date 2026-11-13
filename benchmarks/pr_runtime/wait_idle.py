"""Reuse Ajith's sysfs/KFD idle gate with a bounded correctness-job wait."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "quality" / "diagnostic"))
from wait_for_idle import wait_for_idle


if __name__ == "__main__":
    raise SystemExit(wait_for_idle(Path(sys.argv[1]), timeout=900, interval=300))
