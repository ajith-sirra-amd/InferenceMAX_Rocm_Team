"""Reject skipped or unexecuted new PR correctness cases."""
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


def validate(path, pr):
    cases = list(ET.parse(path).getroot().iter("testcase"))
    selected = [case for case in cases if (
        "context_projection_checkpoint_load_and_reload_preserve_output_columns" in case.get("name", "")
        if pr == 67372 else "norm_cache" in case.get("name", "")
    )]
    errors = []
    if not cases or not selected:
        errors.append("New PR cases did not execute")
    for case in cases:
        if case.find("failure") is not None or case.find("error") is not None:
            errors.append("Failed case: " + case.get("name", ""))
    for case in selected:
        if case.find("skipped") is not None:
            errors.append("New PR case skipped: " + case.get("name", ""))
    return dict(status="failed" if errors else "passed", cases=len(cases),
                selected_cases=len(selected), errors=errors,
                limits="Component suite only; native model/graph lifecycle, accuracy and E2E remain unqualified.")


if __name__ == "__main__":
    result = validate(Path(sys.argv[1]), int(sys.argv[2]))
    Path(sys.argv[3]).write_text(json.dumps(result, indent=2) + "\n")
    raise SystemExit(result["status"] != "passed")
