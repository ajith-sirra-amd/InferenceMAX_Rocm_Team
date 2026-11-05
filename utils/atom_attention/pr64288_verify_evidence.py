"""Reject missing or incomplete evidence from the standalone review workflows."""
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


def validate(root, scope):
    native = json.loads((root / "native-validation.json").read_text())
    assert native["status"] == "passed"
    assert native["native_sha"] == os.environ["NATIVE_SHA"]
    assert native["precompiled_base"] == os.environ["BASE_SHA"]
    cases = ET.parse(root / "native-kernel-tests.xml").getroot().findall(".//testcase")
    assert len(cases) == 12
    assert not any(case.find(tag) is not None for case in cases
                   for tag in ("skipped", "failure", "error"))
    if scope == "kernels":
        numerical = json.loads((root / "native-cache-numerical.json").read_text())
        assert numerical["status"] == "passed"
        assert numerical["native_source"] == os.environ["NATIVE_SHA"]
        assert len(numerical["cases"]) == 8
        assert all(case["passed"] for case in numerical["cases"])
        asm = json.loads((root / "asm/asm.json").read_text())
        assert asm["status"] == "passed" and asm["asm_metadata_present"]
    elif scope == "niah":
        from pr64288_niah_validation import validate as validate_niah
        report = json.loads((root / "niah-validation.json").read_text())
        assert report["status"] == "passed"
        assert report["candidate"] == os.environ["NATIVE_SHA"]
        assert report["base"] == os.environ["BASE_SHA"]
        validate_niah(report)
    elif scope == "serving":
        report = json.loads((root / "serving-validation.json").read_text())
        assert report["status"] == "passed"
        assert report["candidate"] == os.environ["NATIVE_SHA"]
        assert report["base"] == os.environ["BASE_SHA"]
        expected = {"short_context_c8": 32, "short_query_c8": 32,
                    "cached_c8": 64, "cached_c48": 192, "long_cached_c48": 192}
        assert set(report["arms"]) == {"upstream_base", "candidate_off", "candidate_on"}
        for arm, results in report["arms"].items():
            assert len(results) == 5
            assert {case["case"]: case["completed"] for case in results} == expected
            expected_sha = os.environ["BASE_SHA"] if arm == "upstream_base" else os.environ["NATIVE_SHA"]
            assert all(case["native_sha"] == expected_sha for case in results)
            assert sum(case["completed"] for case in results) == 512
    else:
        raise ValueError("Unknown scope: " + scope)
    print("Verified complete evidence for", scope)


if __name__ == "__main__":
    validate(Path(sys.argv[1]), sys.argv[2])
