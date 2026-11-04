"""Run and record exact-source native cached-scale GPU checks."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

evidence = Path(os.environ["REVIEW_EVIDENCE"])
expected = os.environ["NATIVE_SHA"]
source = Path("/work/vllm")
actual = subprocess.check_output(
    ["git", "-c", "safe.directory=/work/vllm", "-C", str(source), "rev-parse", "HEAD"],
    text=True,
).strip()
assert actual == expected, (actual, expected)
report = {
    "native_sha": actual,
    "status": "running",
    "scope": "Native kernel and graph regression tests; serving/model accuracy separate",
}
try:
    import torch
    import vllm

    assert Path(vllm.__file__).resolve().is_relative_to(source)
    report["runtime"] = {
        "torch": torch.__version__, "hip": torch.version.hip,
        "vllm_file": vllm.__file__,
        "device": torch.cuda.get_device_name(),
    }
    test_path = "tests/kernels/attention/test_rocm_aiter_mla_prefill_fusions.py"
    command = [
        sys.executable, "-m", "pytest", test_path, "-k", "flydsl_fp8",
        "-vv", "--junitxml=" + str(evidence / "native-cache-scale-pytest.xml"),
    ]
    proc = subprocess.run(command, cwd=source)
    report["pytest_exit_code"] = proc.returncode
    xml_path = evidence / "native-cache-scale-pytest.xml"
    if xml_path.exists():
        root = ET.parse(xml_path).getroot()
        cases = root.findall(".//testcase")
        report["tests"] = [
            {"name": t.get("name"), "passed": len(t.findall("failure") + t.findall("error") + t.findall("skipped")) == 0}
            for t in cases
        ]
        assert len(cases) == 12, len(cases)
        assert all(t["passed"] for t in report["tests"])
    else:
        raise RuntimeError("No pytest XML report")
    assert proc.returncode == 0, proc.returncode
    numerical = subprocess.run([
        sys.executable, "/repo/utils/atom_attention/pr64288_native_cache_numerical.py",
        str(evidence / "native-cache-numerical.json"),
    ], cwd=source)
    report["numerical_exit_code"] = numerical.returncode
    assert numerical.returncode == 0
    report["status"] = "passed"
except Exception as error:
    report.update(status="failed", error=str(error))
    raise
finally:
    report["source_hashes"] = {
        path: hashlib.sha256((source / path).read_bytes()).hexdigest()
        for path in (
            "vllm/v1/attention/ops/rocm_aiter_mla_fp8_quant.py",
            "vllm/v1/attention/ops/rocm_aiter_mla_fp8_prefill.py",
            "vllm/v1/attention/backends/mla/rocm_aiter_mla.py",
            "vllm/v1/attention/backends/mla/prefill/aiter_flash_attn.py",
            "tests/kernels/attention/test_rocm_aiter_mla_prefill_fusions.py",
        )
    }
    (evidence / "native-cache-scale-validation.json").write_text(json.dumps(report, indent=2))
