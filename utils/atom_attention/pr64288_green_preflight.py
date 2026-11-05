"""Syntax, shell parsing and real routing-evidence fixture checks."""
import ast
import json
from pathlib import Path
import subprocess
import tempfile
import yaml

root = Path("utils/atom_attention")
for filename in (
    "native_vllm_gpu_validate.py", "pr64288_native_cache_numerical.py",
    "pr64288_review_diagnostics.py", "pr64288_matched_serve.py",
    "pr64288_model_accuracy.py", "pr64288_niah_validation.py",
    "pr64288_verify_evidence.py", "pr64288_bench_client_preflight.py",
    "review_serving_observer.py",
):
    ast.parse((root / filename).read_text())
subprocess.run(["bash", "-n", str(root / "pr64288_green_validation.sh")], check=True)
for path in Path(".github/workflows").glob("pr64288-green-*.yml"):
    workflow = yaml.safe_load(path.read_text())
    for job in workflow["jobs"].values():
        for step in job["steps"]:
            if "run" in step:
                subprocess.run(["bash", "-n"], input=step["run"], text=True, check=True)
source = root / "pr64288_matched_serve.py"
function = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "execution_proof")
scope = {"json": json, "MARKER": "PR64288_SERVING "}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), scope)
events = scope["execution_proof"](
    root / "fixtures/pr64288-concurrent-worker-events.log",
    "short_context_c8", False, False)
assert len(events) == 16
with tempfile.TemporaryDirectory() as temporary:
    fixture = Path(temporary, "cutoff.log")
    for query, expected_pass in ((256, True), (1152, False), (4224, False)):
        adjusted = [dict(e, phase="short_query_c8") for e in events]
        for event in adjusted:
            if event["kind"] == "context_attention":
                event.update(max_query_len=query, max_context_len=24576)
        fixture.write_text("".join("PR64288_SERVING " + json.dumps(e) + "\n" for e in adjusted))
        try:
            scope["execution_proof"](fixture, "short_query_c8", False, False)
        except RuntimeError as error:
            assert not expected_pass and "intended cutoff region" in str(error)
        else:
            assert expected_pass
print("Syntax, shell and actual cutoff-evidence checks passed")
