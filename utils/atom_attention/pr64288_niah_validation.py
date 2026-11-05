"""Standalone NIAH validation with the established prompts and acceptance criteria."""
import json
from pr64288_model_accuracy import niah
import pr64288_matched_serve as serving


def validate(report):
    base = report["arms"]["upstream_base"]
    candidate = report["arms"]["candidate_on"]
    expected = {
        f"niah-{length}-{depth}-{repeat}"
        for length in (16384, 32768)
        for depth in (10, 50, 90)
        for repeat in range(2)
    }
    for arm in (base, candidate):
        if len(arm) != 12 or {item["case"] for item in arm} != expected:
            raise RuntimeError("NIAH did not execute all 12 cases")
        if not all(item["correct"] for item in arm):
            raise RuntimeError("One or more NIAH retrieval answers were incorrect")
    if [item["prompt_sha256"] for item in base] != [
        item["prompt_sha256"] for item in candidate
    ]:
        raise RuntimeError("NIAH arms received different prompts")
    for item in candidate:
        for kind in ("fp8_projection", "fp8_fmha"):
            if {event["rank"] for event in item["execution"]
                if event["kind"] == kind} != set(range(8)):
                raise RuntimeError("NIAH lacks all-rank optimized-path evidence")


if __name__ == "__main__":
    report = {
        "status": "running", "arms": {}, "candidate": serving.CANDIDATE,
        "base": serving.BASE,
        "scope": "Normal-routing NIAH above 8K; live MTP; unchanged 12-case workload",
    }
    try:
        serving.main(report, arms=(
            ("upstream_base", serving.BASE, "0"),
            ("candidate_on", serving.CANDIDATE, "1"),
        ), exercise=niah)
        validate(report)
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        (serving.EVIDENCE / "niah-validation.json").write_text(
            json.dumps(report, indent=2))
        print("NIAH_VALIDATION", json.dumps(report), flush=True)
