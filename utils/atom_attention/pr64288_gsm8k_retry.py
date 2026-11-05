"""Full candidate-only GSM8K retry with original baseline provenance."""
import json
import os
from pathlib import Path
import pr64288_matched_serve as serving
from pr64288_model_accuracy import gsm8k

def main():
    prior = Path(os.environ["REVIEW_PRIOR_EVIDENCE"])
    original = json.loads((prior / "model-accuracy.json").read_text())
    baseline = json.loads((prior / "upstream_base/gsm8k-summary.json").read_text())
    if (original["candidate"], original["base"]) != (serving.CANDIDATE, serving.BASE):
        raise RuntimeError("Prior evidence revisions differ")
    if baseline["n"] != 1319 or not baseline["threshold_passed"]:
        raise RuntimeError("Prior baseline GSM8K incomplete")
    report = {"status": "running", "arms": {}, "candidate": serving.CANDIDATE,
              "base": serving.BASE, "baseline_evidence_run": 41861899047,
              "prior_baseline_gsm8k": baseline, "request_timeout_seconds": 1800,
              "scope": "Full GSM8K, normal routing, live MTP; HTTP tracing only"}
    def exercise(arm, env, log, proc):
        for name in ("command.json", "source-manifest.json"):
            if json.loads((prior/"candidate_on"/name).read_text()) != json.loads((serving.EVIDENCE/arm/name).read_text()):
                raise RuntimeError("Changed candidate " + name)
        if report["models"] != original["models"]:
            raise RuntimeError("Model configuration changed")
        return {"gsm8k": gsm8k(arm, env, log, proc)}
    try:
        serving.main(report, arms=(("candidate_on", serving.CANDIDATE, "1"),), exercise=exercise)
        score = report["arms"]["candidate_on"]["gsm8k"]
        if score["task_sha256"] != baseline["task_sha256"]:
            raise RuntimeError("GSM8K task changed")
        if not score["threshold_passed"]:
            raise RuntimeError("GSM8K below repository threshold")
        report["gsm8k_strict_delta_pp"] = 100*(score["strict"]-baseline["strict"])
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        (serving.EVIDENCE/"gsm8k-retry.json").write_text(json.dumps(report, indent=2))
        print("GSM8K_RETRY", json.dumps(report), flush=True)
if __name__ == "__main__":
    main()
