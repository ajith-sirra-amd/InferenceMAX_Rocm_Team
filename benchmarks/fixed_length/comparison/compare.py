#!/usr/bin/env python3
"""Compare two native fixed-length C48 serving results without inferring significance."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validate import finite_number, validate_result

MATCH_FIELDS = (
    "backend",
    "model_id",
    "tokenizer_id",
    "best_of",
    "request_rate",
    "burstiness",
)

RECOMMENDATIONS = [
    "One baseline/candidate pair is screening evidence only.",
    "Start with at least three independent alternating baseline/candidate pairs to assess run variance.",
    (
        "For a 2–5% improvement claim, plan 5–10 or more independent matched pairs, then reassess "
        "using observed run variance; no fixed repeat count guarantees a reliable claim."
    ),
    (
        "Balance AB/BA order, use the same GPU host, avoid concurrent GPU load, and preserve warmup, "
        "server/client commands, image digest, tokenizer, model and draft revisions, seeds, and GPU settings."
    ),
    (
        "Compare run-level means. The 192 concurrent requests are not 192 independent experiments; "
        "do not infer statistical significance from unpaired request samples."
    ),
    (
        "Verify the complete ordered prompt corpus actually sent in both measured runs. Same seed, "
        "token lengths, or four preflight prompt hashes alone do not establish prompt identity."
    ),
    "This synthetic fixed-length benchmark does not establish semantic output quality.",
]


def prompt_digest(prompts):
    if not isinstance(prompts, list) or len(prompts) != 192:
        raise ValueError("Prompt corpus must contain all 192 ordered measured prompts")
    if any(not isinstance(prompt, str) or not prompt for prompt in prompts):
        raise ValueError("Every measured prompt must be a nonempty string")
    encoded = json.dumps(prompts, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def metric(baseline, candidate, lower_is_better=False):
    change = 100 * (candidate / baseline - 1)
    return {
        "baseline": baseline,
        "candidate": candidate,
        "change_percent": change,
        "improvement_percent": -change if lower_is_better else change,
    }


def compare_results(baseline, candidate, baseline_prompts=None, candidate_prompts=None):
    report = {
        "status": "failed",
        "comparison_eligibility": "rejected",
        "errors": [],
        "recommendations": RECOMMENDATIONS,
        "significance_evaluated": False,
        "quality_evaluated": False,
    }
    validations = {
        "baseline": validate_result(baseline),
        "candidate": validate_result(candidate),
    }
    report["validation"] = {
        label: {
            key: value
            for key, value in validation.items()
            if key in ("status", "expected", "errors")
        }
        for label, validation in validations.items()
    }
    for label, validation in validations.items():
        if validation["status"] != "passed":
            report["errors"].extend(
                f"{label}: {error}" for error in validation["errors"]
            )
    if report["errors"]:
        return report

    matched = {}
    for key in MATCH_FIELDS:
        left, right = baseline.get(key), candidate.get(key)
        if left is None or right is None:
            report["errors"].append(f"Missing matchability field: {key}")
        elif key in ("backend", "model_id", "tokenizer_id") and (
            not isinstance(left, str)
            or not left
            or not isinstance(right, str)
            or not right
        ):
            report["errors"].append(f"Invalid matchability field: {key}")
        elif key == "request_rate" and any(
            value != "inf" and not finite_number(value, positive=True)
            for value in (left, right)
        ):
            report["errors"].append("Invalid request_rate")
        elif key in ("best_of", "burstiness") and any(
            not finite_number(value, positive=True) for value in (left, right)
        ):
            report["errors"].append(f"Invalid matchability field: {key}")
        elif left != right:
            report["errors"].append(f"Mismatched {key}: {left!r} versus {right!r}")
        else:
            matched[key] = left
    for label, data in (("baseline", baseline), ("candidate", candidate)):
        if not finite_number(data.get("mean_e2el_ms"), positive=True):
            report["errors"].append(f"{label}: missing or invalid native mean_e2el_ms")
    report["matched_native_fields"] = matched
    report["prompt_identity"] = {"status": "unknown"}
    if baseline_prompts is not None or candidate_prompts is not None:
        try:
            left, right = (
                prompt_digest(baseline_prompts),
                prompt_digest(candidate_prompts),
            )
            report["prompt_identity"] = {
                "status": "matched" if left == right else "mismatched",
                "baseline_sha256": left,
                "candidate_sha256": right,
                "scope": "All 192 ordered prompts supplied by caller; actual-use provenance must be reviewed.",
            }
            if left != right:
                report["errors"].append("Measured prompt corpora differ")
        except ValueError as error:
            report["errors"].append(str(error))
    if report["errors"]:
        return report

    report.update(
        status="passed",
        comparison_eligibility=(
            "workload_matched_pending_environment_review"
            if report["prompt_identity"]["status"] == "matched"
            else "screening_only"
        ),
        mean_request_e2e_ms=metric(
            baseline["mean_e2el_ms"], candidate["mean_e2el_ms"], True
        ),
        request_throughput_per_second=metric(
            validations["baseline"]["request_throughput"],
            validations["candidate"]["request_throughput"],
        ),
        output_tokens_per_second=metric(
            validations["baseline"]["output_tokens_per_second"],
            validations["candidate"]["output_tokens_per_second"],
        ),
        benchmark_duration_seconds=metric(
            baseline["duration"], candidate["duration"], True
        ),
        metric_notes=[
            "Positive improvement_percent means lower E2E/duration or higher throughput.",
            "Primary E2E uses native mean_e2el_ms, not reconstructed stream-chunk timing.",
            "Throughput is recomputed from validated completed requests/token totals and measured duration.",
            "Kernel-time reduction alone is not a request-E2E or throughput improvement.",
            (
                "Native JSON does not prove server commands, runtime image, hardware, load, warmup success, "
                "sampler seed/worker count, or profiler state match; review saved commands and run evidence."
            ),
        ],
    )
    return report


def read_json(path):
    raw = path.read_bytes()
    return json.loads(raw), {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument(
        "--baseline-prompts",
        type=Path,
        help="JSON array of all ordered measured prompt strings",
    )
    parser.add_argument(
        "--candidate-prompts",
        type=Path,
        help="JSON array of all ordered measured prompt strings",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        baseline, baseline_source = read_json(args.baseline)
        candidate, candidate_source = read_json(args.candidate)
        prompts = [
            read_json(path)[0] if path else None
            for path in (args.baseline_prompts, args.candidate_prompts)
        ]
        report = compare_results(baseline, candidate, *prompts)
        report["sources"] = {"baseline": baseline_source, "candidate": candidate_source}
    except (OSError, ValueError, TypeError) as error:
        report = {
            "status": "failed",
            "comparison_eligibility": "rejected",
            "errors": [str(error)],
        }
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
