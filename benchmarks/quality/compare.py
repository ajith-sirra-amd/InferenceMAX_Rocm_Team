"""Compare saved natural-screen arms without promoting a quality claim."""
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import screen


def read(path):
    return json.loads(path.read_text())


def acceptance(directory, summary):
    """Recompute AL and settling from raw samples, checking every series transition."""
    from native_spec_decode import READERS, increase, parse
    before = (directory / "metrics-before.prom").read_text()
    initial = screen.snapshot(before, parse)
    if any(value for name in screen.GAUGES for value in initial[name].values()):
        raise ValueError("Active traffic in raw pre-screen metrics")
    files = sorted(directory.glob("metrics-after-*.prom"))
    if len(files) < 4 or [p.name for p in files] != [f"metrics-after-{i:02d}.prom" for i in range(len(files))]:
        raise ValueError("Missing/noncontiguous raw settling samples")
    previous, last, stable = None, initial, 0
    for path in files:
        after = path.read_text()
        current = screen.snapshot(after, parse)
        screen.compare_series(initial, current)
        screen.compare_series(last, current)
        counts = READERS["vllm"](increase({"server": before}, {"server": after}))
        active = sum(value for name in screen.GAUGES for value in current[name].values())
        stable = stable + 1 if counts is not None and counts == previous and active == 0 else 0
        previous, last = counts, current
    if stable < 3 or previous is None:
        raise ValueError("Raw counters/gauges did not settle")
    accepted, drafts = previous
    if not all(math.isfinite(value) for value in previous) or not (drafts > 0 and 0 <= accepted <= 6 * drafts):
        raise ValueError("Invalid raw accepted/drafts delta")
    result = 1 + accepted / drafts
    for key, expected in (("accepted_draft_tokens", accepted), ("verification_steps", drafts),
                          ("acceptance_length", result)):
        actual = summary.get(key)
        if type(actual) not in (int, float) or not math.isfinite(actual) or actual != expected:
            raise ValueError("Summary differs from finite raw counters: " + key)
    if summary.get("counters_settled") is not True:
        raise ValueError("Summary does not confirm settled counters")
    return result


def compare(baseline, candidate):
    _, fixture = screen.assets()
    expected_answers = {row["id"]: screen.answer(row["answer"]) for row in fixture}
    arms = []
    for path, expected in ((baseline, "baseline"), (candidate, "moe_front_dspark")):
        effective = read(path / "effective-server.json")
        if (effective["optimization"] != expected or not effective["expected_natural_config_match"]
                or effective["sa_throughput_recipe_match"] or effective["sa_config_match"]):
            raise ValueError("Missing truthful natural-config audit")
        if read(path / "optimization-validation.json")["status"] != "passed":
            raise ValueError("Worker activation audit failed")
        if not read(path / "observed-verifier.json")["expected_natural_config_match"]:
            raise ValueError("Observed verifier differs")
        summary = read(path / "quality/summary.json")
        if (summary["status"] != "collected" or summary["score_denominator"] != 48
                or summary["counters_valid"] is not True):
            raise ValueError("Incomplete natural-screen arm")
        requests = read(path / "quality/requests.json")
        model = effective["config"]["server_arguments"]["served-model-name"]
        expected_requests = [dict(id=row["id"], body=screen.prompt(row, model)) for row in fixture]
        if requests != expected_requests:
            raise ValueError("Ordered requests differ from pinned prompts/parameters")
        rows = [json.loads(line) for line in (path / "quality/responses.jsonl").read_text().splitlines()]
        responses = {row["id"]: row for row in rows}
        if len(rows) != 48 or len(responses) != 48 or set(responses) != {r["id"] for r in requests}:
            raise ValueError("Missing/duplicate response IDs")
        if set(responses) != set(expected_answers):
            raise ValueError("Responses differ from pinned question IDs")
        recomputed = 0
        for request in requests:
            row = responses[request["id"]]
            if row["request"] != request["body"]:
                raise ValueError("Request/response provenance differs")
            choice = row["response"]["choices"][0]
            message = choice["message"]
            actual = screen.answer(message.get("content") or message.get("reasoning_content") or "")
            if choice["finish_reason"] != "stop" or actual is None:
                raise ValueError("Incomplete raw response was reported as collected")
            correct = actual == expected_answers[row["id"]]
            if row["correct"] != correct or row["extracted_answer"] != actual:
                raise ValueError("Stored answer score differs from raw response")
            recomputed += correct
        if summary["correct"] != recomputed:
            raise ValueError("Summary score differs from raw responses")
        acceptance(path / "quality", summary)
        image = read(path / "image-digests.json")
        revision = (path / "code-sha.txt").read_text().strip()
        if not image or not revision:
            raise ValueError("Missing image/code provenance")
        actual_argv = list(effective["argv"])
        actual_argv[actual_argv.index("--port") + 1] = "<local-port>"
        arms.append((effective, summary, requests, responses, sorted(image), revision, actual_argv))
    a, b = arms
    if (a[0]["environment"] != b[0]["environment"]
            or a[0]["config"] != b[0]["config"] or a[2] != b[2]
            or a[4:] != b[4:]):
        raise ValueError("Natural server settings or ordered request bodies differ")
    transitions = [dict(id=key, baseline_correct=a[3][key]["correct"],
                        candidate_correct=b[3][key]["correct"],
                        baseline_answer=a[3][key]["extracted_answer"],
                        candidate_answer=b[3][key]["extracted_answer"])
                   for key in a[3]]
    baseline_only = [r["id"] for r in transitions if r["baseline_correct"] and not r["candidate_correct"]]
    candidate_only = [r["id"] for r in transitions if r["candidate_correct"] and not r["baseline_correct"]]
    al_a, al_b = a[1]["acceptance_length"], b[1]["acceptance_length"]
    return dict(status="screen_collected", baseline_correct=a[1]["correct"],
                candidate_correct=b[1]["correct"], score_denominator=48,
                baseline_acceptance_length=al_a, candidate_acceptance_length=al_b,
                acceptance_length_relative_change=al_b / al_a - 1,
                baseline_only_correct=baseline_only, candidate_only_correct=candidate_only,
                transitions=transitions, quality_qualified=False,
                review_required=bool(baseline_only) or al_b < al_a,
                limits="Matched sampled 48-question screen only. Review scored regressions and "
                "AL changes; a workflow success is collection integrity, not model-quality approval.")


if __name__ == "__main__":
    baseline, candidate, output = map(Path, sys.argv[1:])
    try:
        result = compare(baseline, candidate)
    except Exception as error:
        result = dict(status="incomplete", error=str(error), quality_qualified=False)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    raise SystemExit(result["status"] != "screen_collected")
