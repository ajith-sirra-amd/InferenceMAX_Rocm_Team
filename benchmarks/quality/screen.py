#!/usr/bin/env python3
"""C48 natural-verification screen client; never launches a server.

48 pinned GSM8K questions, scored answers, raw responses and native vLLM
acceptance counters. The exact pinned native parser is vendored alongside this
file. No package install or runtime download. Use a 390-second outer timeout.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

FAMILIES = ("vllm:spec_decode_num_accepted_tokens", "vllm:spec_decode_num_drafts")
GAUGES = ("vllm:num_requests_running", "vllm:num_requests_waiting")
PARAMS = dict(temperature=1.0, top_p=0.95, repetition_penalty=1.0,
              max_tokens=2048, stream=False, n=1, chat_template_kwargs={"thinking": True})


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def answer(text):
    """Only score a marked final numeric answer, never an arbitrary last number."""
    matches = re.findall(r"(?m)^\s*####\s*([-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*[.]?\s*$", text)
    if not matches:
        return None
    try:
        return str(Decimal(matches[-1].replace(",", "")).normalize())
    except InvalidOperation:
        return None


def prompt(row, model):
    return dict(PARAMS, model=model, seed=42 + row["index"], messages=[{
        "role": "user", "content": "Solve the following problem. End your response with the "
        "final answer on a separate line as #### <number>.\n\n" + row["question"],
    }])


def generate(row, body, endpoint):
    started, clock = now(), time.monotonic()
    record = dict(id=row["id"], started_at=started, request=body,
                  expected_answer=answer(row["answer"]), response=None,
                  http_status=None, error=None, extracted_answer=None,
                  correct=False, truncated=False, valid_answer=False)
    try:
        request = Request(endpoint + "/v1/chat/completions",
                          data=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=300) as response:
            record["http_status"] = response.status
            raw = response.read().decode("utf-8")
        record["response_text"] = raw
        decoded = record["response"] = json.loads(raw)
        if decoded.get("error"):
            raise ValueError("Server returned an error document")
        choices = decoded["choices"]
        if len(choices) != 1:
            raise ValueError("Expected exactly one response choice")
        choice = choices[0]
        message = choice["message"]
        content = message.get("content") or message.get("reasoning_content") or ""
        if not isinstance(content, str):
            raise ValueError("Response content must be text")
        record["finish_reason"] = choice.get("finish_reason")
        record["truncated"] = choice.get("finish_reason") == "length"
        record["extracted_answer"] = answer(content)
        record["valid_answer"] = (choice.get("finish_reason") == "stop"
                                  and record["extracted_answer"] is not None)
        record["correct"] = (record["valid_answer"]
                             and record["extracted_answer"] == record["expected_answer"])
    except HTTPError as error:
        record.update(http_status=error.code, response_text=error.read().decode("utf-8", "replace"),
                      error=f"HTTP {error.code}")
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
    record.update(completed_at=now(), elapsed_seconds=time.monotonic() - clock)
    return record


def assets():
    directory = Path(__file__).resolve().parent
    config = json.loads((directory / "natural.json").read_text())
    for name, expected in (("gsm8k-48.jsonl", config["dataset_sha256"]),
                           ("native_spec_decode.py", config["counter_reader"]["sha256"])):
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Pinned asset hash differs: " + name)
    rows = [json.loads(line) for line in (directory / "gsm8k-48.jsonl").read_text().splitlines()]
    if len(rows) != 48 or [r["index"] for r in rows] != list(range(48)):
        raise ValueError("Expected the pinned first 48 questions")
    if any(answer(r["answer"]) is None for r in rows):
        raise ValueError("Unscorable ground truth")
    return config, rows


def snapshot(text, parse):
    values = parse(text)
    for name in FAMILIES + GAUGES:
        if not values.get(name):
            raise ValueError("Missing metric family: " + name)
        if any(not math.isfinite(v) or v < 0 for v in values[name].values()):
            raise ValueError("Negative/nonfinite metric: " + name)
    return {name: values[name] for name in FAMILIES + GAUGES}


def compare_series(before, after):
    for name in FAMILIES + GAUGES:
        if set(before[name]) != set(after[name]):
            raise ValueError("Metric series appeared/disappeared: " + name)
        if name in FAMILIES and any(after[name][key] < value
                                   for key, value in before[name].items()):
            raise ValueError("Metric counter reset: " + name)


def serializable(values):
    return {name: [{"labels": dict(sorted(labels)), "value": value}
                   for labels, value in series.items()] for name, series in values.items()}


def capture(args, dataset=None, request_extra=None, scope=None):
    config, rows = assets() if dataset is None else dataset
    denominator=len(rows)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from native_spec_decode import READERS, increase, parse
    args.output.mkdir(parents=True, exist_ok=False)
    endpoint = args.endpoint.rstrip("/")
    metric_url = endpoint + "/metrics"
    requests = [dict(id=row["id"], body=prompt(row, args.model)) for row in rows]
    for request in requests:
        request['body'].update(request_extra or {})
    save(args.output / "requests.json", requests)
    save(args.output / "contract.json", dict(dataset_sha256=config["dataset_sha256"],
         counter_reader=config["counter_reader"], requests=denominator, concurrency=48,
         parameters=PARAMS, seed="42 + item index", metrics_endpoint=metric_url, started_at=now(),
         limits=scope or "Zero-shot, sampled 48-item screen; not official GSM8K or model-quality qualification."))
    records, samples, counter_errors = [], [], []
    accepted, drafts, settled, valid_counters = None, None, False, False

    def scrape(name):
        with urlopen(metric_url, timeout=5) as response:
            text = response.read().decode("utf-8")
        (args.output / name).write_text(text)
        return {metric_url: text}

    try:
        before = scrape("metrics-before.prom")
        initial = snapshot(before[metric_url], parse)
        save(args.output / "series-before.json", serializable(initial))
        if any(v for name in GAUGES for v in initial[name].values()):
            raise ValueError("Server has active requests before this screen")
        with (args.output / "responses.jsonl").open("x") as sink:
            with ThreadPoolExecutor(max_workers=48) as pool:
                futures = [pool.submit(generate, row, req["body"], endpoint)
                           for row, req in zip(rows, requests)]
                for future in as_completed(futures):
                    row = future.result()
                    records.append(row)
                    sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                    sink.flush()
        deadline, previous, stable, last = time.monotonic() + 60, None, 0, initial
        while time.monotonic() < deadline:
            after = scrape(f"metrics-after-{len(samples):02d}.prom")
            parsed_end = parse(after[metric_url])
            # Preserve label identities even when required series disappeared.
            save(args.output / "series-end.json", {
                name: [{"labels": dict(sorted(labels)), "value": value if math.isfinite(value) else str(value)}
                       for labels, value in parsed_end.get(name, {}).items()]
                for name in FAMILIES + GAUGES})
            current = snapshot(after[metric_url], parse)
            compare_series(initial, current)
            compare_series(last, current)
            counters = READERS["vllm"](increase(before, after))
            active = sum(v for name in GAUGES for v in current[name].values())
            stable = stable + 1 if counters is not None and counters == previous and active == 0 else 0
            samples.append(dict(at=now(), counters=counters, active_requests=active))
            previous, last = counters, current
            if stable >= 3:
                settled = True
                break
            time.sleep(2)
        if not settled:
            raise ValueError("Counters or active requests did not settle in 60 seconds")
        accepted, drafts = previous if previous is not None else (None, None)
        if accepted is None or drafts is None or not math.isfinite(accepted) or not math.isfinite(drafts):
            raise ValueError("Missing/nonfinite native counter delta")
        if not (drafts > 0 and 0 <= accepted <= 6 * drafts):
            raise ValueError("Invalid six-draft-token counter delta")
        valid_counters = True
    except Exception as error:
        counter_errors.append(f"{type(error).__name__}: {error}")
    save(args.output / "counter-samples.json", samples)
    errors = sum(row["error"] is not None for row in records)
    missing = denominator - sum(row["valid_answer"] for row in records)
    summary = dict(status="collected" if len(records) == denominator and not errors and not missing
                   and valid_counters else "incomplete", responses=len(records),
                   valid_answers=denominator - missing, correct=sum(row["correct"] for row in records),
                   score_denominator=denominator, missing_or_invalid_answers=missing,
                   truncated=sum(row["truncated"] for row in records), request_errors=errors,
                   counters_valid=valid_counters, counters_settled=settled, counter_errors=counter_errors,
                   accepted_draft_tokens=accepted, verification_steps=drafts,
                   acceptance_length=1 + accepted / drafts if valid_counters else None,
                   completed_at=now(), quality_qualified=False,
                   limits=scope or "All 48 original IDs stay in the denominator. Missing/truncated answers "
                   "make the screen incomplete. Matched seeds do not guarantee identical sampled outputs. "
                   "This is not golden SPEED-Bench AL, performance measurement or model-quality proof.")
    save(args.output / "summary.json", summary)
    print(json.dumps(summary))
    return 0 if summary["status"] == "collected" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default="moonshotai/Kimi-K3.1-Pro")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return capture(args)


if __name__ == "__main__":
    raise SystemExit(main())
