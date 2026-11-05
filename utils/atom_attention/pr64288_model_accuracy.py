"""Normal-routing GSM8K and cached long-context retrieval for the native candidate."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from pr64288_eval_monitor import run_evaluation
import urllib.request

import pr64288_matched_serve as serving

LM_EVAL_REV = "b315ef3b05176acc9732bb7fdec116abe1ecc476"
TASK = Path("/repo/upstream/InferenceX/infx/evals/gsm8k.yaml")


def completion(prompt, max_tokens):
    data = {
        "model": "moonshotai/Kimi-K3", "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0, "seed": 42, "stream": False,
    }
    req = urllib.request.Request(serving.URL + "/v1/completions",
        data=json.dumps(data).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as response:
        result = json.load(response)
    if not result.get("choices"):
        raise RuntimeError("Empty completion response")
    return result


def niah(arm, env, log, proc):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(serving.MODEL), trust_remote_code=True)
    directory = serving.EVIDENCE / arm
    phase = Path(env["REVIEW_PHASE_FILE"])
    output = []
    filler = (
        "The archive contains notes on river levels, forest paths, and weather observations. "
        "Each entry describes an ordinary day. The keeper recorded the location and date, "
        "checked the instruments, and stored the report on a wooden shelf. "
        "These background observations do not specify any access codes.\n"
    )
    tokens = tokenizer.encode(filler, add_special_tokens=False)
    for length in (16384, 32768):
        for depth in (0.1, 0.5, 0.9):
            for repeat in range(2):
                code = str(700001 + length + int(depth * 1000) + repeat * 731)
                key = f"archive-{length}-{int(depth*100)}-{repeat}"
                needle = tokenizer.encode(
                    f"\nThe six-digit access code for {key} is {code}. "
                    "Keep this exact code for later retrieval.\n", add_special_tokens=False)
                body = (tokens * (length // len(tokens) + 1))[:length]
                position = int(length * depth)
                body[position:position+len(needle)] = needle
                content = (
                    "Read this archive and retrieve the requested access code.\n"
                    + tokenizer.decode(body)
                    + f"\nWhat is the six-digit access code for {key}? "
                    "Return only #### followed by that code, without explanation."
                )
                full = tokenizer.apply_chat_template(
                    [{"role": "user", "content": content}],
                    tokenize=True, add_generation_prompt=True,
                    thinking=False, enable_thinking=False,
                )
                if hasattr(full, "tolist"):
                    full = full.tolist()
                if not isinstance(full, list) or not all(isinstance(x, int) for x in full):
                    raise RuntimeError("Expected a token ID list from the chat template")
                prefix = full[:-1024]
                if len(prefix) <= 8192:
                    raise RuntimeError("Retrieval context does not exceed the native cutoff")
                phase.unlink(missing_ok=True)
                assert serving.request("/reset_prefix_cache", method="POST") == {"success": True}
                completion(prefix, 1)
                name = f"niah-{length}-{int(depth*100)}-{repeat}"
                phase.write_text(name)
                response = completion(full, 512)
                text = response["choices"][0]["text"]
                match = re.findall(r"####\s*(\d{6})", text)
                correct = len(match) == 1 and match[0] == code
                proof = serving.execution_proof(
                    log, name, arm == "candidate_on", True)
                entry = {
                    "case": name, "depth": depth, "expected": code,
                    "text": text, "correct": correct,
                    "prompt_tokens": len(full), "cached_prefix_tokens_requested": len(prefix),
                    "suffix_tokens": 1024, "usage": response.get("usage"),
                    "finish_reason": response["choices"][0].get("finish_reason"),
                    "prompt_sha256": hashlib.sha256(json.dumps(full).encode()).hexdigest(),
                    "execution": proof,
                }
                output.append(entry)
                (directory / "niah.json").write_text(json.dumps(output, indent=2))
                print("NIAH_RESULT", json.dumps({k: entry[k] for k in
                      ("case", "correct", "prompt_tokens", "cached_prefix_tokens_requested")}), flush=True)
    return output



def acceptance_snapshot(path):
    with urllib.request.urlopen(serving.URL + "/metrics", timeout=15) as response:
        text = response.read().decode()
    path.write_text(text)
    values = {}
    for suffix in ("drafts", "draft_tokens", "accepted_tokens"):
        name = "vllm:spec_decode_num_" + suffix + "_total"
        matches = re.findall(r"^" + re.escape(name) + r"(?:\{[^\n]*\})?\s+([0-9.eE+-]+)", text, re.MULTILINE)
        if not matches:
            raise RuntimeError("Missing MTP counter: " + name)
        values[suffix] = sum(map(float, matches))
    return values

def gsm8k(arm, env, log, proc):
    directory = serving.EVIDENCE / arm
    phase = Path(env["REVIEW_PHASE_FILE"])
    assert serving.request("/reset_prefix_cache", method="POST") == {"success": True}
    phase.write_text("gsm8k")
    acceptance_before = acceptance_snapshot(directory / "gsm8k-metrics-before.txt")
    eval_root = Path(os.environ["REVIEW_EVAL_ROOT"])
    eval_python = eval_root / ".venv/bin/python"
    eval_env = dict(env, VLLM_PLUGINS="", OPENAI_API_KEY="EMPTY",
                    PYTHONPATH=str(eval_root / "patches"))
    eval_env["REVIEW_GSM8K_TRACE"] = str(directory / "gsm8k-request-events.jsonl")
    output = directory / "gsm8k"
    args = [
        eval_python, "-m", "lm_eval", "--model", "local-chat-completions",
        "--apply_chat_template", "--tasks", TASK, "--output_path", output,
        "--log_samples", "--seed", "0,1234,1234,1234",
        "--model_args",
        f"model=moonshotai/Kimi-K3,base_url={serving.URL}/v1/chat/completions,"
        "api_key=EMPTY,eos_string=</s>,max_retries=0,num_concurrent=48,"
        "timeout=1800,tokenized_requests=False,max_length=65536",
        "--gen_kwargs", "max_tokens=16384,temperature=0,top_p=1",
    ]
    (directory / "gsm8k-command.json").write_text(json.dumps(list(map(str, args)), indent=2))
    run_evaluation(args, env=eval_env, directory=directory, server=proc,
                   server_log=log, stop=serving.stop, arm=arm)
    results = list(output.rglob("results_*.json"))
    if len(results) != 1:
        raise RuntimeError("Expected exactly one GSM8K result")
    raw = json.loads(results[0].read_text())
    acceptance_after = acceptance_snapshot(directory / "gsm8k-metrics-after.txt")
    acceptance_delta = {k: acceptance_after[k] - acceptance_before[k] for k in acceptance_before}
    if acceptance_delta["accepted_tokens"] <= 0:
        raise RuntimeError("No accepted draft tokens during GSM8K")
    scores = raw["results"]["gsm8k"]
    samples = raw.get("n-samples", {}).get("gsm8k", {})
    if samples.get("effective") != 1319:
        raise RuntimeError("GSM8K did not evaluate the complete test split")
    strict = scores["exact_match,strict-match"]
    flexible = scores["exact_match,flexible-extract"]
    thresholds = json.loads(Path("/repo/upstream/InferenceX/infx/evals/thresholds.yaml").read_text())
    minimum = thresholds["models"].get("kimik3", {}).get("gsm8k", thresholds["default"]["gsm8k"])
    # Normal GSM8K prompts can stay below the cutoff; report the actual observed
    # operations without pretending GSM8K proves cached FP8 coverage.
    proof = serving.execution_proof(log, "gsm8k", arm == "candidate_on", None)
    summary = {
        "strict": strict, "flexible": flexible, "n": 1319,
        "repository_threshold": minimum, "threshold_passed": strict >= minimum,
        "normal_routing": True, "execution": proof,
        "task_sha256": hashlib.sha256(TASK.read_bytes()).hexdigest(),
        "lm_eval_revision": LM_EVAL_REV, "mtp_counters": acceptance_delta,
    }
    (directory / "gsm8k-summary.json").write_text(json.dumps(summary, indent=2))
    print("GSM8K_RESULT", json.dumps(summary), flush=True)
    return summary


def exercise(arm, env, log, proc):
    results = {"bench": []}
    destination = serving.EVIDENCE / arm / "matched-results.json"
    # Collect the reviewer's long-context accuracy evidence before the longer
    # serving cases. All cases and the full GSM8K split remain required.
    if arm != "candidate_off":
        results["niah"] = niah(arm, env, log, proc)
        destination.write_text(json.dumps(results, indent=2))
    for case in serving.CASES:
        results["bench"].append(serving.bench(arm, case, env, log, proc))
        destination.write_text(json.dumps(results, indent=2))
    if arm != "candidate_off":
        results["gsm8k"] = gsm8k(arm, env, log, proc)
        destination.write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    report = {"status": "running", "arms": {}, "candidate": serving.CANDIDATE,
              "base": serving.BASE, "scope": "Normal-routing model accuracy; live MTP"}
    try:
        serving.main(report, exercise=exercise)
        base, candidate = report["arms"]["upstream_base"], report["arms"]["candidate_on"]
        if [x["prompt_sha256"] for x in base["niah"]] != [x["prompt_sha256"] for x in candidate["niah"]]:
            raise RuntimeError("NIAH arms received different prompts")
        if not all(x["correct"] for arm in (base, candidate) for x in arm["niah"]):
            raise RuntimeError("One or more NIAH retrieval answers were incorrect")
        if not all(arm["gsm8k"]["threshold_passed"] for arm in (base, candidate)):
            raise RuntimeError("Repository GSM8K threshold not met")
        report["gsm8k_strict_delta_pp"] = 100 * (candidate["gsm8k"]["strict"] - base["gsm8k"]["strict"])
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        (serving.EVIDENCE / "model-accuracy.json").write_text(json.dumps(report, indent=2))
        print("MODEL_ACCURACY", json.dumps(report), flush=True)
