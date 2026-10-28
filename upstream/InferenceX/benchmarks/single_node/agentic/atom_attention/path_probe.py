
"""Controlled prefix-sharing requests against the already-running Kimi server."""
import argparse
import json
import re
import shlex
import subprocess
import urllib.request
from pathlib import Path


def cache_hits(base_url):
    with urllib.request.urlopen(base_url + "/metrics", timeout=30) as response:
        text = response.read().decode()
    pattern = r"^vllm:prefix_cache_hits(?:_total)?(?:\{[^}]*\})?\s+([\d.eE+-]+)\s*$"
    return sum(float(x) for x in re.findall(pattern, text, re.MULTILINE))


def run_probe(model, tokenizer, port, result_dir, server_log):
    destination = Path(result_dir)
    destination.mkdir(parents=True, exist_ok=True)
    base_url = f"http://127.0.0.1:{port}"
    before = cache_hits(base_url)
    stages = []
    for prefix, suffix, output, prompts, concurrency in [
        (32768, 256, 16, 16, 8),
        (131072, 256, 8, 4, 1),
    ]:
        filename = f"path_probe_prefix{prefix}.json"
        command = [
            "vllm", "bench", "serve", "--backend", "vllm",
            "--base-url", base_url, "--endpoint", "/v1/completions",
            "--model", model, "--tokenizer", tokenizer, "--tokenizer-mode", "kimi_k3",
            "--trust-remote-code", "--dataset-name", "prefix_repetition",
            "--prefix-repetition-num-prefixes", "1",
            "--prefix-repetition-prefix-len", str(prefix),
            "--prefix-repetition-suffix-len", str(suffix),
            "--prefix-repetition-output-len", str(output),
            "--num-prompts", str(prompts), "--max-concurrency", str(concurrency),
            "--request-rate", "inf", "--seed", "42", "--num-warmups", "0",
            "--ignore-eos", "--disable-tqdm", "--save-result", "--save-detailed",
            "--result-dir", str(destination), "--result-filename", filename,
        ]
        printable = shlex.join(command)
        print("[ATOM-PATH-PROBE]", printable, flush=True)
        (destination / f"path_probe_prefix{prefix}_command.txt").write_text(
            printable + "\n", encoding="utf-8",
        )
        with (destination / f"path_probe_prefix{prefix}.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, check=True, timeout=600, stdout=log, stderr=subprocess.STDOUT)
        metrics = json.loads((destination / filename).read_text())
        if metrics.get("completed") != prompts or metrics.get("failed", 0):
            raise RuntimeError(f"Prefix{prefix} request failures: {metrics}")
        stages.append({
            "prefix_tokens": prefix, "suffix_tokens": suffix, "output_tokens": output,
            "prompts": prompts, "concurrency": concurrency,
            "completed": metrics["completed"], "failed": metrics.get("failed", 0),
            "duration_seconds": metrics.get("duration"),
        })
    after = cache_hits(base_url)
    text = Path(server_log).read_text(encoding="utf-8", errors="replace")
    def pids(pattern):
        return sorted({int(x) for x in re.findall(pattern, text)})
    causal = pids(r"\[ATOM-PREFILL-PATH\] pid=(\d+) stage=new_tokens .*fmha=BF16")
    cached = pids(r"\[ATOM-FP8-ENGAGED\] pid=(\d+) stage=cached_context")
    metadata = pids(r"\[ATOM-METADATA\] pid=(\d+) prefill_ps=disabled")
    reused = pids(r"\[ATOM-Q-REUSE\] pid=(\d+).*q_reuses=[1-9]")
    pools = re.findall(r"GPU KV cache (?:size|capacity):\s*([\d,]+) tokens", text)
    summary = {
        "model": model, "framework": "vllm", "smoke_type": "prefix_repetition",
        "stages": stages, "prefix_hit_tokens_increment": after-before,
        "gpu_kv_tokens": int(pools[-1].replace(",", "")) if pools else None,
        "bf16_causal_pids": causal, "fp8_cached_context_pids": cached,
        "disabled_asm_metadata_pids": metadata, "query_reuse_pids": reused,
        "scope": "Kernel invocation and HTTP stability; no agentic throughput or model accuracy claim.",
    }
    summary["kernel_invocation_pass"] = (
        after > before and summary["gpu_kv_tokens"] is not None
        and len(causal) >= 8 and len(cached) >= 8 and len(metadata) >= 8
        and len(reused) >= 8
    )
    (destination / "path_probe_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    print("PATH_PROBE_EVIDENCE", json.dumps(summary), flush=True)
    if not summary["kernel_invocation_pass"]:
        raise RuntimeError("Controlled requests completed but invocation gate was not fully demonstrated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--server-log", required=True)
    args = parser.parse_args()
    run_probe(args.model, args.tokenizer, args.port, args.result_dir, args.server_log)
