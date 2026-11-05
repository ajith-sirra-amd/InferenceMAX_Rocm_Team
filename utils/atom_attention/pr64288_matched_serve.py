"""Matched serving checks on exact upstream-base and native candidate sources."""
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path("/work/vllm")
EVIDENCE = Path(os.environ["REVIEW_EVIDENCE"])
PYTHON = ROOT / ".venv/bin/python"
CANDIDATE = os.environ["NATIVE_SHA"]
BASE = os.environ["BASE_SHA"]
PORT = int(os.environ["REVIEW_PORT"])
URL = f"http://127.0.0.1:{PORT}"
MODEL = Path("/models/Kimi-K3")
RECIPE = Path("/repo/upstream/InferenceX/benchmarks/single_node/agentic/kimik3_fp4_mi355x_mtp.sh")
MARKER = "PR64288_SERVING "
SOURCE_PATHS = [
    "vllm/v1/attention/backends/mla/rocm_aiter_mla.py",
    "vllm/v1/attention/backends/mla/prefill/aiter_flash_attn.py",
    "vllm/v1/attention/ops/rocm_aiter_mla_fp8_prefill.py",
    "vllm/v1/attention/ops/rocm_aiter_mla_fp8_quant.py",
]
# Matched TP8/DCP8 normal routing. These are short synthetic serving checks,
# not AgentX performance results or model accuracy measurements.
CASES = [
    ("short_context_c8", 8, 4096, 1024, 32, False),
    # 2 * 12288 cached tokens plus a 128-token shared tail. The observed TP8
    # hybrid-cache alignment is 12288; the remaining query is 256 tokens.
    ("short_query_c8", 8, 24704, 128, 32, False),
    ("cached_c8", 8, 16384, 1024, 64, True),
    ("cached_c48", 48, 16384, 1024, 192, True),
    ("long_cached_c48", 48, 32768, 2048, 192, True),
]


def run(command, *, env=None, capture=False):
    print("SERVING_COMMAND", json.dumps([str(x) for x in command]), flush=True)
    return subprocess.run([str(x) for x in command], cwd=ROOT, env=env,
                          check=True, text=True,
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.STDOUT if capture else None)


def git(*args, capture=False):
    return run(["git", "-c", "safe.directory=/work/vllm", *args], capture=capture)


def request(path, *, method="GET", data=None):
    req = urllib.request.Request(URL + path, method=method,
        data=None if data is None else json.dumps(data).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as response:
        body = response.read()
    return json.loads(body) if body else None


def stop(proc):
    if proc is None or proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=30)


def wait_ready(proc, log):
    deadline = time.monotonic() + 2400
    next_report = 0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("Server exited during startup: " + log.read_text(errors="replace")[-10000:])
        startup_log = log.read_text(errors="replace")
        if "OSError: [Errno 116] Stale file handle" in startup_log:
            raise RuntimeError("Compiler cache read failed during startup: " + startup_log[-10000:])
        try:
            request("/health")
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        if time.monotonic() >= next_report:
            print("SERVER_STARTING", log.read_text(errors="replace")[-1800:], flush=True)
            next_report = time.monotonic() + 60
        # Wait on process completion while polling the explicit health condition.
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    raise TimeoutError("Server health deadline exceeded")


def pin_workers(recipe, directory, env):
    begin = recipe.index("python3 - <<'CCDPY'")
    end = recipe.index("wait_for_server_ready", begin)
    snippet = recipe[begin:end].replace("python3 -", str(PYTHON) + " -")
    snippet += "\npin_workers_to_ccd\n"
    pin_env = dict(env, RESULT_DIR=str(directory), PIN_CCD="1")
    run(["bash", "-c", snippet], env=pin_env)
    mapping = directory / "ccdmap.txt"
    if not mapping.is_file() or not mapping.read_text().strip():
        raise RuntimeError("CCD map missing; refusing unmatched placement")


def execution_proof(log, phase, enabled, require_fp8):
    events = []
    decoder = json.JSONDecoder()
    for payload in log.read_text(errors="replace").split(MARKER)[1:]:
        event, _ = decoder.raw_decode(payload.lstrip())
        if event["phase"] == phase:
            events.append(event)
    if any(e["forced_routing"] or e["graph_capture"] for e in events):
        raise RuntimeError("Invalid execution evidence")
    if {e["rank"] for e in events if e["kind"] == "prefill"} != set(range(8)):
        raise RuntimeError("Missing prefill execution evidence from one or more ranks")
    fp8 = [e for e in events if e["kind"] in ("fp8_projection", "fp8_fmha")]
    if not enabled and fp8:
        raise RuntimeError("OFF arm executed FP8 path")
    if enabled and require_fp8:
        for kind in ("fp8_projection", "fp8_fmha"):
            if {e["rank"] for e in fp8 if e["kind"] == kind} != set(range(8)):
                raise RuntimeError("Missing " + kind + " evidence from one or more ranks")
    if require_fp8 is False:
        if fp8:
            raise RuntimeError("Below-cutoff case unexpectedly executed FP8")
        fallback = [e for e in events if e["kind"] == "context_attention"]
        if {e["rank"] for e in fallback} != set(range(8)):
            raise RuntimeError("Missing BF16 context-attention evidence")
        if any(e[key] != "torch.bfloat16" for e in fallback
               for key in ("dtype", "k_dtype", "v_dtype")):
            raise RuntimeError("Below-cutoff context attention did not use BF16 Q/K/V")
        if phase == "short_query_c8" and any(
            e["max_query_len"] >= 512 or e["max_context_len"] < 8192
            for e in fallback
        ):
            raise RuntimeError("Short-query workload did not reach the intended cutoff region")
        if phase == "short_context_c8" and any(e["max_context_len"] >= 8192 for e in fallback):
            raise RuntimeError("Short-context workload did not reach the intended cutoff region")
    return events


def bench_command(arm, case):
    name, conc, prefix, suffix, prompts, expected_fp8 = case
    directory = EVIDENCE / arm
    return [
        ROOT / ".venv/bin/vllm", "bench", "serve", "--backend", "vllm",
        "--base-url", URL, "--endpoint", "/v1/completions",
        "--model", "moonshotai/Kimi-K3", "--tokenizer", MODEL, "--trust-remote-code",
        "--dataset-name", "random", "--random-prefix-len", str(prefix),
        "--random-input-len", str(suffix), "--random-output-len", "128",
        "--random-range-ratio", "0", "--num-prompts", str(prompts),
        "--max-concurrency", str(conc), "--request-rate", "inf",
        "--num-warmups", "1", "--seed", "42", "--ignore-eos",
        "--save-result", "--save-detailed", "--result-dir", directory,
        "--result-filename", name + ".json", "--disable-tqdm",
    ]


def bench(arm, case, env, log, proc):
    name, conc, prefix, suffix, prompts, expected_fp8 = case
    directory = EVIDENCE / arm
    phase = name
    phase_path = Path(env["REVIEW_PHASE_FILE"])
    phase_path.unlink(missing_ok=True)
    if request("/reset_prefix_cache", method="POST") != {"success": True}:
        raise RuntimeError("Prefix cache reset did not succeed")
    command = bench_command(arm, case)

    bench_log = directory / (name + ".log")
    client = None
    try:
        with bench_log.open("w") as out:
            client = subprocess.Popen([str(x) for x in command], cwd=ROOT, env=env,
                stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
            # The 192-request 32K baseline was still making progress at 20 min.
            # Keep the workload unchanged and bound it to one hour; the workflow
            # and reservation watchdog provide independent overall cutoffs.
            deadline = time.monotonic() + 3600
            next_report = 0
            while client.poll() is None:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Benchmark deadline exceeded: {arm}/{name}")
                if proc.poll() is not None:
                    raise RuntimeError(f"Server exited during benchmark: {arm}/{name}")
                if time.monotonic() >= next_report:
                    print("SERVING_PROGRESS", json.dumps({
                        "arm": arm, "case": name,
                        "remaining_seconds": round(deadline - time.monotonic()),
                        "server_tail": log.read_text(errors="replace")[-2200:],
                    }), flush=True)
                    next_report = time.monotonic() + 60
                if not phase_path.exists() and "Starting main benchmark run..." in bench_log.read_text(errors="replace"):
                    phase_path.write_text(phase)
                try:
                    client.wait(timeout=0.05)
                except subprocess.TimeoutExpired:
                    pass
    finally:
        stop(client)
    if client.returncode:
        raise RuntimeError("Benchmark failed: " + (directory / (name + ".log")).read_text()[-6000:])
    result = json.loads((directory / (name + ".json")).read_text())
    if result["completed"] != prompts:
        raise RuntimeError(f"Request failures: {result['completed']}/{prompts}")
    if proc.poll() is not None:
        raise RuntimeError("Server exited during benchmark")
    proof = execution_proof(log, phase, arm == "candidate_on", expected_fp8)
    (directory / (name + "-execution.json")).write_text(json.dumps(proof, indent=2))
    summary = {key: result.get(key) for key in (
        "completed", "duration", "total_input_tokens", "total_output_tokens",
        "request_throughput", "output_throughput", "total_token_throughput",
        "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms", "mean_tpot_ms",
    )}
    summary.update(case=name, concurrency=conc, prefix_tokens=prefix,
                   suffix_tokens=suffix, native_sha=env["REVIEW_SOURCE_SHA"])
    print("SERVING_CASE_RESULT", json.dumps(summary), flush=True)
    return summary


def main(report, *, arms=None, exercise=None, command_transform=None, env_transform=None):
    recipe = RECIPE.read_text()
    quant = json.loads(re.search(r"--quantization-config '([^']+)'", recipe).group(1))
    dspark_root = Path("/models/models--Inferact--Kimi-K3-DSpark")
    dspark_ref = dspark_root / "refs/main"
    if dspark_ref.is_file():
        draft = dspark_root / "snapshots" / dspark_ref.read_text().strip()
    else:
        snapshots = [p for p in (dspark_root / "snapshots").iterdir() if (p / "config.json").is_file()]
        if len(snapshots) != 1:
            raise RuntimeError("Cannot uniquely select cached DSpark checkpoint")
        draft = snapshots[0]
    if not (MODEL / "config.json").is_file() or not (draft / "config.json").is_file():
        raise RuntimeError("Cached model config missing")
    report["models"] = {str(p): hashlib.sha256((p / "config.json").read_bytes()).hexdigest()
                        for p in (MODEL, draft)}
    spec = {
        "model": str(draft), "num_speculative_tokens": 4, "method": "dspark",
        "attention_backend": "ROCM_AITER_MLA", "kv_cache_dtype": "fp8",
        "draft_sample_method": "probabilistic", "rejection_sample_method": "block",
    }
    report["speculative_config"] = spec
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("ATOM_", "INFERENCEX_NATIVE_")):
            env.pop(key)
    env.update({
        "PATH": str(ROOT / ".venv/bin") + ":" + env["PATH"],
        "PYTHONPATH": str(ROOT), "VLLM_PLUGINS": "pr64288_serving_observer",
        "VLLM_ROCM_USE_AITER": "1", "VLLM_ROCM_USE_AITER_LINEAR": "1",
        "VLLM_SERVER_DEV_MODE": "1",
        "VLLM_ROCM_USE_AITER_MOE_SITUV2": "1", "AITER_BF16_FP8_MOE_BOUND": "0",
        "AITER_FLYDSL_STAGE2_FP8": "1", "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION": "INT4",
        "VLLM_ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_MB": "16",
        "SAFETENSORS_FAST_GPU": "1", "HSA_NO_SCRATCH_RECLAIM": "1",
        "VLLM_USE_BREAKABLE_CUDAGRAPH": "0", "VLLM_USE_DIRECT_DCP_A2A": "0",
        "VLLM_USE_DIRECT_DCP_Q_GATHER": "0", "VLLM_USE_DIRECT_DCP_KV_GATHER": "0",
        "VLLM_ENGINE_READY_TIMEOUT_S": "2400", "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "1200",
        "PYTHONNOUSERSITE": "1", "PYTHONHASHSEED": "42", "PYTHONUNBUFFERED": "1",
        "TORCH_BLAS_PREFER_HIPBLASLT": "0", "TORCH_BLAS_PREFER_CUBLASLT": "0",
    })
    env.pop("VLLM_GPU_SYNC_CHECK", None)
    command = [
        ROOT / ".venv/bin/vllm", "serve", MODEL, "--served-model-name", "moonshotai/Kimi-K3",
        "--host", "127.0.0.1", "--port", str(PORT), "--trust-remote-code",
        "--moe-backend", "auto", "--tensor-parallel-size", "8",
        "--decode-context-parallel-size", "8", "--dcp-comm-backend", "a2a",
        "--gpu-memory-utilization", "0.90", "--language-model-only",
        "--quantization-config", json.dumps(quant),
        "--max-num-seqs", "96", "--max-num-batched-tokens", "8192",
        "--max-model-len", "65536", "--kv-cache-dtype", "fp8",
        "--attention-backend", "ROCM_AITER_MLA",
        "--attention-config", '{"mla_prefill_backend":"ROCM_AITER_FA"}',
        "--enable-prefix-caching", "--enable-auto-tool-choice", "--tool-call-parser", "kimi_k3",
        "--reasoning-parser", "kimi_k3", "--stream-interval", "10",
        "--speculative-config", json.dumps(spec),
        "--compilation-config", json.dumps({
            "mode": 3, "cudagraph_mode": "FULL_AND_PIECEWISE",
            "max_cudagraph_capture_size": 480, "custom_ops": ["+fused_rms_norm_gated"],
            "cudagraph_capture_sizes": list(range(2, 481, 2)),
        }),
    ]
    if command_transform is not None:
        command = command_transform(command)
    if env_transform is not None:
        env = env_transform(env)
    report["command"] = list(map(str, command))
    report["settings"] = {"tp": 8, "dcp": 8, "mtp": True, "synthetic_acceptance": False,
                          "ccd_pinning": True, "cpu_kv_offload": False, "normal_routing": True}
    if arms is None:
        arms = (("upstream_base", BASE, "0"), ("candidate_off", CANDIDATE, "0"),
                ("candidate_on", CANDIDATE, "1"))
    for arm, revision, flag in arms:
        directory = EVIDENCE / arm
        directory.mkdir()
        git("diff", "--exit-code")
        git("fetch", "--depth=1", "origin", revision)
        git("checkout", "--detach", revision)
        assert git("rev-parse", "HEAD", capture=True).stdout.strip() == revision
        manifest = {
            "native_sha": revision, "source_root": str(ROOT),
            "source_hashes": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                              for p in SOURCE_PATHS if (ROOT / p).is_file()},
        }
        manifest_path = directory / "source-manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        env.update(REVIEW_MANIFEST=str(manifest_path),
                   REVIEW_PHASE_FILE=str(directory / "phase"),
                   REVIEW_SOURCE_SHA=revision,
                   VLLM_ROCM_USE_AITER_FLYDSL_FP8_PREFILL_ATTN=flag)
        (directory / "command.json").write_text(json.dumps(report["command"], indent=2))
        compiler_root = os.environ.get("REVIEW_COMPILER_CACHE_ROOT")
        if compiler_root:
            cache = Path(compiler_root) / arm
            cache.mkdir(parents=True)
            filesystem = run(["stat", "-f", "-c", "%T", cache], capture=True).stdout.strip()
            if "nfs" in filesystem.lower():
                raise RuntimeError("Compiler cache must use container-local storage")
            cache_env = {
                "TRITON_CACHE_DIR": str(cache / "triton"),
                "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"),
                "VLLM_CACHE_ROOT": str(cache / "vllm"),
            }
            env.update(cache_env)
            (directory / "compiler-cache.json").write_text(json.dumps(
                {"filesystem": filesystem, "environment": cache_env}, indent=2))
        log = directory / "server.log"
        proc = None
        try:
            with log.open("w") as out:
                proc = subprocess.Popen(list(map(str, command)), cwd=ROOT, env=env,
                    stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
                wait_ready(proc, log)
                pin_workers(recipe, directory, env)
                if exercise is None:
                    report["arms"][arm] = []
                    for case in CASES:
                        report["arms"][arm].append(bench(arm, case, env, log, proc))
                        (EVIDENCE / "serving-validation.json").write_text(json.dumps(report, indent=2))
                else:
                    report["arms"][arm] = exercise(arm, env, log, proc)
        finally:
            stop(proc)
            git("diff", "--exit-code")
    report["status"] = "passed"


if __name__ == "__main__":
    report = {"status": "running", "arms": {}, "candidate": CANDIDATE, "base": BASE,
              "scope": "Short synthetic bench serve with live MTP; no AgentX or model accuracy claim"}
    try:
        main(report)
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        (EVIDENCE / "serving-validation.json").write_text(json.dumps(report, indent=2))
        print("SERVING_VALIDATION", json.dumps(report), flush=True)
