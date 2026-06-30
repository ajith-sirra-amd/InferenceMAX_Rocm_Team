#!/usr/bin/env bash
if [[ $RUNNER_NAME == *gbt350* ]]; then
    HF_HUB_CACHE_MOUNT="/data/hf_hub_cache/actions-runner"
elif [[ $RUNNER_TYPE == "mi355x" ]]; then
    HF_HUB_CACHE_MOUNT="/it-share/hf_cache/"
elif [[ $RUNNER_TYPE == "mi355x-p02-g57" ]]; then
    HF_HUB_CACHE_MOUNT="/mnt/hf_hub_cache/"
elif [[ $RUNNER_TYPE == "mi355x-do" ]]; then
    HF_HUB_CACHE_MOUNT="/data/hf_hub_cache/"
fi

# smci355-ccs-aus-m15-17 has a persistent process bound to 127.0.0.1:8888,
# preventing vllm/sglang from binding the default port. Pin a free port; the
# runner already forwards -e PORT into the container and benchmark_lib.sh reads
# $PORT. Gate on the runner name so other mi355x hosts are unaffected.
if [[ $RUNNER_NAME == *m15-17* ]]; then
    export PORT=8911
fi

MODEL_CODE="${EXP_NAME%%_*}"
if [[ $FRAMEWORK == "vllm" ]]; then
    FRAMEWORK_SUFFIX="_vllm"
elif [[ $FRAMEWORK == "sglang" ]]; then
    FRAMEWORK_SUFFIX="_sglang"
elif [[ $FRAMEWORK == "atom" ]]; then
    FRAMEWORK_SUFFIX="_atom"
fi
SPEC_SUFFIX=$([[ "$SPEC_DECODING" == "mtp" ]] && printf '_mtp' || printf '')

server_name="bmk-server"

# ---------------------------------------------------------------------------
# REUSE_SERVER=1 — skip container restart when server is already healthy.
#
# Use this for concurrency sweeps where the same model/offloading config is
# tested at multiple CONC values sequentially on the same runner:
#
#   First job (or default):  REUSE_SERVER=0 (default) — full start + cleanup
#   Subsequent jobs:         REUSE_SERVER=1 — benchmark only, server stays up
#   Last job:                REUSE_SERVER=0 — benchmark + teardown
#
# Condition for reuse: REUSE_SERVER=1 AND the vLLM/sglang health endpoint
# responds on http://localhost:$PORT/health within 5 s.
# If the health check fails the script falls back to a full restart.
# ---------------------------------------------------------------------------
_port="${PORT:-8888}"
_server_reused=0

if [[ "${REUSE_SERVER:-0}" == "1" ]]; then
    if curl --silent --fail --max-time 5 "http://localhost:${_port}/health" >/dev/null 2>&1; then
        echo "[REUSE_SERVER] Server on port ${_port} is healthy — skipping container restart."
        _server_reused=1
    else
        echo "[REUSE_SERVER] No healthy server on port ${_port} — falling back to full start."
    fi
fi

if [[ "$_server_reused" == "0" ]]; then
    # Cleanup: force-remove any stale server container.
    # Kill first (in case the container is still running and rm -f is slow to stop it),
    # then remove. Wait up to 60 s for Docker to finish removing it.
    docker kill $server_name 2>/dev/null || true
    docker rm -f $server_name 2>/dev/null || true
    for _ in $(seq 1 60); do
        docker ps -aq -f "name=^${server_name}$" | grep -q . || break
        sleep 1
    done
    # Abort if the container is still listed after the wait; don't proceed to docker run.
    if docker ps -aq -f "name=^${server_name}$" | grep -q .; then
        echo "ERROR: container ${server_name} still exists after 60 s cleanup — aborting." >&2
        exit 1
    fi
fi

set -x
docker pull $IMAGE
DIGEST=$(docker inspect --format='{{index .RepoDigests 0}}' "$IMAGE" | cut -d'@' -f2)
echo "The image digest is: $DIGEST"

if [[ "$FRAMEWORK" == "sglang-disagg" ]]; then
    BENCHMARK_SUBDIR="multi_node"
else
    BENCHMARK_SUBDIR="single_node"
fi

if [[ "$OFFLOADING" == "cpu" ]] || [[ "$OFFLOADING" == "none" ]] || [[ "$OFFLOADING" == "lmcache" ]] || [[ "$OFFLOADING" == "hicache" ]]; then
    if [[ $FRAMEWORK == "atom" ]]; then
        BENCHMARK_PATH=upstream/InferenceX/benchmarks/${BENCHMARK_SUBDIR}/agentic/${MODEL_CODE}_${PRECISION}_mi355x_atom${SPEC_SUFFIX}.sh
    else
        BENCHMARK_PATH=upstream/InferenceX/benchmarks/${BENCHMARK_SUBDIR}/agentic/${MODEL_CODE}_${PRECISION}_mi355x${SPEC_SUFFIX}.sh
    fi
else
    BENCHMARK_PATH=upstream/InferenceX/benchmarks/${BENCHMARK_SUBDIR}/${MODEL_CODE}_${PRECISION}_mi355x${SPEC_SUFFIX}.sh
fi

export PYTHONDONTWRITEBYTECODE=1

# Common docker flags used in both full-start and client-only modes.
_DOCKER_COMMON=(
    --rm --init --network host
    --shm-size=128g --ipc=host
    --ulimit memlock=-1 --ulimit stack=67108864
    --privileged --cap-add=CAP_SYS_ADMIN
    --device=/dev/kfd --device=/dev/dri --device=/dev/mem
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined
    -v "$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE"
    -v "$GITHUB_WORKSPACE:/workspace/" -w /workspace/
    -e IMAGE -e HF_TOKEN -e HF_HUB_CACHE
    -e MODEL -e TP -e CONC -e ISL -e OSL
    -e MAX_MODEL_LEN -e RANDOM_RANGE_RATIO -e RESULT_FILENAME
    -e EP_SIZE -e DP_ATTENTION -e RUN_EVAL -e OFFLOADING
    -e TOTAL_CPU_DRAM_GB -e DURATION -e PORT -e RESULT_DIR
    -e PYTHONDONTWRITEBYTECODE
    --entrypoint=/bin/bash
)

if [[ "$_server_reused" == "0" ]]; then
    # Full start: server + benchmark inside a single named container.
    docker rm -f $server_name 2>/dev/null || true
    docker run "${_DOCKER_COMMON[@]}" \
        --pull always \
        --name="$server_name" \
        "$IMAGE" \
        "$BENCHMARK_PATH"
else
    # Reuse mode: server stays up; run only the benchmark client in an
    # ephemeral container (no --name, no --pull always to save time).
    # Sources benchmark_lib.sh so resolve_trace_source / install_agentic_deps
    # / build_replay_cmd / run_agentic_replay_and_write_outputs are available.
    docker run "${_DOCKER_COMMON[@]}" \
        "$IMAGE" -c '
            set -euo pipefail
            cd /workspace
            source upstream/InferenceX/benchmarks/benchmark_lib.sh
            resolve_trace_source
            install_agentic_deps
            build_replay_cmd "$RESULT_DIR"
            run_agentic_replay_and_write_outputs "$RESULT_DIR"
        '
fi

if ls gpucore.* 1> /dev/null 2>&1; then
  echo "gpucore files exist. not good"
  rm -f gpucore.*
fi

if [[ "$_server_reused" == "0" ]]; then
    # Cleanup: stop server container (only when we own it).
    docker stop $server_name 2>/dev/null || true
    docker rm $server_name 2>/dev/null || true
else
    echo "[REUSE_SERVER] Server container kept alive for next sweep run."
fi
