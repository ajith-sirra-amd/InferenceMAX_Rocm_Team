#!/usr/bin/env bash

set -x

cat /proc/meminfo | grep -E 'MemTotal|MemFree|MemAvailable|Cached|Buffers|SwapCached'
# sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
# sudo sh -c 'echo 0 > /proc/sys/vm/drop_caches'
# cat /proc/meminfo | grep -E 'MemTotal|MemFree|MemAvailable|Cached|Buffers|SwapCached'

if [[ $RUNNER_TYPE == "mi355x" ]]; then
    HF_HUB_CACHE_MOUNT="/it-share/hf_cache/"  # Temp solution
elif [[ $RUNNER_TYPE == "mi355x-p02-g57" ]]; then
    HF_HUB_CACHE_MOUNT="/mnt/hf_hub_cache/"
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

# Use PID to avoid container name conflicts when multiple jobs run concurrently
# on the same physical runner (e.g. original + _clone jobs on p01_g07).
server_name="bmk-server-$$"

# Derive a unique PORT from PID to avoid "[Errno 98] Address already in use"
# when two _clone jobs run concurrently on the same host.
# Base 8800 + (PID mod 100) gives a port in [8800, 8899], safely away from
# common defaults (8888) and SSH tunnels.
export PORT="${PORT:-$((8800 + ($$ % 100)))}"

# Acquire an exclusive host-level lock before running the benchmark.
# This serializes jobs across all runner processes on the same physical machine
# (e.g. "mi355x-amd_p02_g17" and "mi355x-amd_p02_g17_clone"), preventing
# GPU/NCCL conflicts and CPU RAM exhaustion from concurrent HiCache allocations.
# flock -x blocks until the lock is available; it is released automatically
# when this script exits (fd 9 is closed by the shell).
BMK_LOCK=/tmp/bmk-mi355x-amd.lock
echo "[lock] Waiting for host-level benchmark lock (${BMK_LOCK}) ..."
exec 9>"${BMK_LOCK}"
flock -x 9
echo "[lock] Lock acquired (PID=$$)."

# After acquiring the lock, reset any GPU compute processes that a previous
# crashed job (SIGKILL/OOM) may have left behind, preventing NCCL
# "unhandled cuda error" on ncclCommInitRank.
echo "[lock] Killing stale GPU compute processes ..."
for dev in /dev/dri/render*; do
    fuser -k "$dev" 2>/dev/null || true
done
sleep 3

# chown_workspace_back() {
#     docker run --rm -v "$GITHUB_WORKSPACE":/ws --entrypoint sh "$IMAGE" -c "rm -rf /ws/* /ws/.[!.]* /ws/..?*" 2>/dev/null || true
# }
# trap chown_workspace_back EXIT

# Cleanup: force remove any existing container with the same name
docker rm -f $server_name 2>/dev/null || true

docker pull $IMAGE
DIGEST=$(docker inspect --format='{{index .RepoDigests 0}}' "$IMAGE" | cut -d'@' -f2)
echo "The image digest is: $DIGEST"

if [[ "$FRAMEWORK" == "sglang-disagg" ]]; then
    BENCHMARK_SUBDIR="multi_node"
else
    BENCHMARK_SUBDIR="single_node"
fi

if [[ "$OFFLOADING" == "cpu" ]] || [[ "$OFFLOADING" == "none" ]] || [[ "$OFFLOADING" == "lmcache" ]] || [[ "$OFFLOADING" == "hicache" ]]; then
    BENCHMARK_PATH=upstream/InferenceX/benchmarks/${BENCHMARK_SUBDIR}/agentic/${MODEL_CODE}_${PRECISION}_mi355x${SPEC_SUFFIX}.sh
else
    BENCHMARK_PATH=upstream/InferenceX/benchmarks/${BENCHMARK_SUBDIR}/${MODEL_CODE}_${PRECISION}_mi355x${SPEC_SUFFIX}.sh
fi

export PYTHONDONTWRITEBYTECODE=1

set -x
docker run --rm --init --network host --shm-size=128g --name=$server_name \
--ipc=host \
--ulimit memlock=-1 --ulimit stack=67108864 --pull always \
--privileged --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
--cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
-v $HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
-v $GITHUB_WORKSPACE:/workspace/ -w /workspace/ \
-e HF_TOKEN \
-e HF_HUB_CACHE \
-e MODEL \
-e TP \
-e CONC \
-e ISL \
-e OSL \
-e MAX_MODEL_LEN \
-e RANDOM_RANGE_RATIO \
-e RESULT_FILENAME \
-e EP_SIZE \
-e DP_ATTENTION \
-e RUN_EVAL \
-e OFFLOADING \
-e TOTAL_CPU_DRAM_GB \
-e RESULT_DIR \
-e DURATION \
-e PORT \
-e PYTHONDONTWRITEBYTECODE \
--entrypoint=/bin/bash \
$IMAGE \
$BENCHMARK_PATH

if ls gpucore.* 1> /dev/null 2>&1; then
  echo "gpucore files exist. not good"
  rm -f gpucore.*
fi

# Cleanup: force remove any existing container with the same name
docker rm -f $server_name 2>/dev/null || true