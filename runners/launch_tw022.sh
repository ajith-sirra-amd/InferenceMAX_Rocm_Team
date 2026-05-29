#!/usr/bin/env bash
if [[ $RUNNER_TYPE == "mi300x" ]]; then
    HF_HUB_CACHE_MOUNT="/home/amd/models/"  # shared AMD model cache on the mi300x host
fi

# tw022 host has a persistent ssh tunnel bound to 127.0.0.1:8888 (sglang's
# default PORT), so the server can't bind there. Pin a free port; the runner
# forwards -e PORT and both the launcher and aiperf read $PORT.
export PORT=8911

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

# Cleanup: stop server container
docker stop $server_name 2>/dev/null || true
docker rm $server_name 2>/dev/null || true

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
    BENCHMARK_PATH=upstream/InferenceX/benchmarks/${BENCHMARK_SUBDIR}/agentic/${MODEL_CODE}_${PRECISION}_mi300x${SPEC_SUFFIX}.sh
else
    BENCHMARK_PATH=upstream/InferenceX/benchmarks/${BENCHMARK_SUBDIR}/${MODEL_CODE}_${PRECISION}_mi300x${SPEC_SUFFIX}.sh
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
-e DURATION \
-e PORT \
-e RESULT_DIR \
-e PYTHONDONTWRITEBYTECODE \
--entrypoint=/bin/bash \
$IMAGE \
$BENCHMARK_PATH

if ls gpucore.* 1> /dev/null 2>&1; then
  echo "gpucore files exist. not good"
  rm -f gpucore.*
fi

# Cleanup: stop server container
docker stop $server_name 2>/dev/null || true
docker rm $server_name 2>/dev/null || true
