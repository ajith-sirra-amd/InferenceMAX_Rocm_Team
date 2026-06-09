#!/usr/bin/env bash
if [[ $RUNNER_NAME == *gbt350* ]]; then
    HF_HUB_CACHE_MOUNT="/data/hf_hub_cache/actions-runner"
elif [[ $RUNNER_TYPE == "mi355x" ]]; then
    HF_HUB_CACHE_MOUNT="/it-share/hf_cache/"
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

server_name="bmk-server"

# Cleanup: force-remove any stale server container.
docker rm -f $server_name 2>/dev/null || true
for _ in $(seq 1 30); do
    docker ps -aq -f "name=^${server_name}$" | grep -q . || break
    sleep 1
done

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
    BENCHMARK_PATH=upstream/InferenceX/benchmarks/${BENCHMARK_SUBDIR}/agentic/${MODEL_CODE}_${PRECISION}_mi355x${SPEC_SUFFIX}.sh
else
    BENCHMARK_PATH=upstream/InferenceX/benchmarks/${BENCHMARK_SUBDIR}/${MODEL_CODE}_${PRECISION}_mi355x${SPEC_SUFFIX}.sh
fi

export PYTHONDONTWRITEBYTECODE=1

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
