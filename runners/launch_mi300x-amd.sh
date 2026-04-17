#!/usr/bin/bash

sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'

#HF_HUB_CACHE_MOUNT="/shareddata/hf_hub_cache_$(hostname)/"
HF_HUB_CACHE_MOUNT="/home/amd/models"
PORT=8888

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

set -x
docker run --rm --init --network host --shm-size=16g --name=$server_name \
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
--entrypoint=/bin/bash \
$IMAGE \
benchmarks/${BENCHMARK_SUBDIR}/${MODEL_CODE}_${PRECISION}_mi300x${FRAMEWORK_SUFFIX}${SPEC_SUFFIX}.sh

if ls gpucore.* 1> /dev/null 2>&1; then
  echo "gpucore files exist. not good"
  rm -f gpucore.*
fi

# Cleanup: stop server container 
docker stop $server_name 2>/dev/null || true
docker rm $server_name 2>/dev/null || true
