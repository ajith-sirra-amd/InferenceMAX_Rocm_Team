#!/usr/bin/bash
HF_HUB_CACHE_MOUNT="/mnt/nvme7n1"

MODEL_CODE="${EXP_NAME%%_*}"
if [[ $FRAMEWORK == "vllm" ]]; then
    FRAMEWORK_SUFFIX="_vllm"
elif [[ $FRAMEWORK == "sglang" ]]; then
    FRAMEWORK_SUFFIX="_sglang"
elif [[ $FRAMEWORK == "trt" ]]; then
    FRAMEWORK_SUFFIX="_trt"
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
    BENCHMARK_PATH=upstream/InferenceX/benchmarks/${BENCHMARK_SUBDIR}/agentic/${MODEL_CODE}_${PRECISION}_b200${SPEC_SUFFIX}.sh
else
    BENCHMARK_PATH=upstream/InferenceX/benchmarks/${BENCHMARK_SUBDIR}/${MODEL_CODE}_${PRECISION}_b200${SPEC_SUFFIX}.sh
fi

export PYTHONDONTWRITEBYTECODE=1


docker run --rm --network=host --name=$server_name \
--runtime=nvidia --gpus=all --ipc=host --privileged --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
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
-e PYTHONPYCACHEPREFIX=/tmp/pycache/ -e TORCH_CUDA_ARCH_LIST="9.0" -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
-e OFFLOADING \
-e TOTAL_CPU_DRAM_GB \
-e RESULT_DIR \
-e PYTHONDONTWRITEBYTECODE \
--entrypoint=/bin/bash \
$IMAGE \
$BENCHMARK_PATH

# Cleanup: stop server container 
docker stop $server_name 2>/dev/null || true
docker rm $server_name 2>/dev/null || true