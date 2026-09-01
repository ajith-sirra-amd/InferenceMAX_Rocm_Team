#!/usr/bin/env bash
if [[ $RUNNER_NAME == *mi355x* ]]; then
    HF_HUB_CACHE_MOUNT="/it-share/models"
elif [[ $RUNNER_NAME == *gbt* ]]; then
    HF_HUB_CACHE_MOUNT="/data/hf_hub_cache"
elif [[ $RUNNER_NAME == *m15_g17* ]]; then
    HF_HUB_CACHE_MOUNT="/data/models"
elif [[ $RUNNER_NAME == *p02_g17* ]]; then
    HF_HUB_CACHE_MOUNT="/it-share/models"
fi

HF_HUB_CACHE_MOUNT="/data/hf_hub_cache"

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
# Local-image support. A locally built tag (kimi-k3-vllm:v4) is in no registry,
# so `docker pull` fails and `.RepoDigests` is empty -- the old
# `{{index .RepoDigests 0}}` aborted with "index out of range". Fall back to the
# image ID and stop `docker run` re-pulling. Registry images are unaffected.
PULL_POLICY=always
if ! docker pull "$IMAGE"; then
    docker image inspect "$IMAGE" >/dev/null 2>&1 || {
        echo "[image] $IMAGE is neither pullable nor present locally" >&2
        exit 1
    }
    echo "[image] pull failed -- using LOCAL image $IMAGE"
    PULL_POLICY=never
fi
DIGEST=$(docker inspect --format='{{if .RepoDigests}}{{index .RepoDigests 0}}{{end}}' "$IMAGE" | cut -d'@' -f2)
if [ -z "$DIGEST" ]; then
    DIGEST=$(docker inspect --format='{{.Id}}' "$IMAGE")
    PULL_POLICY=never
    echo "[image] no registry digest; using local image id"
fi
echo "The image digest is: $DIGEST"

if [[ "$FRAMEWORK" == "sglang-disagg" ]]; then
    BENCHMARK_SUBDIR="multi_node"
else
    BENCHMARK_SUBDIR="single_node"
fi

if [[ $FRAMEWORK == "atom" ]]; then
    BENCHMARK_PATH=upstream/InferenceX/benchmarks/${BENCHMARK_SUBDIR}/${SCENARIO_SUBDIR}${MODEL_CODE}_${PRECISION}_mi355x_atom${SPEC_SUFFIX}.sh
else
    BENCHMARK_PATH=upstream/InferenceX/benchmarks/${BENCHMARK_SUBDIR}/${SCENARIO_SUBDIR}${MODEL_CODE}_${PRECISION}_mi355x${SPEC_SUFFIX}.sh
fi

# MODEL_PATH: where the model weights live inside the container
export MODEL_NAME="${MODEL##*/}"
export MODEL_PATH="${HF_HUB_CACHE%/}/${MODEL_NAME}"

export PYTHONDONTWRITEBYTECODE=1

docker run --rm --init --network host --shm-size=512g --name=$server_name \
--ipc=host \
--ulimit memlock=-1 --ulimit stack=67108864 --pull ${PULL_POLICY:-always} \
--privileged --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
--cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
-v $HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
-v $GITHUB_WORKSPACE:/workspace/ -w /workspace/ \
-e HF_TOKEN \
-e HF_HUB_CACHE \
-e MODEL \
-e MODEL_PATH \
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
-e KV_OFFLOADING \
-e KV_OFFLOAD_BACKEND \
-e TOTAL_CPU_DRAM_GB \
-e DURATION \
-e PORT \
-e RESULT_DIR \
-e PYTHONDONTWRITEBYTECODE \
-e IMAGE \
-e MODEL_PREFIX \
-e "AIPERF_DIR=/workspace/upstream/InferenceX/utils/aiperf" \
-e "AGENTIC_DIR=/workspace/upstream/InferenceX/utils/agentic-benchmark" \
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