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

# Host path for the weight cache. Both nodes carry the label cluster:mi355x-amds,
# so the if-chain above (which keys off *mi355x*) cannot tell them apart -- the
# discriminator has to be the full RUNNER_NAME. Overridable by env for a new host.
case "$RUNNER_NAME" in
    mi355x-amd_guest*) HF_HUB_CACHE_MOUNT="${HF_HUB_CACHE_MOUNT_OVERRIDE:-/home/models}" ;;
    *)                 HF_HUB_CACHE_MOUNT="${HF_HUB_CACHE_MOUNT_OVERRIDE:-/data/hf_hub_cache}" ;;
esac
echo "[hf-cache] RUNNER_NAME=$RUNNER_NAME -> HF_HUB_CACHE_MOUNT=$HF_HUB_CACHE_MOUNT"

# benchmark_lib.sh derefs this unguarded since the InferenceX sync.
export INFMAX_CONTAINER_WORKSPACE="/workspace"

# benchmark_lib.sh used to default PORT; the synced copy does not, and upstream
# now supplies it as a workflow env (PORT: '8888'). Offsetting by the runner
# slot -- the scheme upstream's own mi355x launcher uses -- keeps concurrent
# jobs on one host off each other's port.
# Base 8890 so 8888 is never handed out. Runner names ending in a digit key off
# it; names that do not (mi355x-amd_guest) get a stable offset from a checksum of
# the name rather than all collapsing onto one port.
PORT_SUFFIX="${RUNNER_NAME: -1}"
if [[ "$PORT_SUFFIX" =~ ^[0-9]$ ]]; then
    PORT_OFFSET="$PORT_SUFFIX"
else
    PORT_OFFSET=$(( $(printf '%s' "$RUNNER_NAME" | cksum | cut -d' ' -f1) % 10 ))
fi
export PORT=$(( 8890 + PORT_OFFSET ))

# /workspace is the InferenceX root; the workflow looks for the result json at
# the repo root, so outputs go through a second mount.
export RESULT_DIR=/outputs/results

# The synced benchmark_lib.sh validates AIPERF_PYTHON_VERSION and ~30 siblings it
# no longer defaults (check_env_vars at ~3117). Upstream sources these from
# benchmarks/runtime_settings.sh in its own benchmark-tmpl.yml; this repo runs an
# older root workflow that does not, so source it here and forward the list the
# file itself declares via INFERENCEX_RUNTIME_ENV_VARS.
RUNTIME_SETTINGS="${INFMAX_HOST_REPO:-upstream/InferenceX}/benchmarks/runtime_settings.sh"
if [[ -f "$RUNTIME_SETTINGS" ]]; then
    source "$RUNTIME_SETTINGS"
else
    echo "WARNING: $RUNTIME_SETTINGS not found; aiperf env will be incomplete" >&2
fi
# runtime_settings.sh names these in INFERENCEX_RUNTIME_ENV_VARS but does not
# export them -- upstream supplies them from its own benchmark-tmpl.yml, which
# this repo's older root workflow predates. build_replay_cmd and the power check
# validate them, so an unset value fails the agentic path only (the fixed-length
# client never calls either). Defaults match upstream's.
export AIPERF_EXPERIMENTAL_FAST="${AIPERF_EXPERIMENTAL_FAST:-0}"   # 1 => 1200s profile, warmup 1/lane
export REQUIRE_POWER="${REQUIRE_POWER:-0}"
export IS_MULTINODE="${IS_MULTINODE:-false}"
# validate_required_agentic_server_metrics (reached from
# run_agentic_replay_and_write_outputs) checks these; single-node has no
# pipeline or prefill-context parallelism, so 1 is correct, not a placeholder.
export PP_SIZE="${PP_SIZE:-1}"
export PCP_SIZE="${PCP_SIZE:-1}"

# aiperf defaults to http://localhost:$PORT (benchmark_lib:3264) while every
# other client in that file uses an IPv4 literal. localhost resolves to ::1
# first here and vLLM binds --host 0.0.0.0, which is IPv4 only, so the agentic
# warmup gets ClientConnectorError against a healthy server. Pin the literal.
export AIPERF_SERVER_URL="${AIPERF_SERVER_URL:-http://127.0.0.1:${PORT}}"

# aiperf warmup replays every lane with zero idle delay, so C64 opens
# lanes x N connections at once (640 at the default 10). 63 requests were served
# and then 162 refused with ECONNREFUSED against a healthy server. Dropping to 2
# cuts the burst 5x; if the failure survives that, it is not connection volume.
export AIPERF_WARMUP_REQUESTS_PER_LANE="${AIPERF_WARMUP_REQUESTS_PER_LANE:-2}"

# infx/results/agentic validates KV_OFFLOAD_BACKEND_METADATA, not just
# KV_OFFLOAD_BACKEND -- the SystemExit text names the wrong variable. It fails on
# `backend_metadata is None`, and metadata["name"] must equal KV_OFFLOAD_BACKEND.
# Only the agentic result writer reads it, so fixed-length runs never noticed.
if [ -n "${KV_OFFLOAD_BACKEND:-}" ] && [ "${KV_OFFLOAD_BACKEND}" != "none" ]; then
    export KV_OFFLOAD_BACKEND_METADATA="${KV_OFFLOAD_BACKEND_METADATA:-{\"name\": \"${KV_OFFLOAD_BACKEND}\"}}"
fi

RUNTIME_ENV_ARGS=()
for _v in ${INFERENCEX_RUNTIME_ENV_VARS:-}; do
    [[ -n "${!_v+x}" ]] && RUNTIME_ENV_ARGS+=(-e "$_v")
done
unset _v

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
    BENCHMARK_PATH=benchmarks/${BENCHMARK_SUBDIR}/${SCENARIO_SUBDIR}${MODEL_CODE}_${PRECISION}_mi355x_atom${SPEC_SUFFIX}.sh
else
    BENCHMARK_PATH=benchmarks/${BENCHMARK_SUBDIR}/${SCENARIO_SUBDIR}${MODEL_CODE}_${PRECISION}_mi355x${SPEC_SUFFIX}.sh
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
-v $GITHUB_WORKSPACE/upstream/InferenceX:/workspace/ -w /workspace/ \
-v $GITHUB_WORKSPACE:/outputs \
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
-e INFMAX_CONTAINER_WORKSPACE \
-e IS_MULTINODE \
-e PP_SIZE \
-e PCP_SIZE \
-e AIPERF_SERVER_URL \
-e KV_OFFLOAD_BACKEND_METADATA \
-e IMAGE \
-e MODEL_PREFIX \
"${RUNTIME_ENV_ARGS[@]}" \
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