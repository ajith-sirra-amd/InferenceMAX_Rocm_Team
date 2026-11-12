#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
upstream_dir=$(cd "$script_dir/../../upstream/InferenceX" && pwd)
run_dir=${1:?Supply a fresh shared benchmark output directory}
optimization=${2:?Supply baseline, attnres, dspark, moe_front or moe_front_dspark}
workload=${3:-fixed}
case "$workload" in fixed|natural) ;; *) echo 'Unknown workload' >&2; exit 2 ;; esac
if [[ "$workload" == natural && "$optimization" != baseline && "$optimization" != moe_front_dspark ]]; then
    echo 'Natural validation supports baseline and combined only' >&2; exit 2
fi
case "$optimization" in baseline|attnres|dspark|moe_front|moe_front_dspark) ;; *) echo 'Unknown optimization' >&2; exit 2 ;; esac
case "$run_dir" in /share_nfs/ajith/benchmarks/*) ;; *) echo 'Use the shared benchmark output directory' >&2; exit 2 ;; esac
mkdir -p "$run_dir"
if [[ -e "$run_dir/recipe.json" ]]; then echo 'Refusing to overwrite an earlier run' >&2; exit 2; fi
if [[ "$workload" == natural ]]; then
    if [[ -e "$run_dir/base-recipe.json" ]]; then echo 'Refusing to overwrite an earlier natural run' >&2; exit 2; fi
    cp "$script_dir/baseline.json" "$run_dir/base-recipe.json"
else
    cp "$script_dir/baseline.json" "$run_dir/recipe.json"
fi
mkdir -p "$run_dir/work/home" "$run_dir/work/cache" "$run_dir/work/tmp"
image=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["image"])' "$script_dir/baseline.json")
container="kimi-fixed-$(basename "$run_dir")"
if [[ "$workload" == natural ]]; then container="kimi-natural-$(basename "$run_dir")"; fi
printf '%s\n' "$container" > "$run_dir/container-name.txt"
port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
run_uid=$(id -u)
run_gid=$(id -g)
cleanup() {
    rc=$?
    trap - EXIT INT TERM
    if [[ "$workload" == natural ]]; then
        timeout 50s docker stop --time 45 "$container" >/dev/null 2>&1 || true
        timeout 10s docker rm -f "$container" "$container-install" >/dev/null 2>&1 || true
        timeout 30s docker run --rm --name "$container-ownership" --network none \
            --mount "type=bind,src=$run_dir,dst=/logs" \
            --entrypoint chown "$image" -R "$run_uid:$run_gid" /logs || rc=1
        timeout 5s docker rm -f "$container-ownership" >/dev/null 2>&1 || true
        remaining=$(timeout 10s docker ps --all --filter "name=^/${container}(-ownership|-install)?$" --format '{{.ID}}') || rc=1
        if [[ -n "$remaining" ]]; then echo 'Owned natural-validation container cleanup incomplete' >&2; rc=1; fi
        exit "$rc"
    fi
    docker stop --time 120 "$container" >/dev/null 2>&1 || true
    docker rm -f "$container" >/dev/null 2>&1 || true
    docker run --rm --network none --mount "type=bind,src=$run_dir,dst=/logs" \
        --entrypoint chown "$image" -R "$run_uid:$run_gid" /logs || true
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
docker image inspect "$image" >/dev/null 2>&1 || docker pull "$image"
docker image inspect --format '{{json .RepoDigests}}' "$image" > "$run_dir/image-digests.json"
git -C "$script_dir" rev-parse HEAD > "$run_dir/code-sha.txt"
for directory in /share_nfs/models/moonshotai/Kimi-K3.1-Pro /share_nfs/models/Inferact/Kimi-K3.1-Pro-DSpark-preview; do
    test -d "$directory" || { echo "Missing model directory: $directory" >&2; exit 1; }
done
overlays=()
entry_command='set -e; export PATH=/opt/runtime-venv/bin:$PATH; export PYTHONPATH=/inferencex; cd /inferencex; python3 /fixed-tools/check_client.py; exec bash benchmarks/single_node/agentic/kimik3_fp4_mi355x_mtp.sh'
if [[ "$workload" == natural ]]; then
    quality_dir=$(cd "$script_dir/../quality" && pwd)
    overlays+=(--mount "type=bind,src=$quality_dir,dst=/quality,readonly")
    entry_command='set -e; export PATH=/opt/runtime-venv/bin:$PATH; export PYTHONPATH=/inferencex; exec /opt/runtime-venv/bin/python /quality/launch.py'
fi
if [[ "$optimization" != baseline ]]; then
    installer=(docker run --rm --network none)
    if [[ "$workload" == natural ]]; then
        installer=(timeout --signal=TERM --kill-after=10s 80s docker run --rm --name "$container-install" --network none)
    fi
    "${installer[@]}" \
        --mount "type=bind,src=$script_dir,dst=/fixed-tools,readonly" \
        --mount "type=bind,src=$run_dir,dst=/logs" \
        --env PYTHONDONTWRITEBYTECODE=1 \
        --entrypoint /opt/runtime-venv/bin/python "$image" \
        "/fixed-tools/optimizations/$optimization/install.py" \
        --source /opt/runtime-venv/lib/python3.12/site-packages --output /logs/work/overlay
    cp "$run_dir/work/overlay/manifest.json" "$run_dir/optimization.json"
    if [[ "$optimization" == attnres ]]; then
        sources=(vllm/models/kimi_k3/amd/linear.py vllm/models/kimi_k3/amd/ops/attn_res.py)
        overlays+=(--env KIMI_ATTNRES_PRECOMPUTE_WEIGHTS=1)
    elif [[ "$optimization" == dspark ]]; then
        sources=(vllm/models/kimi_k3/nvidia/dspark_mla.py)
        overlays+=(--env KIMI_DSPARK_SHARD_CONTEXT_PROJ=1)
    else
        sources=(vllm/models/kimi_k3/amd/linear.py vllm/models/kimi_k3/amd/latent_moe_runner.py)
        overlays+=(--env KIMI_MOE_MERGED_FRONT=1)
        if [[ "$optimization" == moe_front_dspark ]]; then
            sources+=(vllm/models/kimi_k3/nvidia/dspark_mla.py)
            overlays+=(--env KIMI_DSPARK_SHARD_CONTEXT_PROJ=1)
        fi
    fi
    for source in "${sources[@]}"; do
        overlays+=(--mount "type=bind,src=$run_dir/work/overlay/candidate/$source,dst=/opt/runtime-venv/lib/python3.12/site-packages/$source,readonly")
    done
fi
if [[ "$workload" == natural ]]; then
    echo "Starting server; workload=natural; optimization=$optimization; output=$run_dir; port=$port"
else
    echo "Starting SA-configured server; optimization=$optimization; output=$run_dir; port=$port"
fi
docker run --rm --init --name "$container" --label "kimi.fixed.run=$(basename "$run_dir")" \
    "${overlays[@]}" --env "KIMI_FIXED_OPTIMIZATION=$optimization" \
    --network host --ipc host --device /dev/kfd --device /dev/dri \
    --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --mount "type=bind,src=$script_dir,dst=/fixed-tools,readonly" \
    --mount "type=bind,src=$upstream_dir,dst=/inferencex,readonly" \
    --mount "type=bind,src=$run_dir,dst=/logs" \
    --mount "type=bind,src=$run_dir/work,dst=/work" \
    --mount "type=bind,src=$run_dir/work/home,dst=/root" \
    --mount type=bind,src=/share_nfs/models,dst=/models,readonly \
    --mount type=bind,src=/share_nfs/models/moonshotai/Kimi-K3.1-Pro,dst=/model,readonly \
    --tmpfs /tmp:rw,exec,size=64g \
    --env "PORT=$port" --env TMPDIR=/tmp --env PYTHONDONTWRITEBYTECODE=1 \
    --env PYTHONPYCACHEPREFIX=/work/cache/python \
    --env XDG_CACHE_HOME=/work/cache --env FLYDSL_RUNTIME_CACHE_DIR=/work/cache/flydsl \
    --env CCACHE_DIR=/work/cache/ccache --env SCCACHE_DIR=/work/cache/sccache \
    --env MODEL=moonshotai/Kimi-K3.1-Pro --env MODEL_PATH=/model \
    --env DRAFT_MODEL_PATH=/models/Inferact/Kimi-K3.1-Pro-DSpark-preview \
    --env TP=8 --env DCP_SIZE=8 --env CONC=48 --env EP_SIZE=1 \
    --env KV_OFFLOADING=dram --env KV_OFFLOAD_BACKEND=vllm-native --env TOTAL_CPU_DRAM_GB=1799 \
    --env RESULT_DIR=/logs --env RESULT_FILENAME=baseline \
    --env EVAL_ONLY=false --env DURATION=600 --env PROFILE=0 \
    --env TEST=1 --env TEST_ISL=8192 --env TEST_OSL=1024 \
    --env TEST_RANGE_RATIO=1 --env TEST_NUM_PROMPTS=192 \
    --env SA_REFERENCE_CONFIG=/fixed-tools/baseline.json --env SA_CONFIG_AUDIT=/fixed-tools/audit_server.py \
    --entrypoint /bin/bash "$image" -c \
    "$entry_command"
if [[ "$workload" == fixed ]]; then
    python3 "$script_dir/validate.py" "$run_dir/baseline.json" --output "$run_dir/validation.json"
fi
python3 "$script_dir/validate_optimization.py" "$run_dir"
