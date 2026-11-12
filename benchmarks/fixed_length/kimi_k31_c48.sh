#!/usr/bin/env bash
# Direct fixed-length baseline: pinned SA server, 8192 input / 1024 output, C48.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
upstream_dir=$(cd "$script_dir/../../upstream/InferenceX" && pwd)
run_dir=${1:?Usage: bash kimi_k31_c48.sh /share_nfs/ajith/benchmarks/<new-run>}
case "$run_dir" in /share_nfs/ajith/benchmarks/*) ;; *) echo 'Use the shared benchmark output directory' >&2; exit 2 ;; esac
mkdir -p "$run_dir"
if [[ -e "$run_dir/recipe.json" ]]; then echo 'Refusing to overwrite an earlier run' >&2; exit 2; fi
cp "$script_dir/baseline.json" "$run_dir/recipe.json"
mkdir -p "$run_dir/work/home" "$run_dir/work/cache" "$run_dir/work/tmp"
image=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["image"])' "$script_dir/baseline.json")
container="kimi-fixed-$(basename "$run_dir")"
printf '%s\n' "$container" > "$run_dir/container-name.txt"
port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
run_uid=$(id -u)
run_gid=$(id -g)
cleanup() {
    rc=$?
    trap - EXIT INT TERM
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
echo "Starting unchanged SA server; output=$run_dir; port=$port"
docker run --rm --init --name "$container" --label "kimi.fixed.run=$(basename "$run_dir")" \
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
    'set -e; export PATH=/opt/runtime-venv/bin:$PATH; export PYTHONPATH=/inferencex; cd /inferencex; python3 /fixed-tools/check_client.py; exec bash benchmarks/single_node/agentic/kimik3_fp4_mi355x_mtp.sh'
python3 "$script_dir/validate.py" "$run_dir/baseline.json" --output "$run_dir/validation.json"
