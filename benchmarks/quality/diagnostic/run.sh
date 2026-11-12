#!/usr/bin/env bash
# A separate baseline-only diagnostic. No installed runtime source is modified.
set -euo pipefail
repo=$(realpath -- "${1:?Supply the team checkout}")
out=$(realpath -m -- "${2:?Supply a fresh shared diagnostic output directory}")
if [[ "${3:-}" != --bounded ]]; then
    exec timeout --signal=TERM --kill-after=120s 2400s bash "$0" "$repo" "$out" --bounded
fi
case "$out" in /share_nfs/ajith/benchmarks/*) ;; *) echo 'Use shared benchmark storage' >&2; exit 2 ;; esac
tools=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
test -f "$repo/benchmarks/quality/launch.py"
test -f "$repo/benchmarks/fixed_length/baseline.json"
for model in /share_nfs/models/moonshotai/Kimi-K3.1-Pro /share_nfs/models/Inferact/Kimi-K3.1-Pro-DSpark-preview; do
    test -d "$model" || { echo "Missing model directory: $model" >&2; exit 2; }
done
mkdir -p "$out"
if [[ -n "$(find "$out" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo 'Refusing to overwrite earlier data' >&2; exit 2
fi
mkdir -p "$out/work/home" "$out/work/cache" "$out/work/tmp"
image=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["image"])' "$repo/benchmarks/fixed_length/baseline.json")
name="kimi-diagnostic-$(basename "$out")"
uid=$(id -u)
gid=$(id -g)
cleanup() {
    code=$?
    trap - EXIT INT TERM
    timeout 50s docker stop --time 45 "$name" >/dev/null 2>&1 || true
    timeout 10s docker rm -f "$name" >/dev/null 2>&1 || true
    timeout 30s docker run --rm --name "$name-ownership" --network none \
      --mount "type=bind,src=$out,dst=/logs" --entrypoint chown "$image" -R "$uid:$gid" /logs || code=1
    timeout 5s docker rm -f "$name-ownership" >/dev/null 2>&1 || true
    remaining=$(timeout 10s docker ps --all --filter "name=^/${name}(-ownership)?$" --format '{{.ID}}') || code=1
    if [[ -n "$remaining" ]]; then code=1; fi
    exit "$code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
timeout 30s docker image inspect "$image" >/dev/null 2>&1 || timeout 300s docker pull "$image"
docker image inspect --format '{{json .RepoDigests}}' "$image" > "$out/image-digests.json"
git -C "$repo" rev-parse HEAD > "$out/code-sha.txt"
port=$(python3 -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()')
docker run --rm --init --name "$name" \
  --network host --ipc host --device /dev/kfd --device /dev/dri \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --mount "type=bind,src=$repo/benchmarks/fixed_length,dst=/fixed-tools,readonly" \
  --mount "type=bind,src=$repo/benchmarks/quality,dst=/quality,readonly" \
  --mount "type=bind,src=$tools,dst=/quality-diagnostic,readonly" \
  --mount "type=bind,src=$repo/upstream/InferenceX,dst=/inferencex,readonly" \
  --mount "type=bind,src=$out,dst=/logs" \
  --mount "type=bind,src=$out/work,dst=/work" \
  --mount "type=bind,src=$out/work/home,dst=/root" \
  --mount type=bind,src=/share_nfs/models,dst=/models,readonly \
  --mount type=bind,src=/share_nfs/models/moonshotai/Kimi-K3.1-Pro,dst=/model,readonly \
  --tmpfs /tmp:rw,exec,size=64g \
  --env "PORT=$port" --env TMPDIR=/tmp --env PYTHONDONTWRITEBYTECODE=1 \
  --env PYTHONPYCACHEPREFIX=/work/cache/python --env PYTHONPATH=/inferencex \
  --env XDG_CACHE_HOME=/work/cache --env FLYDSL_RUNTIME_CACHE_DIR=/work/cache/flydsl \
  --env CCACHE_DIR=/work/cache/ccache --env SCCACHE_DIR=/work/cache/sccache \
  --env KIMI_FIXED_OPTIMIZATION=baseline --env RESULT_DIR=/logs \
  --env MODEL=moonshotai/Kimi-K3.1-Pro --env MODEL_PATH=/model \
  --entrypoint /opt/runtime-venv/bin/python "$image" /quality-diagnostic/diagnostic.py
