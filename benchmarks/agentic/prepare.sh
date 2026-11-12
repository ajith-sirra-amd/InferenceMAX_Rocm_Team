#!/usr/bin/env bash
# Prepare the measured SA client on CPU before either paired server starts.
set -euo pipefail
tools_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
pair_dir=${1:?Supply a fresh pair client directory under shared benchmarks}
sa_checkout=${2:?Supply the measured checkout inferencex-e2e directory}
case "$pair_dir" in /share_nfs/ajith/benchmarks/*) ;; *) echo 'Use shared benchmark storage' >&2; exit 2 ;; esac
pair_dir=$(realpath -m -- "$pair_dir")
case "$pair_dir" in /share_nfs/ajith/benchmarks/*) ;; *) echo 'Client directory escapes shared benchmark storage' >&2; exit 2 ;; esac
sa_checkout=$(realpath -- "$sa_checkout")
test -d "$sa_checkout/infx/bench/agentic"
expected=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["sa_source_sha"])' "$tools_dir/pins.json")
test "$(git -C "$sa_checkout" rev-parse HEAD)" = "$expected"
test -z "$(git -C "$sa_checkout" status --porcelain --untracked-files=all -- infx benchmarks)"
mkdir -p "$pair_dir"
if [[ -n "$(find "$pair_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo 'Refusing to overwrite client preparation; use a fresh pair directory' >&2; exit 2
fi
image=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["image"])' "$tools_dir/../fixed_length/baseline.json")
container="sa-agentic-prepare-$(basename "$(dirname "$pair_dir")")"
run_uid=$(id -u)
run_gid=$(id -g)
cleanup() {
    rc=$?
    trap - EXIT INT TERM
    timeout 15s docker stop --time 10 "$container" >/dev/null 2>&1 || true
    timeout 10s docker rm -f "$container" >/dev/null 2>&1 || true
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
timeout --signal=TERM --kill-after=10s 5m docker image inspect "$image" >/dev/null 2>&1 || \
    timeout --signal=TERM --kill-after=10s 10m docker pull "$image"
docker image inspect --format '{{json .RepoDigests}}' "$image" > "$pair_dir/image-digests.json"
printf '%s\n' "$expected" > "$pair_dir/source-sha.txt"
# No GPU devices or host networking. Client files remain owned by the runner.
timeout --signal=TERM --kill-after=30s 30m docker run --rm --init --name "$container" \
    --user "$run_uid:$run_gid" --cap-drop ALL --security-opt no-new-privileges \
    --mount "type=bind,src=$pair_dir,dst=/agentic-work" \
    --mount "type=bind,src=$sa_checkout,dst=/sa-client,readonly" \
    --mount "type=bind,src=$tools_dir,dst=/agentic-tools,readonly" \
    --mount "type=bind,src=$tools_dir/../fixed_length/baseline.json,dst=/baseline.json,readonly" \
    --env HF_TOKEN --env PYTHONDONTWRITEBYTECODE=1 \
    --entrypoint /opt/runtime-venv/bin/python "$image" /agentic-tools/prepare.py
