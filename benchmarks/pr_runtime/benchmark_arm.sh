#!/usr/bin/env bash
set -eo pipefail
tools=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(realpath "$tools/../..")
out=${1:?};optimization=${2:?};workload=${3:-fixed}
test "$workload" = fixed
case "$out" in /share_nfs/gguasti/benchmarks/pr-perf-*) ;; *) exit 2 ;; esac
case "$optimization" in baseline) arm=baseline ;; dspark|attnres) arm=candidate ;; *) exit 2 ;; esac
pr=${IPAT_PR_NUMBER:?}
python3 "$tools/runtime.py" "${IPAT_PR_MANIFEST:?}" --pr "$pr" --arm "$arm" --output "$out/runtime" --verify
python3 "$tools/wait_idle.py" "$out/gpu.availability.jsonl"
image=$(cat "$out/runtime/runtime-image.txt")
sha=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))[sys.argv[2]]["source_sha"])' "$IPAT_PR_MANIFEST" "$arm")
module_sha=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["source_sha256"]["vllm/models/kimi_k3/nvidia/dspark_mla.py"])' "$out/runtime/runtime-verification.json")
name="kimi-pr-perf-$(basename "$(dirname "$out")")-$(basename "$out")"
uid=$(id -u);gid=$(id -g)
cleanup() {
    rc=$?;trap - EXIT INT TERM
    timeout 50s docker stop --time 45 "$name" >/dev/null 2>&1 || true
    timeout 10s docker rm -f "$name" >/dev/null 2>&1 || true
    timeout 30s docker run --rm --network none --mount "type=bind,src=$out,dst=/logs" --entrypoint chown "$image" -R "$uid:$gid" /logs || rc=1
    exit "$rc"
}
trap cleanup EXIT;trap 'exit 130' INT;trap 'exit 143' TERM
port=$(python3 -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()')
mkdir -p "$out/cache"
timeout --signal=TERM --kill-after=120s 2400s docker run --rm --init --name "$name" \
    --label "ipat.pr.output=$out" --label "ipat.pr.pair=$(dirname "$out")" --network host --ipc host --device /dev/kfd --device /dev/dri \
    --security-opt seccomp=unconfined --ulimit memlock=-1 --ulimit stack=67108864 \
    --mount "type=bind,src=$out,dst=/logs" --mount "type=bind,src=$tools,dst=/pr-tools,readonly" \
    --mount "type=bind,src=$repo/benchmarks/quality,dst=/quality,readonly" \
    --mount "type=bind,src=$repo/benchmarks/fixed_length,dst=/fixed-tools,readonly" \
    --mount "type=bind,src=$repo/upstream/InferenceX,dst=/inferencex,readonly" \
    --mount type=bind,src=/share_nfs/models,dst=/models,readonly \
    --mount type=bind,src=/share_nfs/models/moonshotai/Kimi-K3.1-Pro,dst=/model,readonly \
    --tmpfs /tmp:rw,exec,size=64g --env OMP_NUM_THREADS=4 --env PYTHONDONTWRITEBYTECODE=1 \
    --env PYTHONPATH=/pr-tools:/inferencex --env XDG_CACHE_HOME=/logs/cache --env TMPDIR=/tmp \
    --env "PORT=$port" --env "IPAT_ARM=$arm" --env "IPAT_PR_NUMBER=$pr" \
    --env "EXPECTED_SOURCE_SHA=$sha" --env "EXPECTED_MODULE_SHA=$module_sha" --env "EXPECTED_IMAGE=$image" \
    --env PROFILE=0 --env VLLM_GPU_SYNC_CHECK= --env VLLM_TARGET_DEVICE=rocm \
    --entrypoint /opt/runtime-venv/bin/python "$image" /pr-tools/benchmark_check.py 2>&1 | tee "$out/coordinator.log"
