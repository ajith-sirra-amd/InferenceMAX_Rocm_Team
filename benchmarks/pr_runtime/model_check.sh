#!/usr/bin/env bash
set -eo pipefail
tools=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(realpath "$tools/../..")
out=${IPAT_PR_OUTPUT:?}
case "$out" in /share_nfs/gguasti/benchmarks/pr-model-*) ;; *) exit 2 ;; esac
mkdir -p "$out"
for arm in baseline candidate; do
    python3 "$tools/runtime.py" "$IPAT_PR_MANIFEST" --pr 67372 --arm "$arm" --output "$out/$arm" --verify
done
test -d /share_nfs/models/moonshotai/Kimi-K3.1-Pro
test -d /share_nfs/models/Inferact/Kimi-K3.1-Pro-DSpark-preview
git -C "$repo" rev-parse HEAD > "$out/team-sha.txt"
uid=$(id -u); gid=$(id -g)
name="kimi-pr-model-$(basename "$out")"
cleanup() {
    rc=$?; trap - EXIT INT TERM
    timeout 50s docker stop --time 45 "$name" >/dev/null 2>&1 || true
    timeout 10s docker rm -f "$name" >/dev/null 2>&1 || true
    timeout 30s docker run --rm --network none --mount "type=bind,src=$out,dst=/logs" --entrypoint chown "$image" -R "$uid:$gid" /logs || rc=1
    exit "$rc"
}
image=$(cat "$out/baseline/runtime-image.txt")
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for arm in baseline candidate; do
    python3 "$tools/wait_idle.py" "$out/gpu.availability.jsonl"
    image=$(cat "$out/$arm/runtime-image.txt")
    sha=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))[sys.argv[2]]["source_sha"])' "$IPAT_PR_MANIFEST" "$arm")
    module_sha=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["source_sha256"]["vllm/models/kimi_k3/nvidia/dspark_mla.py"])' "$out/$arm/runtime-verification.json")
    mkdir -p "$out/$arm/cache"
    port=$(python3 -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()')
    timeout --signal=TERM --kill-after=120s 2400s docker run --rm --init --name "$name" \
        --label "ipat.pr.output=$out" --network host --ipc host --device /dev/kfd --device /dev/dri \
        --security-opt seccomp=unconfined --ulimit memlock=-1 --ulimit stack=67108864 \
        --mount "type=bind,src=$out/$arm,dst=/logs" --mount "type=bind,src=$tools,dst=/pr-tools,readonly" \
        --mount "type=bind,src=$repo/benchmarks/quality,dst=/quality,readonly" \
        --mount "type=bind,src=$repo/benchmarks/fixed_length,dst=/fixed-tools,readonly" \
        --mount type=bind,src=/share_nfs/models,dst=/models,readonly \
        --mount type=bind,src=/share_nfs/models/moonshotai/Kimi-K3.1-Pro,dst=/model,readonly \
        --tmpfs /tmp:rw,exec,size=64g --env OMP_NUM_THREADS=4 --env PYTHONDONTWRITEBYTECODE=1 \
        --env PYTHONPATH=/pr-tools --env XDG_CACHE_HOME=/logs/cache --env TMPDIR=/tmp \
        --env "PORT=$port" --env "IPAT_ARM=$arm" --env "EXPECTED_SOURCE_SHA=$sha" --env "EXPECTED_MODULE_SHA=$module_sha" \
        --env VLLM_TARGET_DEVICE=rocm --env VLLM_GPU_SYNC_CHECK=error \
        --entrypoint /opt/runtime-venv/bin/python "$image" /pr-tools/model_check.py \
        2>&1 | tee "$out/$arm/coordinator.log"
done
