#!/usr/bin/env bash
set -eo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/../../upstream/InferenceX/benchmarks/benchmark_lib.sh" --validation-only
check_env_vars IPAT_PR_NUMBER IPAT_PR_MANIFEST IPAT_PR_OUTPUT IPAT_PR_DATA_ROOT
case "$IPAT_PR_NUMBER" in 67372|67373) ;; *) echo 'Unsupported PR' >&2; exit 2 ;; esac
case "$IPAT_PR_OUTPUT" in "$IPAT_PR_DATA_ROOT"/pr-*) ;; *) echo 'Use a fresh shared PR output directory' >&2; exit 2 ;; esac
case "$IPAT_PR_MANIFEST" in /share_nfs/*) ;; *) echo 'Use a prepared manifest in shared storage' >&2; exit 2 ;; esac
test -f "$IPAT_PR_MANIFEST"
mkdir -p "$IPAT_PR_OUTPUT"
for arm in baseline candidate; do
    python3 "$script_dir/runtime.py" "$IPAT_PR_MANIFEST" --pr "$IPAT_PR_NUMBER" \
        --arm "$arm" --output "$IPAT_PR_OUTPUT/$arm" --verify \
        --recipe "$script_dir/../fixed_length/baseline.json"
done
git -C "$script_dir" rev-parse HEAD > "$IPAT_PR_OUTPUT/team-sha.txt"
hostname > "$IPAT_PR_OUTPUT/host.txt"
container="kimi-pr-correctness-$(basename "$IPAT_PR_OUTPUT")"
printf '%s\n' "$container" > "$IPAT_PR_OUTPUT/container-name.txt"
image=$(cat "$IPAT_PR_OUTPUT/candidate/runtime-image.txt")
suite=$(cat "$IPAT_PR_OUTPUT/candidate/runtime-suite.txt")
run_uid=$(id -u)
run_gid=$(id -g)
cleanup() {
    rc=$?
    trap - EXIT INT TERM
    timeout 50s docker stop --time 45 "$container" >/dev/null 2>&1 || true
    timeout 10s docker rm -f "$container" >/dev/null 2>&1 || true
    timeout 30s docker run --rm --name "$container-ownership" --network none \
        --mount "type=bind,src=$IPAT_PR_OUTPUT,dst=/logs" \
        --entrypoint chown "$image" -R "$run_uid:$run_gid" /logs || rc=1
    timeout 10s docker rm -f "$container-ownership" >/dev/null 2>&1 || true
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
target=rocm
devices=(--device /dev/kfd --device /dev/dri)
if [[ "$IPAT_PR_NUMBER" == 67372 ]]; then
    # CPU loading/reconstruction; real BF16 serving is a separate GPU stage.
    target=cpu
    devices=()
    printf '%s\n' 'CPU component suite; no GPU devices exposed' > "$IPAT_PR_OUTPUT/execution-scope.txt"
else
    # An idle Actions runner is insufficient; never stop a foreign workload.
    python3 "$script_dir/wait_idle.py" "$IPAT_PR_OUTPUT/gpu.availability.jsonl"
    rocm-smi > "$IPAT_PR_OUTPUT/gpu-host.txt"
    printf '%s\n' 'ROCm suite after eight-GPU idle gate' > "$IPAT_PR_OUTPUT/execution-scope.txt"
fi

docker run --rm --init --name "$container" --network none --ipc host \
    "${devices[@]}" --security-opt seccomp=unconfined \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --mount "type=bind,src=$IPAT_PR_OUTPUT,dst=/logs" \
    --tmpfs /tmp:rw,exec,size=16g --workdir /vllm-workspace \
    --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONNOUSERSITE=1 \
    --env "VLLM_TARGET_DEVICE=$target" --env VLLM_GPU_SYNC_CHECK=error \
    --env PYTHONPATH=/vllm-workspace --env HF_HUB_OFFLINE=1 \
    --entrypoint /opt/runtime-venv/bin/python "$image" \
    -m pytest "$suite" -v --junitxml=/logs/correctness.xml \
    2>&1 | tee "$IPAT_PR_OUTPUT/correctness.log"
python3 "$script_dir/check_junit.py" "$IPAT_PR_OUTPUT/correctness.xml" \
    "$IPAT_PR_NUMBER" "$IPAT_PR_OUTPUT/correctness-validation.json"
