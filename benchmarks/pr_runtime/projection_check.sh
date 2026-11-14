#!/usr/bin/env bash
set -eo pipefail
tools=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
out=${IPAT_PR_OUTPUT:?}
case "$out" in /share_nfs/gguasti/benchmarks/pr-*) ;; *) exit 2 ;; esac
mkdir -p "$out"
for arm in baseline candidate; do
    python3 "$tools/runtime.py" "$IPAT_PR_MANIFEST" --pr 67372 --arm "$arm" --output "$out/$arm" --verify
done
python3 "$tools/wait_idle.py" "$out/gpu.availability.jsonl"
image=$(cat "$out/candidate/runtime-image.txt")
sha=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["candidate"]["source_sha"])' "$IPAT_PR_MANIFEST")
module_sha=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["source_sha256"]["vllm/models/kimi_k3/nvidia/dspark_mla.py"])' "$out/candidate/runtime-verification.json")
name="kimi-pr-projection-$(basename "$out")"
uid=$(id -u); gid=$(id -g)
cleanup() {
    rc=$?; trap - EXIT INT TERM
    timeout 50s docker stop --time 45 "$name" >/dev/null 2>&1 || true
    timeout 10s docker rm -f "$name" >/dev/null 2>&1 || true
    timeout 30s docker run --rm --name "$name-ownership" --network none \
        --mount "type=bind,src=$out,dst=/logs" --entrypoint chown "$image" -R "$uid:$gid" /logs || rc=1
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
git -C "$tools" rev-parse HEAD > "$out/team-sha.txt"
timeout --signal=TERM --kill-after=60s 1200s docker run --rm --init --name "$name" \
    --label "ipat.pr.output=$out" --network host --ipc host --device /dev/kfd --device /dev/dri \
    --security-opt seccomp=unconfined --ulimit memlock=-1 --ulimit stack=67108864 \
    --mount "type=bind,src=$out,dst=/logs" --mount "type=bind,src=$tools,dst=/pr-tools,readonly" \
    --tmpfs /tmp:rw,exec,size=64g --env OMP_NUM_THREADS=4 --env PYTHONDONTWRITEBYTECODE=1 \
    --env VLLM_TARGET_DEVICE=rocm --env VLLM_GPU_SYNC_CHECK=error --env "EXPECTED_SOURCE_SHA=$sha" --env "EXPECTED_MODULE_SHA=$module_sha" \
    --entrypoint /opt/runtime-venv/bin/python "$image" \
    -m torch.distributed.run --standalone --nproc-per-node=8 /pr-tools/projection_check.py \
    2>&1 | tee "$out/projection.log"
python3 - "$out" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); records=[json.loads((root/'projection'/f'rank-{rank}.json').read_text()) for rank in range(8)]
assert all(r['status']=='passed' and len(r['cases'])==10 for r in records)
assert len({r['module_sha256'] for r in records})==1
(root/'projection-validation.json').write_text(json.dumps(dict(status='passed',ranks=8,cases_per_rank=10,
 limits='Native TP8 BF16 layers and graphs; not full model activation or accuracy'),indent=2)+'\n')
PY
