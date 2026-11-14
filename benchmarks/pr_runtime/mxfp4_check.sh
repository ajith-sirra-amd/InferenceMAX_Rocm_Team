#!/usr/bin/env bash
set -eo pipefail
tools=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
out=${IPAT_PR_OUTPUT:?}
case "$out" in /share_nfs/gguasti/benchmarks/pr-mxfp4-*) ;; *) exit 2 ;; esac
mkdir -p "$out"
pr=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["pr"])' "$IPAT_PR_MANIFEST")
for arm in baseline candidate; do
    python3 "$tools/runtime.py" "$IPAT_PR_MANIFEST" --pr "$pr" --arm "$arm" --output "$out/$arm" --verify
done
python3 "$tools/wait_idle.py" "$out/gpu.availability.jsonl"
image=$(cat "$out/candidate/runtime-image.txt")
name="kimi-pr-mxfp4-$(basename "$out")"
uid=$(id -u); gid=$(id -g)
cleanup() {
    rc=$?; trap - EXIT INT TERM
    timeout 50s docker stop --time 45 "$name" >/dev/null 2>&1 || true
    timeout 10s docker rm -f "$name" >/dev/null 2>&1 || true
    timeout 30s docker run --rm --network none --mount "type=bind,src=$out,dst=/logs" --entrypoint chown "$image" -R "$uid:$gid" /logs || rc=1
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if [[ "${IPAT_ATTNRES_DIAG:-0}" == 1 ]]; then
    command=(/pr-tools/attnres_diagnosis.py)
elif [[ "${IPAT_METADATA:-0}" == 1 ]]; then
    command=(-m torch.distributed.run --standalone --nproc-per-node=8 /pr-tools/mxfp4_metadata.py)
elif [[ "${IPAT_NUMERICAL:-0}" == 1 ]]; then
    command=(/pr-tools/mxfp4_numerical.py)
else
    command=(-m pytest /vllm-workspace/tests/kernels/attention/test_kimi_mxfp4_indexer.py --runxfail -v -rA --junitxml=/logs/junit.xml)
fi
timeout --signal=TERM --kill-after=60s 1800s docker run --rm --init --name "$name" \
    --label "ipat.pr.output=$out" --network host --ipc host --device /dev/kfd --device /dev/dri \
    --security-opt seccomp=unconfined --ulimit memlock=-1 --ulimit stack=67108864 \
    --mount "type=bind,src=$out,dst=/logs" --mount "type=bind,src=$tools,dst=/pr-tools,readonly" --tmpfs /tmp:rw,exec,size=64g \
    --env OMP_NUM_THREADS=4 --env PYTHONDONTWRITEBYTECODE=1 --env VLLM_TARGET_DEVICE=rocm \
    --env XDG_CACHE_HOME=/logs/cache --env VLLM_GPU_SYNC_CHECK=error \
    --entrypoint /opt/runtime-venv/bin/python "$image" \
    "${command[@]}" 2>&1 | tee "$out/kernel.log"
if [[ "${IPAT_ATTNRES_DIAG:-0}" == 1 ]]; then
    test -f "$out/attnres-diagnosis.json"
    exit 0
elif [[ "${IPAT_METADATA:-0}" == 1 ]]; then
    python3 - "$out" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[1]);records=[json.loads((out/'metadata'/f'rank-{rank}.json').read_text()) for rank in range(8)]
assert all(r['status']=='passed' and len(r['cases'])==4 for r in records)
(out/'metadata-validation.json').write_text(json.dumps(dict(status='passed',ranks=8,cases_per_rank=4),indent=2)+'\n')
PY
    exit 0
elif [[ "${IPAT_NUMERICAL:-0}" == 1 ]]; then
    test -f "$out/numerical-validation.json"
    exit 0
fi
python3 - "$out" <<'PY'
import json,sys,xml.etree.ElementTree as ET
from pathlib import Path
out=Path(sys.argv[1]);cases=ET.parse(out/'junit.xml').findall('.//testcase')
assert len(cases)==10,len(cases)
assert all(not any(c.find(k) is not None for k in ['failure','error','skipped']) for c in cases)
(out/'kernel-validation.json').write_text(json.dumps(dict(status='passed',cases=len(cases),expected_failures_disabled=True,limits='Native cache isolation, FULL/PIECEWISE graph and causal-tail suite; full model quality remains pending'),indent=2)+'\n')
PY
