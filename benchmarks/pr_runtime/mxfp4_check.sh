#!/usr/bin/env bash
set -eo pipefail
tools=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
out=${IPAT_PR_OUTPUT:?}
case "$out" in /share_nfs/gguasti/benchmarks/pr-mxfp4-*) ;; *) exit 2 ;; esac
mkdir -p "$out"
for arm in baseline candidate; do
    python3 "$tools/runtime.py" "$IPAT_PR_MANIFEST" --pr 67372 --arm "$arm" --output "$out/$arm" --verify
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
timeout --signal=TERM --kill-after=60s 1800s docker run --rm --init --name "$name" \
    --label "ipat.pr.output=$out" --network host --ipc host --device /dev/kfd --device /dev/dri \
    --security-opt seccomp=unconfined --ulimit memlock=-1 --ulimit stack=67108864 \
    --mount "type=bind,src=$out,dst=/logs" --tmpfs /tmp:rw,exec,size=64g \
    --env OMP_NUM_THREADS=4 --env PYTHONDONTWRITEBYTECODE=1 --env VLLM_TARGET_DEVICE=rocm \
    --env XDG_CACHE_HOME=/logs/cache --env VLLM_GPU_SYNC_CHECK=error \
    --entrypoint /opt/runtime-venv/bin/python "$image" \
    -m pytest /vllm-workspace/tests/kernels/attention/test_kimi_mxfp4_indexer.py \
    --runxfail -v -rA --junitxml=/logs/junit.xml 2>&1 | tee "$out/kernel.log"
python3 - "$out" <<'PY'
import json,sys,xml.etree.ElementTree as ET
from pathlib import Path
out=Path(sys.argv[1]);cases=ET.parse(out/'junit.xml').findall('.//testcase')
assert len(cases)==10,len(cases)
assert all(not any(c.find(k) is not None for k in ['failure','error','skipped']) for c in cases)
(out/'kernel-validation.json').write_text(json.dumps(dict(status='passed',cases=len(cases),expected_failures_disabled=True,limits='Native cache isolation, FULL/PIECEWISE graph and causal-tail suite; full model quality remains pending'),indent=2)+'\n')
PY
