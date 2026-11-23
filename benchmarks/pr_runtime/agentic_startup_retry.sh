#!/usr/bin/env bash
# Linked replacement of the failed second pair; preserve candidate/baseline order.
set -euo pipefail
tools=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
out=${IPAT_PR_OUTPUT:?};client=${IPAT_PR_AGENTIC_CLIENT:?};source=${IPAT_PR_AGENTIC_SOURCE:?}
parent=/share_nfs/gguasti/benchmarks/pr-perf-44244569921-1
allocation_gate() {
    python3 "$tools/allocation.py" --minimum-seconds "$1"
}
allocation_gate 21600
IPAT_VALIDATE_QUALITY_ONLY=1 bash "$tools/benchmark_pair.sh"
export IPAT_PR_NUMBER=67372
python3 - "$parent" "$out" "$tools" <<'PY'
import hashlib,json,sys
from pathlib import Path
parent,out,tools=map(Path,sys.argv[1:])
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old,new=read(parent/'pair-start.json'),read(out/'pair-start.json')
assert old==new and new['pr']==67372,'Frozen quality/runtime prerequisites differ'
failure=read(parent/'replica-2/baseline/performance-result.json')
assert failure['status']=='failed' and failure['error']=='TimeoutError: Readiness exceeded1800 seconds'
paths=[parent/'pair-start.json',parent/'replica-2/baseline/performance-result.json']
paths += [tools/name for name in ['benchmark_arm.sh','benchmark_check.py','runtime.py','performance_gate.py','startup_host_observer.py','agentic_startup_retry.sh']]
(out/'recovery-start.json').write_text(json.dumps(dict(
    parent=str(parent),parent_failed_arm='replica-2/baseline',parent_failure=failure,
    scope='Fresh reversed candidate/baseline pair; historical incomplete pair retained',
    order=['candidate','baseline'],replica=2,startup_only_observation=True,
    input_sha256={str(p):sha(p) for p in paths}),indent=2)+'\n')
PY
observer=''
cleanup_observer() {
    if [[ -n "$observer" ]]; then kill "$observer" 2>/dev/null || true;wait "$observer" 2>/dev/null || true;fi
}
trap cleanup_observer EXIT
index=0
for arm in candidate baseline; do
    allocation_gate "$((21600-index*10200))"
    d="$out/replica-2/$arm"
    test ! -e "$d"
    mkdir -p "$d"
    optimization=baseline
    if [[ "$arm" == candidate ]]; then optimization=dspark;fi
    python3 -u "$tools/startup_host_observer.py" --arm-output "$d" \
        --container "kimi-pr-perf-replica-2-$arm" \
        --output "$out/startup-$arm.jsonl" --timeout 3000 >"$out/startup-$arm-observer.log" 2>&1 &
    observer=$!
    python3 "$tools/allocation_guard.py" --journal "$d/allocation-guard.jsonl" --output "$d" -- \
        bash "$tools/benchmark_arm.sh" "$d" "$optimization" agentic "$client" "$source"
    wait "$observer";observer=''
    python3 - "$out/startup-$arm.jsonl" <<'PY'
import json,sys
records=[json.loads(line) for line in open(sys.argv[1])]
assert records[-1]['event'] in ('stopped_before_client_or_terminal_arm','stopped_before_client'), 'Startup observer did not stop before client'
PY
    index=$((index+1))
done
python3 - "$out" "$tools" <<'PY'
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[2])
from performance_gate import normalize_recipe
root=Path(sys.argv[1]);arms={};recipes=[]
for arm in ['candidate','baseline']:
    d=root/'replica-2'/arm
    assert json.loads((d/'performance-result.json').read_text())['status']=='collected'
    start=json.loads((d/'agentic-client-start.json').read_text())
    assert start['native_plan']['replay']['duration']==3600 and start['native_plan']['replay']['concurrency']==48
    recipes.append(normalize_recipe(json.loads((d/'recipe.json').read_text()),67372,arm))
    arms[arm]=json.loads((d/'agentic.json').read_text())
assert recipes[0]==recipes[1]
(root/'recovery-comparison.json').write_text(json.dumps(dict(
    status='native_agentic_reversed_recovery_pair_collected',replica=2,**arms,
    limits='One fresh reversed pair with startup-only host observations; original failed pair preserved. Full two-pair eligibility requires separately verified provenance.'),indent=2)+'\n')
PY
