#!/usr/bin/env bash
set -euo pipefail
tools=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
out=${IPAT_PR_OUTPUT:?}
quality=${IPAT_PR_QUALITY_OUTPUT:?}
case "$out" in /share_nfs/gguasti/benchmarks/pr-perf-*) ;; *) exit 2 ;; esac
case "$quality" in /share_nfs/gguasti/benchmarks/pr-model-*) ;; *) exit 2 ;; esac
test ! -e "$out/pair-start.json"
mkdir -p "$out"
python3 "$tools/model_compare.py" "$quality"
python3 - "$quality" "${IPAT_PR_MANIFEST:?}" "$out" <<'PY'
import json,sys,hashlib
from pathlib import Path
q,p,o=map(Path,sys.argv[1:]);pair=json.loads(p.read_text());comparison=json.loads((q/'model-comparison.json').read_text())
assert pair['pr']==67372,'AttnRes frozen candidate has failed its numerical gate'
assert comparison['candidate_correct']>=comparison['baseline_correct'],'Screen shows a correctness regression'
for arm in ['baseline','candidate']:
    manifest=json.loads((q/arm/'runtime-manifest.json').read_text())
    assert manifest['pair']==pair,'Performance images differ from screened images'
(o/'pair-start.json').write_text(json.dumps(dict(pr=pair['pr'],quality_output=str(q),quality_comparison_sha256=hashlib.sha256((q/'model-comparison.json').read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),limits='Matched sampled48 screen prerequisite only; full accuracy remains unqualified'),indent=2)+'\n')
PY
if [[ "${IPAT_VALIDATE_QUALITY_ONLY:-0}" == 1 ]]; then exit 0; fi
export IPAT_PR_NUMBER=67372
for arm in baseline candidate; do
    optimization=baseline
    if [[ "$arm" == candidate ]]; then optimization=dspark; fi
    mkdir -p "$out/$arm"
    bash "$tools/benchmark_arm.sh" "$out/$arm" "$optimization" fixed
done
python3 "$tools/benchmark_compare.py" "$out"
