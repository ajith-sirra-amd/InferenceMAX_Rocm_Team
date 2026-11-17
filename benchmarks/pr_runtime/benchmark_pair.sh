#!/usr/bin/env bash
set -euo pipefail
tools=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
out=${IPAT_PR_OUTPUT:?}
quality=${IPAT_PR_QUALITY_OUTPUT:?}
case "$out" in /share_nfs/gguasti/benchmarks/pr-perf-*) ;; *) exit 2 ;; esac
case "$quality" in /share_nfs/gguasti/benchmarks/pr-model-*) ;; *) exit 2 ;; esac
test ! -e "$out/pair-start.json"
mkdir -p "$out"
python3 "$tools/performance_gate.py" --quality "$quality" --manifest "${IPAT_PR_MANIFEST:?}" \
    --output "$out" --component-output "${IPAT_PR_COMPONENT_OUTPUT:-}"
if [[ "${IPAT_VALIDATE_QUALITY_ONLY:-0}" == 1 ]]; then exit 0; fi
export IPAT_PR_NUMBER=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["pr"])' "$IPAT_PR_MANIFEST")
for arm in baseline candidate; do
    optimization=baseline
    if [[ "$arm" == candidate ]]; then
        if [[ "$IPAT_PR_NUMBER" == 67372 ]]; then optimization=dspark; else optimization=attnres; fi
    fi
    mkdir -p "$out/$arm"
    bash "$tools/benchmark_arm.sh" "$out/$arm" "$optimization" fixed
done
python3 "$tools/benchmark_compare.py" "$out"
