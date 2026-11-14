#!/usr/bin/env bash
set -euo pipefail
tools=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
out=${IPAT_PR_OUTPUT:?};client=${IPAT_PR_AGENTIC_CLIENT:?};source=${IPAT_PR_AGENTIC_SOURCE:?}
IPAT_VALIDATE_QUALITY_ONLY=1 bash "$tools/benchmark_pair.sh"
export IPAT_PR_NUMBER=67372
for replica in 1 2; do
    # Reverse the second pair to expose order-sensitive warmup/system effects.
    arms=(baseline candidate)
    if [[ "$replica" == 2 ]]; then arms=(candidate baseline); fi
    for arm in "${arms[@]}"; do
        optimization=baseline
        if [[ "$arm" == candidate ]]; then optimization=dspark; fi
        mkdir -p "$out/replica-$replica/$arm"
        bash "$tools/benchmark_arm.sh" "$out/replica-$replica/$arm" "$optimization" agentic "$client" "$source"
    done
done
python3 "$tools/agentic_compare.py" "$out"
