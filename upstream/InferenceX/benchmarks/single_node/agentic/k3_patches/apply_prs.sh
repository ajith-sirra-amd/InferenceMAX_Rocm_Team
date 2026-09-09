#!/usr/bin/env bash
# Per-PR runtime patching. APPLY_PR_<num>=1 applies runtime/<num>.diff into
# site-packages. Everything defaults to 0, so this is a no-op unless a run
# opts in. Never fatal: a PR that will not apply degrades the run to the
# baseline rather than killing it, and the [pr] gate line records which way
# it went so a number is never silently mis-attributed.
#
# Only valid on an image WITHOUT the PRs baked in. Against a pre-patched image
# the hunks are already present, `patch --forward` exits non-zero, and the PR
# is reported failed when nothing is actually wrong.
set -uo pipefail
RT_DIR="$(cd "$(dirname "$0")" && pwd)/runtime"
[ -d "$RT_DIR" ] || { echo "[pr] no runtime dir -- nothing to apply"; exit 0; }
SP=$(python3 -c 'import vllm,os;print(os.path.dirname(os.path.dirname(vllm.__file__)))' 2>/dev/null || echo "")
[ -n "$SP" ] || { echo "[pr] could not resolve site-packages -- nothing applied"; exit 0; }
on=""; off=""; bad=""
for f in "$RT_DIR"/*.diff; do
    [ -f "$f" ] || continue
    n=$(basename "$f" .diff)
    v="APPLY_PR_${n}"
    if [ "${!v:-0}" != "1" ]; then off="$off $n"; continue; fi
    # dry-run first: patch is all-or-nothing per file, so a partial apply would
    # leave site-packages in a state no manifest describes.
    if patch -p1 -d "$SP" --forward --batch --dry-run < "$f" >/dev/null 2>&1 \
       && patch -p1 -d "$SP" --forward --batch < "$f" >/dev/null 2>&1; then
        on="$on $n"
    else
        bad="$bad $n"
    fi
done
echo "[pr] on:${on:- none} off:${off:- none} failed:${bad:- none} site=$SP"
exit 0
