#!/usr/bin/env bash
# dispatch.sh — the ONLY sanctioned way to start a benchmark run.
# Everything runs through the GitHub runner; the owner starts/stops that runner
# around each reservation, which is the second, independent interlock.
#
#   usage:  .claude/dispatch.sh <minutes_needed> "<label>"
#           .claude/dispatch.sh 70  "T278 v2 GSM8K-200"
#           .claude/dispatch.sh 105 "T278 v2 perf C72"
#
# exit 0 = dispatched, exit 1 = refused.

set -uo pipefail
REPO=/home/asirra/imx-repo
REPOSLUG=ajith-sirra-amd/InferenceMAX_Rocm_Team
BRANCH=chore/sa-agentx-v1.0
CFG="upstream/InferenceX/configs/amd-master.yaml"
KEY="kimik3-fp4-mi355x-vllm-agentic-mtp"
NEED="${1:?usage: dispatch.sh <minutes_needed> \"<label>\"}"
LABEL="${2:-unlabelled}"

cd "$REPO" || exit 1

# gate 1 — full preflight (slot, time budget, GPUs, zombies, pushed, config)
./.claude/preflight.sh "$NEED" || { echo; echo "ABORT: preflight failed — not dispatching '$LABEL'"; exit 1; }

# gate 2 (removed 2026-09-10): was a stale duplicate of preflight's gate 7,
# and buggy on two counts -- --limit 1 only sees the newest run (misses an
# in-flight run superseded by a shorter one finishing after it), and no
# runner-name filtering (fired on ajith-glm-5.2, a DIFFERENT physical node,
# mi355x-amd_p02_g17 vs our mi355x-amd_b23_07). preflight.sh's gate 7 above
# already scans the last 15 runs AND filters to our own runner; it is the
# authoritative check. Do not re-add a duplicate here.

echo
echo "dispatching: $LABEL  (needs ${NEED}m)"
gh workflow run "End-to-End Tests" --repo $REPOSLUG --ref $BRANCH \
   -f generate-cli-command="test-config --config-files $CFG --config-keys $KEY" \
   || { echo "ABORT: gh workflow run failed"; exit 1; }

sleep 12
RID=$(gh run list --repo $REPOSLUG --workflow "End-to-End Tests" --limit 1 --json databaseId -q '.[].databaseId')
printf '%s | dispatched %s | run %s\n' "$(TZ=Asia/Kolkata date '+%Y-%m-%d %H:%M IST')" "$LABEL" "$RID"
