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

# gate 2 — never double-dispatch
PRE=$(gh run list --repo $REPOSLUG --workflow "End-to-End Tests" --limit 1 --json databaseId,status \
      -q '.[]|"\(.databaseId) \(.status)"' 2>/dev/null)
case "$PRE" in
  *queued*|*in_progress*|*requested*|*waiting*)
    echo "ABORT: a run is already active/queued ($PRE)"; exit 1;;
esac

echo
echo "dispatching: $LABEL  (needs ${NEED}m)"
gh workflow run "End-to-End Tests" --repo $REPOSLUG --ref $BRANCH \
   -f generate-cli-command="test-config --config-files $CFG --config-keys $KEY" \
   || { echo "ABORT: gh workflow run failed"; exit 1; }

sleep 12
RID=$(gh run list --repo $REPOSLUG --workflow "End-to-End Tests" --limit 1 --json databaseId -q '.[].databaseId')
printf '%s | dispatched %s | run %s\n' "$(TZ=Asia/Kolkata date '+%Y-%m-%d %H:%M IST')" "$LABEL" "$RID"
