#!/usr/bin/env bash
# heartbeat.sh — sleep, then emit a status snapshot.
#
# Run it with the Bash tool's run_in_background. When it exits, the harness
# re-invokes the agent with a task-notification, which is the wake-up. The agent
# then acts on the snapshot and launches the next heartbeat, forming a chain.
#
# This exists because the cron scheduler did NOT fire during W2 (9/7 23:30 →
# 9/8 07:22) and 7h51m of a 12h reservation was lost. Background-task completion
# notifications have fired reliably all session, so the chain is built on those.
#
#   usage:  .claude/heartbeat.sh [sleep_seconds]      default 540 (9 min)
#
# Deliberately cheap outside a reservation: preflight hard-stops after two date
# comparisons when out of window, touching neither GPU nor network.

SECS="${1:-540}"
REPO=/home/asirra/imx-repo
sleep "$SECS"

echo "=== HEARTBEAT $(TZ=Asia/Kolkata date '+%Y-%m-%d %H:%M IST (%a)') ==="

cd "$REPO" || exit 0

# Window + kill-switch first. Outside a slot this touches nothing.
./.claude/preflight.sh 0 2>&1 | sed 's/\x1b\[[0-9;]*m//g' \
  | grep -E "reservation|kill-switch|remaining|VERDICT|==>" | sed 's/^/  /'

# If we are outside the window, stop here — no node access.
# NOTE: preflight colourises its output, so ANSI codes must be stripped before
# grepping. Without the sed this never matched and the heartbeat silently
# reported "outside reservation" while inside one.
if ! ./.claude/preflight.sh 0 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -q "PASS  reservation"; then
  echo "  (outside reservation — no further checks)"
  exit 0
fi

echo "--- run ---"
gh run list --repo ajith-sirra-amd/InferenceMAX_Rocm_Team --workflow "End-to-End Tests" \
   --limit 1 --json databaseId,status,conclusion \
   -q '.[]|"  run \(.databaseId) \(.status) \(.conclusion // "")"' 2>&1

echo "--- progress ---"
docker logs bmk-server 2>/dev/null \
  | grep -aoE "Phase [a-z]+ progress \| returned=[0-9]+/[0-9]+ \| sent=[0-9]+ \| in_flight=[0-9]+ \| errors=[0-9]+ \| elapsed=[0-9.]+s" \
  | tail -1 | sed 's/^/  /'
docker logs bmk-server 2>/dev/null | grep -a "srv  " | tail -1 | cut -c1-190 | sed 's/^/  /'
docker logs bmk-server 2>/dev/null \
  | grep -aoE "GPU KV cache size: [0-9,]+ tokens|Throughput per GPU: [0-9]+ tok/s|exact_match.{0,24}" \
  | sort -u | tail -3 | sed 's/^/  /'

echo "--- node ---"
V=$(timeout 40 rocm-smi --showmemuse 2>/dev/null \
     | grep -oE "GPU Memory Allocated \(VRAM%\): [0-9]+" | awk '{if($NF>m)m=$NF}END{print m+0}')
echo "  vram_max=${V}%  bmk-server=$(docker ps --format '{{.Names}}' 2>/dev/null | grep -c '^bmk-server$')"
