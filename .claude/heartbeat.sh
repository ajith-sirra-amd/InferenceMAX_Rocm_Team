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
#   usage:  .claude/heartbeat.sh [sleep_seconds]      default 900 (15 min)
#
# Cadence set to 15 min on 2026-09-08 to cut token cost (~700-1100 per cycle at
# 9 min). Worst-case blind spot is 15 min, still far tighter than the W1 failure
# where a run died 7 min in and sat unnoticed for 2h48m.
#
# Deliberately cheap outside a reservation: preflight hard-stops after two date
# comparisons when out of window, touching neither GPU nor network.

# Read the whole script into memory before sleeping, so editing this file mid-run
# cannot corrupt a sleeping instance. Bash reads scripts incrementally by file
# offset; a 2026-09-08 edit to a sleeping heartbeat made it resume mid-token and
# die with "syntax error near unexpected token".
SECS="${1:-900}"
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
RS0=$(gh run list --repo ajith-sirra-amd/InferenceMAX_Rocm_Team --workflow "End-to-End Tests" --limit 1 --json status -q '.[].status' 2>/dev/null)
gh run list --repo ajith-sirra-amd/InferenceMAX_Rocm_Team --workflow "End-to-End Tests" \
   --limit 1 --json databaseId,status,conclusion \
   -q '.[]|"  run \(.databaseId) \(.status) \(.conclusion // "")"' 2>&1

# ONE bounded log read per wake, cached. Previously this called docker logs 7x,
# each re-reading a 100MB+ log -- CPU stolen from the vLLM workers mid-measurement,
# which risks skewing the number being collected.
HB=/tmp/hb.log
docker logs --tail 4000 bmk-server > "$HB" 2>/dev/null
# KV line appears early, so --tail misses it; capture once per run and cache.
# Keyed on container start time: the cache went stale across runs and reported
# the PREVIOUS run's fingerprint, which is exactly how a result gets misattributed.
KVF=/tmp/hb.kv
CID=$(docker inspect -f '{{.State.StartedAt}}' bmk-server 2>/dev/null)
if [ "$(cat /tmp/hb.kv.cid 2>/dev/null)" != "$CID" ] || [ ! -s "$KVF" ]; then
  docker logs bmk-server 2>/dev/null | grep -aoE "GPU KV cache size: [0-9,]+ tokens" | head -1 > "$KVF"
  printf '%s' "$CID" > /tmp/hb.kv.cid
fi

grep -aoE "Phase [a-z]+ progress \| returned=[0-9]+/[0-9]+ \| sent=[0-9]+ \| in_flight=[0-9]+ \| errors=[0-9]+ \| elapsed=[0-9.]+s" "$HB" | tail -1 | sed 's/^/  /'
grep -a "srv  " "$HB" | tail -1 | cut -c1-190 | sed 's/^/  /'
sed 's/^/  /' "$KVF" 2>/dev/null
grep -aoE "Throughput per GPU: [0-9]+ tok/s|exact_match.{0,24}" "$HB" | sort -u | tail -2 | sed 's/^/  /'

echo "--- node ---"
# numa_balancing MUST be 0. It resets to 1 on reboot, and with a ~1.8 TB host
# offload pool the kernel migrates that working set continuously - warmup crawls
# while GPUs look busy. Cost on 2026-09-09: four runs and ~7 h of a 13 h slot
# before it was found. First thing to check when a run is slow.
NB=$(cat /proc/sys/kernel/numa_balancing 2>/dev/null)
if [ "$NB" != "0" ]; then
  echo "  *** ALERT: numa_balancing=$NB (must be 0). Runs will crawl."
  echo "  ***   sudo sysctl -w kernel.numa_balancing=0"
  echo "  ***   echo 'kernel.numa_balancing = 0' | sudo tee /etc/sysctl.d/99-numa.conf"
else
  echo "  numa_balancing=0 ok"
fi
V=$(timeout 40 rocm-smi --showmemuse 2>/dev/null \
     | grep -oE "GPU Memory Allocated \(VRAM%\): [0-9]+" | awk '{if($NF>m)m=$NF}END{print m+0}')
BMK=$(docker ps --format '{{.Names}}' 2>/dev/null | grep -c '^bmk-server$')
echo "  vram_max=${V}%  bmk-server=$BMK"
# A run that is in_progress with NO container is a distinct failure state:
# the job is hung before Launch job script (e.g. actions/checkout). T291 sat
# 24 min that way with GPUs idle while four polls reported "loading".
if [ "$RS0" = "in_progress" ] && [ "$BMK" -eq 0 ] && [ "${V:-100}" -le 10 ]; then
  echo "  *** ALERT: run in_progress but NO container and GPUs idle."
  echo "  *** Job is hung before the server starts - check gh run view <id> steps."
  echo "  *** If stuck in checkout/setup: cancel and re-dispatch, it will not recover."
fi

# --- verdict line: compare against the reference numbers automatically -------
# BASE = T274 11,095 tok/s/GPU @ KV 28,653,478. Anchor 11,027. Noise +/-1.2%.
TP=$(grep -aoE "Throughput per GPU: [0-9]+" "$HB" | tail -1 | grep -oE "[0-9]+$")
EM=$(grep -aoE "exact_match\|[^|]*\| *[0-9.]+" "$HB" | tail -1 | grep -oE "[0-9.]+$")
EXT=$(grep -a "srv  " "$HB" | tail -1 | grep -oE "ext_cache_hit=[0-9.]+" | cut -d= -f2)
KVU=$(grep -a "srv  " "$HB" | tail -1 | grep -oE "kv_usage=[0-9.]+" | cut -d= -f2)
echo "--- verdict ---"
[ -n "$TP" ] && awk -v t="$TP" 'BEGIN{d=(t-11095)/11095*100; printf "  THROUGHPUT %s tok/s/GPU  vs T274 11,095 = %+.2f%%  %s\n", t, d, (d>1.2?"WIN (outside noise)":(d<-1.2?"LOSS":"inside +/-1.2% noise"))}'
[ -n "$EM" ] && awk -v e="$EM" 'BEGIN{printf "  GSM8K %s  %s\n", e, (e>=0.98?"PASS":"FAIL - investigate")}'
[ -n "$EXT" ] && echo "  ext_cache_hit=${EXT}%  kv_usage=${KVU}%   (T274 ref: 0->82.6% over the hour, kv_usage 45-55%)"
[ -z "$TP" ] && [ -z "$EM" ] && echo "  no result yet"

# --- heartbeat checks the AGENT --------------------------------------------
# Inside a slot the node must never be idle. If there is no run in flight and
# the GPUs are free, the agent has failed to dispatch - say so loudly.
RS=$(gh run list --repo ajith-sirra-amd/InferenceMAX_Rocm_Team --workflow "End-to-End Tests" \
     --limit 1 --json status -q '.[].status' 2>/dev/null)
if [ "$RS" = "completed" ] && [ "${V:-100}" -le 10 ]; then
  echo "  *** ALERT: NODE IDLE INSIDE SLOT - no run in flight, GPUs free."
  echo "  *** The agent has not dispatched. Record the last result and dispatch"
  echo "  *** the next queue item NOW (see RUN-CONTINUOUSLY RULES in EXPERIMENT-QUEUE.md)."
fi
# Backstop check: a recurring cron must exist, since this chain has died 4x today.
if ! grep -q '"cron"' /home/asirra/.claude/scheduled_tasks.json 2>/dev/null; then
  echo "  *** ALERT: no durable cron armed - the heartbeat is the ONLY wake-up."
fi
