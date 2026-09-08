#!/usr/bin/env bash
# preflight.sh — validator that must PASS before ANY dispatch, build, or GPU/CPU work
# on the shared node smci355-1-b23-07.
#
#   usage:  .claude/preflight.sh <minutes_needed> [--dispatch]
#           .claude/preflight.sh 105            # perf run
#           .claude/preflight.sh 70             # GSM8K-200
#           .claude/preflight.sh 25             # image build (CPU only, still slot-bound)
#
# exit 0 = every gate passed, safe to proceed.  exit 1 = ABORT.
# Every check that has actually cost us a slot or a node is represented here.

set -uo pipefail
REPO=/home/asirra/imx-repo
NEED_MIN="${1:-0}"
REPOSLUG=ajith-sirra-amd/InferenceMAX_Rocm_Team
BRANCH=chore/sa-agentx-v1.0
LAUNCHER=$REPO/upstream/InferenceX/benchmarks/single_node/agentic/kimik3_fp4_mi355x_mtp.sh
YAML=$REPO/upstream/InferenceX/configs/amd-master.yaml

# W3 and W4 abut with no gap -> merged into one 13h block.
WINDOWS=(
  "2026-09-07 08:30|2026-09-07 12:30|W1"
  # W2 extended 11:30 -> 13:30 on 2026-09-08 at 11:33 IST, on the owner's explicit
  # instruction ("I will negotiate time slot" -> "Run" -> "Start GH Run with None
  # .. now"). The owner holds the reservation and controls the GH runner, which is
  # the independent interlock. Agent did NOT choose to extend on its own judgement.
  "2026-09-07 23:30|2026-09-08 13:30|W2 (extended by owner)"
  "2026-09-08 23:30|2026-09-09 12:30|W3+W4"
  "2026-09-10 08:00|2026-09-11 08:00|W5"
)

FAIL=0
ok()   { printf '  \033[32mPASS\033[0m  %-28s %s\n' "$1" "$2"; }
bad()  { printf '  \033[31mFAIL\033[0m  %-28s %s\n' "$1" "$2"; FAIL=1; }
warn() { printf '  \033[33mWARN\033[0m  %-28s %s\n' "$1" "$2"; }

echo "PREFLIGHT  $(TZ=Asia/Kolkata date '+%Y-%m-%d %H:%M IST (%a)')  need=${NEED_MIN}m"
echo

# 1. KILL SWITCH -------------------------------------------------------------
ST=$(grep -oE 'Now[[:space:]]*:[[:space:]]*[A-Za-z]+' "$REPO/Run_Status.txt" 2>/dev/null | tail -1 | awk '{print $NF}')
if [ "${ST,,}" = "run" ]; then ok "kill-switch" "Run_Status.txt = Run"
else bad "kill-switch" "Run_Status.txt = '${ST:-unreadable}' (expected Run)"; fi

# 2. RESERVATION WINDOW ------------------------------------------------------
NOW=$(TZ=Asia/Kolkata date +%s); IN=0; REM=0; WNAME=""; WEND=""
for w in "${WINDOWS[@]}"; do
  IFS='|' read -r s e n <<< "$w"
  S=$(TZ=Asia/Kolkata date -d "$s" +%s); E=$(TZ=Asia/Kolkata date -d "$e" +%s)
  if [ "$NOW" -ge "$S" ] && [ "$NOW" -lt "$E" ]; then
    IN=1; REM=$(( (E-NOW)/60 )); WNAME=$n; WEND=$(TZ=Asia/Kolkata date -d "$e" '+%a %H:%M'); break
  fi
done
if [ "$IN" -eq 1 ]; then ok "reservation" "$WNAME, closes $WEND, ${REM}m left"
else
  NXT=""
  for w in "${WINDOWS[@]}"; do IFS='|' read -r s e n <<< "$w"; S=$(TZ=Asia/Kolkata date -d "$s" +%s)
    if [ "$NOW" -lt "$S" ]; then NXT="$n opens in $(( (S-NOW)/60 ))m"; break; fi; done
  bad "reservation" "OUTSIDE slot — ${NXT:-no further windows}"
  # HARD STOP. Everything below this line touches the node: rocm-smi queries the
  # GPU driver, git fetch and gh hit the network, docker inspects the daemon.
  # Outside a reservation we do NONE of it -- the owner's rule is no GPU *and*
  # no CPU work between slots, and a poll running every 11 minutes would
  # otherwise breach that around the clock.
  echo
  echo "  (skipped all node-touching gates — outside reservation)"
  echo "  ==> PREFLIGHT FAIL — ABORT, do not dispatch"
  exit 1
fi

# 3. TIME BUDGET — never strand a job at a slot boundary ---------------------
if [ "$IN" -eq 1 ] && [ "$NEED_MIN" -gt 0 ]; then
  if [ "$REM" -ge "$NEED_MIN" ]; then ok "time budget" "need ${NEED_MIN}m, have ${REM}m"
  else bad "time budget" "need ${NEED_MIN}m, only ${REM}m — would be killed at slot close"; fi
fi

# 4. GPUs ACTUALLY FREE — launcher opens with wait_for_amd_gpu_clean (<=10%) --
VRAM=$(timeout 45 rocm-smi --showmemuse 2>/dev/null \
        | grep -oE "GPU Memory Allocated \(VRAM%\): [0-9]+" \
        | awk '{if ($NF>m) m=$NF} END{print m+0}')
if [ -z "$VRAM" ]; then warn "gpu free" "rocm-smi unreadable/timed out — verify by hand"
elif [ "$VRAM" -le 10 ]; then ok "gpu free" "max VRAM ${VRAM}%"
else bad "gpu free" "max VRAM ${VRAM}% — wait_for_amd_gpu_clean will fail (needs <=10%)"; fi

# 5. NO ZOMBIE KFD ENTRIES — the stranded-VRAM failure from T273/T275/T276 ----
Z=$(timeout 40 rocm-smi --showpids 2>/dev/null | grep -c UNKNOWN)
if [ "${Z:-0}" -eq 0 ]; then ok "no zombies" "0 orphan KFD entries"
else bad "no zombies" "$Z orphan KFD entries — node not released, wait for reclaim"; fi

# 6. NO STALE CONTAINER ------------------------------------------------------
C=$(docker ps -a --format '{{.Names}}' 2>/dev/null | grep -c '^bmk-server$')
if [ "${C:-0}" -eq 0 ]; then ok "no stale container" "bmk-server absent"
else warn "no stale container" "bmk-server exists — docker rm -f it first"; fi

# 7. NOTHING ALREADY IN FLIGHT ----------------------------------------------
RS=$(gh run list --repo $REPOSLUG --workflow "End-to-End Tests" --limit 1 \
     --json status -q '.[].status' 2>/dev/null)
if [ "$RS" = "completed" ] || [ -z "$RS" ]; then ok "no run in flight" "last run: ${RS:-none}"
else bad "no run in flight" "a run is '$RS' — do not double-dispatch"; fi

# 8. WORK IS PUSHED — dispatch resolves the BRANCH, not the worktree.
#    Unpushed edits silently do not take effect. This has bitten us.
B=$(git -C $REPO rev-parse --abbrev-ref HEAD)
if [ "$B" = "$BRANCH" ]; then ok "branch" "$B"; else bad "branch" "on '$B', expected $BRANCH"; fi
if [ -z "$(git -C $REPO status --porcelain)" ]; then ok "worktree clean" "nothing uncommitted"
else bad "worktree clean" "uncommitted edits — dispatch would run the OLD config"; fi
git -C $REPO fetch -q origin $BRANCH 2>/dev/null
L=$(git -C $REPO rev-parse HEAD 2>/dev/null); R=$(git -C $REPO rev-parse origin/$BRANCH 2>/dev/null)
if [ "$L" = "$R" ]; then ok "pushed" "local == origin/$BRANCH"
else bad "pushed" "local != origin/$BRANCH — push before dispatching"; fi

# 9. LAUNCHER INTEGRITY ------------------------------------------------------
if bash -n "$LAUNCHER" 2>/dev/null; then ok "launcher syntax" "bash -n clean"
else bad "launcher syntax" "bash -n FAILED"; fi
W=$(grep -c wait_for_server_ready "$LAUNCHER")
if [ "$W" -ge 2 ]; then ok "wait_for_server_ready" "present (x$W)"
else bad "wait_for_server_ready" "MISSING — an edit dropped it"; fi

# 10. YAML PARSES + REFERENCED IMAGE EXISTS LOCALLY -------------------------
if python3 -c "import yaml,sys;yaml.safe_load(open('$YAML'))" 2>/dev/null; then ok "yaml parses" "ok"
else bad "yaml parses" "amd-master.yaml is invalid"; fi
IMG=$(awk '/^kimik3-fp4-mi355x-vllm-agentic-mtp:/{f=1} f&&/^  image:/{print $2;exit}' "$YAML")
if [ -n "$IMG" ] && docker image inspect "$IMG" >/dev/null 2>&1; then ok "image present" "$IMG"
else bad "image present" "${IMG:-<none>} not found locally — build it first"; fi

# 11. ECHO THE ONE-VARIABLE SURFACE for human review ------------------------
echo
echo "  config surface (confirm ONE variable changed vs the comparison run):"
grep -oE 'EVAL_ONLY="\$\{EVAL_ONLY:-[a-z]+\}' "$LAUNCHER" | head -1 | sed 's/^/    /'
grep -oE 'MBT_DEFAULT="\$\{K3_MNBT:-[0-9]+\}' "$LAUNCHER" | head -1 | sed 's/^/    /'
grep -oE 'GPU_MEM_UTIL="\$\{K3_GMU:-[0-9.]+\}' "$LAUNCHER" | head -1 | sed 's/^/    /'
grep -oE 'VLLM_USE_BREAKABLE_CUDAGRAPH="\$\{[^}]*\}' "$LAUNCHER" | head -1 | sed 's/^/    /'
echo "    image: $IMG"
awk '/^kimik3-fp4-mi355x-vllm-agentic-mtp:/{f=1} f&&/^      - \{ tp:/{print "    " $0; exit}' "$YAML" | cut -c1-150

echo
if [ "$FAIL" -eq 0 ]; then echo "  ==> PREFLIGHT PASS — safe to proceed"; exit 0
else echo "  ==> PREFLIGHT FAIL — ABORT, do not dispatch"; exit 1; fi
