#!/usr/bin/env bash
set -euo pipefail
set -x
source "$(dirname "$0")/../../benchmark_lib.sh"
wait_for_amd_gpu_clean

# GSM8K ACCURACY GATE. The nightly stack (overlay + #53940 a4w4 MoE kernels +
# chunk 16384) is a large numerics change that went straight to throughput with
# RUN_EVAL=false on T188/T189/T190. 9,482 tok/s/GPU is currently unvalidated.
# EVAL_ONLY=true runs GSM8K instead of the benchmark; EVAL_LIMIT=200 keeps it short.
# T251 gate PASSED (GSM8K 0.995 on rec-no53940) -- back to false.
export EVAL_ONLY="${EVAL_ONLY:-false}"
export EVAL_LIMIT="${EVAL_LIMIT:-200}"
export AIPERF_EXPERIMENTAL_FAST=0
export AIPERF_WARMUP_REQUESTS_PER_LANE=1
check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE

DP_SIZE=1
export DP_SIZE
TOTAL_RANKS=$(( TP * DP_SIZE ))

if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi

rocm-smi || true
amd-smi || true
resolve_trace_source
install_agentic_deps

# ---- K3 nightly overlay ------------------------------------------------------
# SA's 8,953 tok/s/GPU at C52 (run 33324464095, node amds_01) is NOT the bare
# nightly -- it is nightly-46638857 plus a 264 KB patch overlay applied into
# site-packages. The overlay touches models/kimi_k3/amd/{mla,kda,linear,
# latent_moe_runner}.py, layers/mla.py, mla_attention.py, fused_moe/runner/*,
# platforms/rocm.py and envs.py. Without it the nightly is missing every
# Kimi-K3 kernel path and will regress, not improve.
#
# The overlay is cut against nightly-46638857 specifically. On any other image
# the dry-run fails. REQUIRE_K3_OVERLAY defaults to 1 here so that case is a
# hard failure rather than a silent unpatched run producing a misleading number.
K3_PATCH_DIR="$(cd "$(dirname "$0")" && pwd)/k3_patches"

# Pre-baked image short-circuit. kimi-k3-vllm:v4 ships the overlay AND the PR
# stack already applied and drops a manifest at /etc/k3-image-manifest. Its
# presence is the contract: site-packages is already patched, so re-applying
# would fail the dry-run and -- with REQUIRE_K3_OVERLAY=1 -- kill the run.
# One overlay for every concurrency in that image; no C1-vs-C52 split.
if [ -f /etc/k3-image-manifest ]; then
    K3_OVERLAY_APPLIED=1
    export SKIP_KIMI_PATCHES=1
    echo "[k3-overlay] baked into image -- runtime patching SKIPPED"
    echo "[pr-stack] baked into image -- runtime patching SKIPPED"
    sed 's/^/[k3-image] /' /etc/k3-image-manifest
else
# ONE overlay for every concurrency, matching kimi-k3-vllm:v4. The c1 cut is
# retired: it is a different, older snapshot, not a C1 tuning. Folding C1 onto
# c16_c52 gains latent_moe_runner, the KDA chunk kernels and a much newer
# simple_kv_offload/manager, and loses an online-quantization subsystem that C72
# demonstrably does not need. Numerics-affecting for C1 -> GSM8K gates it.
K3_OVERLAY_PATCH="${K3_OVERLAY_PATCH:-$K3_PATCH_DIR/vllm_nightly_46638857_k3_c16_c52_current.patch}"
REQUIRE_K3_OVERLAY="${REQUIRE_K3_OVERLAY:-1}"
K3_OVERLAY_APPLIED=0

# ---- Overlay ablation (A/B/C/D/E) -------------------------------------------
# k3_patches/overlay_split/ carries the SAME 264 KB overlay partitioned by
# concern. Recombined it is byte-identical to the monolith (264,116 B, 199
# hunks, 34 files), and all five groups were verified to apply independently on
# a pristine 46638857 tree, so any subset is a legal experiment.
#
#   A  dcp a2a buffer pool          1 file    3,555 B
#   B  spec-decode cudagraph        3 files   4,434 B
#   C  kv-offload + cache manager   9 files  76,944 B
#   D  ROCm AITER MLA backend       5 files  76,806 B
#   E  Kimi-K3 model path          16 files 102,377 B
#
# OVERLAY_GROUPS=ABCDE (default) is equivalent to the monolith. Drop a letter to
# ablate that group. Purpose: prune to the minimum set that preserves 10,632, so
# there is less to carry forward and upstream.
K3_OVERLAY_SPLIT="${K3_OVERLAY_SPLIT:-0}"
OVERLAY_GROUPS="${OVERLAY_GROUPS:-ABCD}"
if [ "$K3_OVERLAY_SPLIT" = "1" ]; then
    SITE_PKGS=$(python3 -c 'import vllm,os;print(os.path.dirname(os.path.dirname(vllm.__file__)))')
    SPLIT_DIR="$K3_PATCH_DIR/overlay_split"
    SPLIT_OK=0; SPLIT_MISS=""
    for _g in A B C D E; do
        case "$OVERLAY_GROUPS" in *"$_g"*) ;; *) continue ;; esac
        _gp=$(ls "$SPLIT_DIR"/${_g}_*.patch 2>/dev/null | head -1)
        [ -n "$_gp" ] || { echo "[overlay-split] MISSING group $_g" >&2; exit 1; }
        if ( cd "$SITE_PKGS" && patch -p1 --forward --batch < "$_gp" ) >/dev/null 2>&1; then
            SPLIT_OK=$(( SPLIT_OK + 1 ))
        else
            SPLIT_MISS="$SPLIT_MISS $_g"
        fi
    done
    echo "[overlay-split] groups=$OVERLAY_GROUPS applied=$SPLIT_OK failed:${SPLIT_MISS:- none}"
    [ -n "$SPLIT_MISS" ] && [ "$REQUIRE_K3_OVERLAY" = "1" ] && exit 1
    K3_OVERLAY_APPLIED=1
elif [ -f "$K3_OVERLAY_PATCH" ]; then
    SITE_PKGS=$(python3 -c 'import vllm,os;print(os.path.dirname(os.path.dirname(vllm.__file__)))')
    if ( cd "$SITE_PKGS" && patch -p1 --forward --batch --dry-run < "$K3_OVERLAY_PATCH" ) \
            >/tmp/k3_overlay_dryrun.log 2>&1; then
        echo "[k3-overlay] applying $(basename "$K3_OVERLAY_PATCH") into $SITE_PKGS"
        if ( cd "$SITE_PKGS" && patch -p1 --forward --batch < "$K3_OVERLAY_PATCH" ); then
            K3_OVERLAY_APPLIED=1
        elif [ "$REQUIRE_K3_OVERLAY" = "1" ]; then
            echo "[k3-overlay] APPLY FAILED" >&2; exit 1
        fi
    else
        echo "[k3-overlay] does not match this image: $K3_OVERLAY_PATCH" >&2
        head -40 /tmp/k3_overlay_dryrun.log >&2 || true
        python3 -c 'import vllm;print("vllm",vllm.__version__)' >&2 || true
        [ "$REQUIRE_K3_OVERLAY" = "1" ] && exit 1
    fi
elif [ "$REQUIRE_K3_OVERLAY" = "1" ]; then
    echo "[k3-overlay] missing: $K3_OVERLAY_PATCH" >&2; exit 1
fi
echo "[k3-overlay] applied=$K3_OVERLAY_APPLIED conc=$CONC"
# The legacy patch script edits files the overlay also carries; running it after
# a successful overlay shifts context and silently breaks it.
if [ "$K3_OVERLAY_APPLIED" = "1" ]; then export SKIP_KIMI_PATCHES=1; fi

# ---- Upstream PR stack, layered ON TOP of the K3 overlay ---------------------
# Goal is 12,500 tok/s/GPU; the overlay alone is worth ~8,953 on SA's numbers.
# These are open vLLM PRs, all PURE PYTHON (no csrc/), so `patch -p1` into
# site-packages is sufficient -- no image rebuild, no Docker push.
#
#   #53940  a4w4 flydsl kernels for Kimi-K3 (_aiter_ops, rocm_aiter_moe,
#           oracle/mxfp4, envs) -- AMD MoE path
#   #50813  Opt-in K3 SiTUv2 A8W4 routed MoE (quark_moe) -- our script already
#           exports VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1 and AITER_SITUV2_A8W4=1,
#           so we were setting flags for a code path that was not present.
#
# DISCARDED #54095 (aiter per-stream workspace): its cudagraph_utils.py hunk 2
# is cut against a newer tree and FAILED at line 362 on 46638857 (T187, T188,
# T190 all logged it as skipped). Dead weight, removed.
#
# Excluded deliberately: #53154 and #37682 both edit files the K3 overlay
# rewrites (amd/mla.py, layers/mla.py, rocm_aiter_mla.py) and will not apply on
# top of it; #50647 and #54255 are the NVIDIA path.
#
# NON-FATAL by design, unlike the K3 overlay: a PR that stops applying should
# degrade to the overlay-only baseline, not kill the run. The [pr-stack] gate
# line records which way it went, so a number is never silently mis-attributed.
# Per-file, not one monolithic patch. T187 proved why: cudagraph_utils.py Hunk #2
# FAILED at 362 (it is cut against a newer tree than 46638857) and `patch` is
# all-or-nothing per invocation, so that ONE hunk vetoed all five files. Applying
# each file separately means a stale hunk costs only its own file.
PR_STACK_DIR="${PR_STACK_DIR:-$K3_PATCH_DIR/pr_stack}"
PR_STACK_APPLIED=0
PR_STACK_SKIPPED=""
if [ "${APPLY_PR_STACK:-1}" = "1" ] && [ "$K3_OVERLAY_APPLIED" = "1" ] && [ -d "$PR_STACK_DIR" ]; then
    for _pp in "$PR_STACK_DIR"/*.patch; do
        [ -f "$_pp" ] || continue
        if ( cd "$SITE_PKGS" && patch -p1 --forward --batch --dry-run < "$_pp" ) >/dev/null 2>&1; then
            ( cd "$SITE_PKGS" && patch -p1 --forward --batch < "$_pp" ) >/dev/null 2>&1 \
                && PR_STACK_APPLIED=$(( PR_STACK_APPLIED + 1 ))
        else
            PR_STACK_SKIPPED="$PR_STACK_SKIPPED $(basename "$_pp")"
        fi
    done
fi
echo "[pr-stack] applied=$PR_STACK_APPLIED files, skipped:${PR_STACK_SKIPPED:- none} (#53940 a4w4-flydsl; #50813 pruned -- dead code for this model)"
fi

if [ -n "${DCP_SIZE:-}" ]; then
    DCP_SOURCE=matrix
else
    if [ "$CONC" -le 4 ]; then DCP_SIZE=1; else DCP_SIZE=8; fi
    DCP_SOURCE=conc-fallback
fi
export DCP_SIZE
echo "[dcp] size=$DCP_SIZE source=$DCP_SOURCE conc=$CONC"

export VLLM_ROCM_AITER_MLA_ASM_PADDING=asm
export VLLM_ROCM_USE_AITER=1
export SAFETENSORS_FAST_GPU=1
export VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1
export AITER_BF16_FP8_MOE_BOUND=0
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export GPU_ARCHS=gfx950
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION="${VLLM_ROCM_QUICK_REDUCE_QUANTIZATION:-NONE}"
export AITER_SITUV2_A8W4=1
export HSA_NO_SCRATCH_RECLAIM=1

# T234: HSA_STATUS_ERROR_OUT_OF_RESOURCES workaround -- BARE IMAGES ONLY.
#
# On unpatched nightly-7c5dc571 at C52/DCP8 with kv-offloading=none, the GPU
# queue aborts on resource exhaustion and takes the worker with it
# (SA run 33596998428, Worker_TP2_DCP2, 07:03:06):
#
#   :0:rocdevice.cpp:3715: Callback: Queue 0x70e817200000 Aborting with error :
#   HSA_STATUS_ERROR_OUT_OF_RESOURCES: The runtime failed to allocate the
#   necessary resources
#   -> hipErrorUnknown
#   -> segfault in TensileLite::GetDevice / torch.cuda.current_device
#   -> Worker proc died -> EngineDeadError
#
# The segfault frames are CORPSE FRAMES. Once the HIP context is dead any
# device-property query faults, which is why two unrelated call sites (Triton
# autotune and hipBLASLt Tensile) both crashed in device lookups. The cause is
# the queue allocation failure that precedes them.
#
# Warmup had completed 107/107 with errors=0; it died at the warmup->profiling
# handoff. With offload disabled the whole KV working set sits on GPU, which is
# what exhausted the queue allocation.
#
# gmu 0.85 leaves headroom for queue/scratch. Direction matters: the ledger only
# ever pushed gmu UP at C52 and it broke both times -- T157 gmu 0.95 hung the
# engine, T166 gmu 0.92 gave 0/103 in warmup. Downward is untested.
#
# GATED on the absence of /etc/k3-image-manifest: bare images take the 0.88
# override, patched images fall through to the GPU_MEM_UTIL default set below.
# That default was 0.9 for the whole prune ladder and is 0.85 as of T242 -- so
# do NOT read this branch as pinning patched images to 0.9 any more. The single
# authoritative value is the "[gmu] gpu_memory_utilization=..." line printed
# just before the serve command; trust that, not this one.
if [ ! -f /etc/k3-image-manifest ]; then
    GPU_MEM_UTIL_OVERRIDE="${GPU_MEM_UTIL_OVERRIDE:-0.88}"   # 0.88 CONFIRMED working (user, SA C52)
    echo "[gmu] bare image -- override ${GPU_MEM_UTIL_OVERRIDE} for HSA_STATUS_ERROR_OUT_OF_RESOURCES"
else
    echo "[gmu] patched image -- using script default (see [gmu] line below)"
fi
export VLLM_K3_KDA_SAFE_STAGES=1
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1

export VLLM_ENGINE_READY_TIMEOUT_S=7200
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
export PYTHONNOUSERSITE=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600

# ---- Profiling ---------------------------------------------------------------
# T202 killed the torch-profiler design. nightly-46638857 has NO
# VLLM_TORCH_PROFILER_DIR and NO start_profile handler in
# entrypoints/openai/api_server.py -- the server logged "Unknown vLLM
# environment variable detected" and the endpoint never existed, so the run
# produced zero traces. Do not retry that path on this image.
#
# rocprofv3 IS in the image (/opt/rocm/bin/rocprofv3) and the tree carries
# VLLM_NVTX_SCOPES_FOR_PROFILING hooks meant for exactly this. So PROFILE=1 now
# wraps the SERVER process in rocprofv3 kernel tracing.
#
# --kernel-trace only: a full HIP-API trace at batch 72 is enormous and the
# question here is which GPU kernels own the decode step, not which host calls
# were made. Output is one CSV per rank under $RESULT_DIR, which survives the
# artifact upload (the agentic_* artifact is exactly RESULT_DIR).
#
# Still caps the workload: PROFILE_PROMPTS defaults to one concurrency wave and
# TEST_OSL to 256, because the trace scales with kernel count, not wall-clock.
PROFILE_WRAP=()
if [ "${PROFILE:-0}" = "1" ]; then
    PROFILE_OUT="${PROFILE_OUT:-$RESULT_DIR/rocprof}"
    mkdir -p "$PROFILE_OUT"
    export VLLM_NVTX_SCOPES_FOR_PROFILING=1
    TEST_NUM_PROMPTS="${TEST_NUM_PROMPTS:-${PROFILE_PROMPTS:-$CONC}}"
    export TEST_NUM_PROMPTS
    export TEST_OSL="${TEST_OSL:-256}"
    if command -v rocprofv3 >/dev/null 2>&1; then
        PROFILE_WRAP=(rocprofv3 --kernel-trace --stats
                      -d "$PROFILE_OUT" -o k3 --output-format csv --)
        echo "[profile] rocprofv3 kernel-trace -> $PROFILE_OUT prompts=$TEST_NUM_PROMPTS osl=$TEST_OSL conc=$CONC"
    else
        echo "[profile] rocprofv3 NOT FOUND -- running unprofiled" >&2
    fi
else
    echo "[profile] disabled"
fi
SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"
SERVER_PID=""
LMCACHE_PID=""

# Stop the LMCache MP server and WAIT for it to fully exit.
#
# T239 stranded 58 GB on one GPU and needed a node reboot. server.log's last
# line was:
#   Memory critical error by agent node-0 ... Reason: Memory in use.
# vLLM freed KV buffers that LMCacheMPConnector still had registered for DMA.
# The registrations must be gone BEFORE vLLM tears down, so this is called
# explicitly after the workload -- not left to the EXIT trap, which races
# EngineCore (in T239 the engine had already died a second earlier).
#
# Observed shutdown took >50 s under a 5-request fixed-len load; the old 60 s
# allowance was borderline. Budget 300 s and verify the process is actually
# gone before returning.
lmcache_shutdown() {
    [ -n "${LMCACHE_PID:-}" ] || return 0
    kill -0 "$LMCACHE_PID" 2>/dev/null || { LMCACHE_PID=""; return 0; }
    echo "[lmcache] stopping server (pid $LMCACHE_PID) before vLLM teardown..."
    kill -TERM "$LMCACHE_PID" 2>/dev/null || true
    local waited=0
    while kill -0 "$LMCACHE_PID" 2>/dev/null; do
        [ "$waited" -ge 300 ] && break
        sleep 2; waited=$(( waited + 2 ))
    done
    if kill -0 "$LMCACHE_PID" 2>/dev/null; then
        echo "[lmcache] STILL ALIVE after ${waited}s -- SIGKILL. GPU memory may be stranded; check VRAM before the next run." >&2
        kill -KILL "$LMCACHE_PID" 2>/dev/null || true
        sleep 5
    else
        echo "[lmcache] server exited cleanly after ${waited}s"
    fi
    LMCACHE_PID=""
}

cleanup_agentic_services() {
    local exit_code=$?
    trap - EXIT INT TERM
    set +e
    # LMCache first: its DMA registrations must be released before vLLM frees
    # the buffers they point at.
    lmcache_shutdown
    stop_background_process_tree "$SERVER_PID" "vLLM server" 60
    exit "$exit_code"
}
trap cleanup_agentic_services EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

export PYTHONHASHSEED=42

OFFLOAD_ARGS=()
if agentic_kv_offload_enabled; then
    case "${KV_OFFLOAD_BACKEND:-}" in
      vllm-simple)
        require_agentic_kv_offload_backend "$KV_OFFLOAD_BACKEND"
        CPU_BYTES_PER_RANK=$(( TOTAL_CPU_DRAM_GB * 1000 * 1000 * 1000 / TOTAL_RANKS ))
        export PYTHONHASHSEED=42
        SIMPLE_LAZY_OFFLOAD="${SIMPLE_LAZY_OFFLOAD:-false}"
        OFFLOAD_ARGS=(
            --kv-transfer-config
            "{\"kv_connector\":\"SimpleCPUOffloadConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use_per_rank\":$CPU_BYTES_PER_RANK,\"lazy_offload\":$SIMPLE_LAZY_OFFLOAD}}"
        )
        echo "SimpleCPUOffloadConnector: ${CPU_BYTES_PER_RANK} B/rank x ${TOTAL_RANKS} ranks, lazy_offload=$SIMPLE_LAZY_OFFLOAD"
        ;;
      lmcache)
        require_agentic_kv_offload_backend "$KV_OFFLOAD_BACKEND"
        export PYTHONHASHSEED=42

        # Wiring from the SA reference (read-only): image stays stock, only the
        # LMCache runtime is added. --no-deps so torch/ROCm are untouched.
        LMCACHE_VERSION="${LMCACHE_VERSION:-0.5.5rc3+rocm7.2}"
        LMCACHE_RELEASE="v${LMCACHE_VERSION%%+*}"
        agentic_pip_install --quiet --no-cache-dir --no-deps \
            "sortedcontainers==2.4.0" \
            "opentelemetry-exporter-prometheus==0.61b0" \
            "cupy-rocm-7-0==14.1.1" \
            "lmcache==${LMCACHE_VERSION}" \
            --find-links "https://github.com/LMCache/LMCache/releases/expanded_assets/${LMCACHE_RELEASE}-rocm"

        for _lib in libglog.so.0 libjsoncpp.so.25 libibverbs.so.1 librdmacm.so.1 libnuma.so.1; do
            if ! ldconfig -p | grep -q "$_lib"; then
                apt-get update -qq
                apt-get install -y -qq libgoogle-glog0v5 libjsoncpp25 libibverbs1 librdmacm1 libnuma1
                break
            fi
        done
        python3 -c "import cupy; import opentelemetry.exporter.prometheus; from lmcache.v1.multiprocess.http_server import run_http_server"

        # --chunk-size MUST divide every KV group's tokens_per_block. The hybrid
        # KDA/MLA layout registers 1536 (attention) and 3072 (KDA state), so
        # 12288 divides both. The upstream Kimi-K3 recipe says 768 -- that is the
        # CUDA path and is WRONG here.
        # OPEN RISK: those sizes are quoted at DCP=1. At DCP=8 the per-group
        # geometry changes. Verify from the engine log before trusting 12288.
        LMCACHE_PORT="${LMCACHE_PORT:-6555}"
        LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-8090}"
        LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-12288}"
        # BACK TO 1 (T253). The 1 -> TP change was reasoned from T240's stranded
        # ~52 GB "asymmetry matches a single GPU worker" -- that reasoning was
        # wrong, and the evidence now runs the other way:
        #
        #   T239  --max-gpu-workers 1  ->  PASS, 5/5, 920 tok/s, no fault
        #   T250  --max-gpu-workers 8  ->  FAIL, 0/144 AND 0/893, 100%
        #   SA reference recipe        ->  1
        #
        # Every other lmcache server arg and the connector JSON match the SA
        # reference exactly; this was the only difference. --help calls it
        # "Worker threads for the GPU affinity pool" -- threads, not memory.
        LMCACHE_GPU_WORKERS="${LMCACHE_GPU_WORKERS:-1}"
        LMCACHE_LOG="$RESULT_DIR/lmcache_server.log"

        LMCACHE_CMD=(
            lmcache server
            --host 127.0.0.1 --port "$LMCACHE_PORT"
            --http-host 127.0.0.1 --http-port "$LMCACHE_HTTP_PORT"
            --l1-size-gb "$TOTAL_CPU_DRAM_GB" --l1-init-size-gb 10
            --chunk-size "$LMCACHE_CHUNK_SIZE"
            --separate-object-groups
            --enable-extra-logging --extra-logging-interval 30
            --max-cpu-workers 8 --max-gpu-workers "$LMCACHE_GPU_WORKERS"
            --eviction-policy LRU
            --supported-transfer-mode lmcache_driven
            --shm-name ""
        )
        printf '%q ' "${LMCACHE_CMD[@]}" > "$RESULT_DIR/lmcache_command.txt"; printf '\n' >> "$RESULT_DIR/lmcache_command.txt"
        echo "[lmcache] chunk=$LMCACHE_CHUNK_SIZE l1=${TOTAL_CPU_DRAM_GB}GB gpu_workers=$LMCACHE_GPU_WORKERS port=$LMCACHE_PORT"
        "${LMCACHE_CMD[@]}" > "$LMCACHE_LOG" 2>&1 &
        LMCACHE_PID=$!

        # Our benchmark_lib.sh has no wait_for_ready (that helper lives in a
        # newer SA lib), so the readiness poll is inlined.
        _lmc_ready=0
        for _i in $(seq 1 600); do
            if ! kill -0 "$LMCACHE_PID" 2>/dev/null; then
                echo "[lmcache] server died during startup; tail of $LMCACHE_LOG:" >&2
                tail -40 "$LMCACHE_LOG" >&2 || true
                exit 1
            fi
            if curl -sf "http://127.0.0.1:${LMCACHE_HTTP_PORT}/healthcheck" >/dev/null 2>&1; then
                _lmc_ready=1; break
            fi
            sleep 1
        done
        if [ "$_lmc_ready" != "1" ]; then
            echo "[lmcache] healthcheck did not pass in 600s; tail of $LMCACHE_LOG:" >&2
            tail -40 "$LMCACHE_LOG" >&2 || true
            exit 1
        fi
        echo "[lmcache] server READY on :$LMCACHE_HTTP_PORT after ${_i}s"

        # mq_timeout 6000 -- 100k-330k-token agentic prefixes make single
        # retrieves large; our ISL p50 is ~89k.
        OFFLOAD_ARGS=(
            --kv-transfer-config
            "{\"kv_connector\":\"LMCacheMPConnector\",\"kv_connector_module_path\":\"lmcache.integration.vllm.lmcache_mp_connector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.port\":$LMCACHE_PORT,\"lmcache.mp.mq_timeout\":6000.0}}"
        )
        ;;
      *)
        echo "KV offload requested (KV_OFFLOADING=$KV_OFFLOADING) but backend '${KV_OFFLOAD_BACKEND:-unset}' is not handled here" >&2
        ;;
    esac
fi

KV_CACHE_DTYPE=fp8
EP_ARGS=()
if [ "${EP_SIZE:-1}" -gt 1 ]; then
    EP_ARGS=(--enable-expert-parallel)
    echo "EP: expert parallelism ON (EP_SIZE=$EP_SIZE)"
fi

CP_ARGS=(--attention-backend ROCM_AITER_MLA)
if [ "$DCP_SIZE" -gt 1 ]; then
    CP_ARGS+=(
        --decode-context-parallel-size "$DCP_SIZE"
        --dcp-comm-backend a2a
        --cp-kv-cache-interleave-size 1
    )
    # N9 IMPOSSIBLE ON THIS IMAGE -- these three stay 0. T167 flipped a2a to 1
    # and every worker died during cudagraph capture with
    #   AttributeError: '_OpNamespace' '_C' object has no attribute
    #                   'direct_dcp_a2a_lse_reduce'
    # The direct DCP path needs a compiled C++ op from #51705 that
    # aigmkt/kimi-k3-vllm does not ship. So these were never "force-disabling a
    # fast path" -- they disable an op that is absent. Re-enabling requires a
    # rebuilt image, which is out of bounds here.
    # These five are an AIGMKT WORKAROUND, not a tuning choice. On aigmkt the
    # direct-DCP op does not exist (T167: AttributeError '_OpNamespace' '_C'
    # has no attribute 'direct_dcp_a2a_lse_reduce'), so we pinned them off.
    #
    # The K3 overlay ADDS that op. T184 forced them off anyway on
    # nightly+overlay and warmup died in a hung _ALLGATHER (NCCL collective
    # timeout, rank 1, VllmWorker-3) -- i.e. we shoved DCP onto a fallback
    # gather path that SA never exercises. SA's launcher sets NONE of these
    # five. So when the overlay lands, set nothing and match SA.
    if [ "${K3_OVERLAY_APPLIED:-0}" = "1" ]; then
        echo "[dcp-direct] overlay applied -- leaving VLLM_USE_DIRECT_DCP_*/ALLOW_DCP_FULL_CUDAGRAPH/Q_REPLICATE at engine defaults (matches SA)"
    else
        export VLLM_USE_DIRECT_DCP_A2A=0
        export VLLM_USE_DIRECT_DCP_Q_GATHER=0
        export VLLM_USE_DIRECT_DCP_KV_GATHER=0
        echo "[dcp-direct] a2a=0 q_gather=0 kv_gather=0 (op absent in this image)"
        export VLLM_ALLOW_DCP_FULL_CUDAGRAPH=1
        export VLLM_DCP_Q_REPLICATE=1
    fi
    echo "[dcp] ENABLED size=$DCP_SIZE backend=a2a interleave=1"
elif [ "${DCP_COMM_ARGS_AT_1:-0}" = "1" ]; then
    CP_ARGS+=(--dcp-comm-backend a2a --cp-kv-cache-interleave-size 1)
    echo "[dcp] size=1, comm args RETAINED (a2a, interleave=1), no DCP env"
else
    echo "[dcp] DISABLED -- no DCP args, no DCP env"
fi
export VLLM_ROCM_USE_AITER_MLA=1
export AITER_DISABLE_FMHA_OPUS=1

SPEC_ENABLE="${SPEC_DECODING:-}"
case "${RESULT_FILENAME:-}" in *_spec-mtp_*) SPEC_ENABLE=mtp;; esac
case "$CONC" in
    1|2|4)   SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-8}" ;;
    *)       SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-0}" ;;
esac
if [ "$SPEC_NUM_TOKENS" -eq 0 ]; then SPEC_ENABLE=""; fi
SPEC_ARGS=()
if [ "$SPEC_ENABLE" = "mtp" ]; then
    case "$SPEC_NUM_TOKENS" in
        1) SYNTHETIC_ACCEPT_LEN=1.85 ;;
        2) SYNTHETIC_ACCEPT_LEN=2.51 ;;
        3) SYNTHETIC_ACCEPT_LEN=3.00 ;;
        4) SYNTHETIC_ACCEPT_LEN=3.36 ;;
        5) SYNTHETIC_ACCEPT_LEN=3.62 ;;
        6) SYNTHETIC_ACCEPT_LEN=3.75 ;;
        7) SYNTHETIC_ACCEPT_LEN=3.84 ;;
        8) SYNTHETIC_ACCEPT_LEN=4.00 ;;
        *) echo "[spec] no golden AL wired for num_speculative_tokens=$SPEC_NUM_TOKENS; take it from golden_al_distribution/kimik3_dspark_probabilistic_sample_method_block_rejection_sample_method.yaml and add the case" >&2; exit 1 ;;
    esac
    DRAFT_KV_DTYPE="${DRAFT_KV_DTYPE:-fp8}"
    SPEC_ARGS=(
        --speculative-config
        "{\"model\":\"Inferact/Kimi-K3-DSpark\",\"num_speculative_tokens\":$SPEC_NUM_TOKENS,\"method\":\"dspark\",\"attention_backend\":\"TRITON_MLA\",\"kv_cache_dtype\":\"$DRAFT_KV_DTYPE\",\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\": \"synthetic\", \"synthetic_acceptance_length\": $SYNTHETIC_ACCEPT_LEN}"
    )
    echo "MTP: speculative decoding ON (k=$SPEC_NUM_TOKENS, synthetic accept=$SYNTHETIC_ACCEPT_LEN, draft kv=$DRAFT_KV_DTYPE)"
fi

# N4 SETTLED: 8192 is the optimum, do not move it. T164 measured 4096 at 7,528
# against T163's 8,127 -- -7.4%, far worse than 16384's -2.5%. The curve has a
# clear peak at 8192 and both sides are downhill.
CHUNKED_PREFILL_ARGS=(--max-num-batched-tokens "${MAX_BATCHED_TOKENS:-16384}")
echo "[chunk] max_num_batched_tokens=${MAX_BATCHED_TOKENS:-16384} conc=$CONC"
# N2 SETTLED NEGATIVE, do not re-enable. T162 C52 measured 7,686 against T161's
# 7,824 on the identical config -- -1.8%. Smaller than the -9.2% on the old
# engine, but still the wrong sign after 175 commits. The host prep the profile
# blames for ~150 s of idle is evidently not what async overlaps here.
if [ "${ASYNC_SCHED:-0}" = "1" ]; then
    ASYNC_SCHED_ARGS=(--async-scheduling)
else
    ASYNC_SCHED_ARGS=(--no-async-scheduling)
fi
MLA_PREFILL_ARGS=(--attention-config "{\"mla_prefill_backend\":\"ROCM_AITER_FA\"}")

LOAD_FORMAT="${LOAD_FORMAT:-fastsafetensors}"
echo "[load] load_format=$LOAD_FORMAT conc=$CONC"

# N5 SETTLED NEGATIVE: mns 96 KILLS THE ENGINE. T165 C52 died mid-replay with
# EngineDeadError from engine_core_sentinel -> mq.dequeue timeout, the same
# trace as the C1 crash. Not memory: the dump shows kv_cache_usage=0.28 and
# num_running_reqs=45. A 96-row batch simply makes the step longer than the
# executor's RPC dequeue timeout, and the sentinel promotes that to fatal.
# mns 80 completed twice on this exact image (T163, T164). Do not raise it
# again without first raising that timeout.
if [ -z "${MAX_NUM_SEQS:-}" ]; then
    if [ "$DCP_SIZE" -gt 1 ]; then
        # Flat 80. Tracking conc was tried (T219, mns 20 at C16) and caused total
        # starvation: the agentic replay runs MORE lanes than CONC (Running: 79
        # at conc 72), so mns must exceed conc generously. T208's match-the-batch
        # rule is C1/MTP-specific and does not generalise here.
        # T230: 96 at C76. T229 (C76, mns 80) fell 2.7% below the C72 band with
        # only 4 slots of slack -- the C80 starvation regime (T196/T197). 96 was
        # neutral at C72 (T198) so it cannot flatter the baseline.
        MAX_NUM_SEQS=96
    else
        MAX_NUM_SEQS=$(( CONC + CONC / 4 ))
        if [ "$MAX_NUM_SEQS" -lt 1 ]; then MAX_NUM_SEQS=1; fi
        if [ "$MAX_NUM_SEQS" -gt 80 ]; then MAX_NUM_SEQS=80; fi
    fi
fi
echo "[mns] max_num_seqs=$MAX_NUM_SEQS conc=$CONC offload=${KV_OFFLOADING:-none}"
if [ "$MAX_NUM_SEQS" -ge 80 ] && ! agentic_kv_offload_enabled; then
    echo "[mns] note: mns=$MAX_NUM_SEQS with KV_OFFLOADING=${KV_OFFLOADING:-none}. Proven on mi355x-amds_01 (8204 tok/s/GPU); OOMs on mi355x-amd_b23_07. Export MAX_NUM_SEQS=65 if HSA_STATUS_ERROR_OUT_OF_RESOURCES."
fi

SPEC_ROWS=1
if [ "${#SPEC_ARGS[@]}" -gt 0 ]; then SPEC_ROWS=$(( SPEC_NUM_TOKENS + 1 )); fi

# ONE RULE: the captured ladder ALWAYS covers mns x SPEC_ROWS. No clamp below it.
#
# Why the old LADDER_MAX clamp is gone. A batch of N sequences with speculative
# decoding carries N x (k+1) rows. If the ladder stops short of that, the larger
# batches fall off the cudagraph path into eager execution -- a different
# allocation path. That mismatch is the single config difference between SA's
# passing C52 (capture 80, mns 80) and their HSA failure on the SAME node and
# image (capture 65, mns 80). We were carrying the same latent mismatch at C1:
# mns 8 x 9 rows = 72 needed, ladder clamped to 32.
#
# At C1 the live batch is only 1 x 9 = 9 rows, so the extra graphs are mostly
# unused -- the cost is capture time and a little memory, and it buys removal of
# a whole failure class. Cheap insurance.
MAX_CUDAGRAPH_CAPTURE_SIZE=$(( MAX_NUM_SEQS * SPEC_ROWS ))
CUDAGRAPH_CAPTURE_SIZES=$(seq -s, 1 "$MAX_CUDAGRAPH_CAPTURE_SIZE")
echo "graphs: dense ladder 1..$MAX_CUDAGRAPH_CAPTURE_SIZE (mns=$MAX_NUM_SEQS x $SPEC_ROWS rows), DCP=$DCP_SIZE"
CUDAGRAPH_MODE=FULL_AND_PIECEWISE
COMPILATION_CONFIG_ARGS=(--compilation-config "{\"mode\":3,\"cudagraph_mode\":\"$CUDAGRAPH_MODE\",\"max_cudagraph_capture_size\":$MAX_CUDAGRAPH_CAPTURE_SIZE,\"custom_ops\":[\"+fused_rms_norm_gated\"],\"cudagraph_capture_sizes\":[$CUDAGRAPH_CAPTURE_SIZES]}")

# N7 SETTLED NEGATIVE: gmu > 0.90 hangs this node. T166 at 0.92 got 0/103 --
# the server came up and KV grew 59.8 -> 65.6 GiB (+9.7%), then it hung in
# warmup and never served a request. T157 at 0.95 hung the same way (0/57).
# Two points above 0.90 both hang; 0.90 works. Memory headroom is NOT the
# free capacity it looks like. Do not raise this again.
# C<=4 has no batch to hold, so KV headroom should not bind -- but 0.92 was
# assumed for C1 early on and never actually measured. Last of the four tail
# hypotheses.
# REVERTED to 0.90 for T243. T242 ran 0.85 on the T236 image and collapsed in
# warmup exactly like T241 (which was at 0.90) -- so gmu does NOT explain both.
# 0.90 is the value behind every good number in the ledger; hold it fixed while
# we establish whether the node still serves traffic at all.
# Do not raise: 0.92 and 0.95 both hang (T211, T157). 0.88 is the tested ceiling.
GPU_MEM_UTIL=0.9   # 0.92 measured CATASTROPHIC at C1 (T211): mean 9.06 -> 21.61 ms
# T253: LMCache-ONLY overrides. Scoped deliberately -- the C72 baseline
# (11,027 tok/s/GPU, n=2) was measured at gmu 0.90 / mns 96 and must not move.
# SA run 33631260867 is the only LMCache configuration known to serve:
#   conc 48, mns 80, gmu 0.88, --max-gpu-workers 1  ->  8,997 tok/s/GPU
# Ours has never served at conc 72 / mns 96 / gmu 0.90. Reproduce the reference,
# then walk ONE knob at a time back toward C72.
# Gated on CONC=48, not on the backend: T254 is the LMCache CONTROL and must
# differ from T253 in the backend ONLY. Tying these to the lmcache branch would
# have made the control a three-variable change (backend + gmu + mns) and left
# the -37% inseparable from the concurrency drop. C72 defaults untouched.
# T255: gate on the LMCache backend, not on conc. T255 must differ from T253 in
# CONC ONLY (48 -> 72); gating on conc would have reverted gmu/mns and made it a
# three-variable change. Non-LMCache C72 runs keep 0.90/96 and are unaffected.
if [ "${KV_OFFLOAD_BACKEND:-}" = "lmcache" ]; then
    GPU_MEM_UTIL="${GPU_MEM_UTIL_LMCACHE:-0.88}"
    MAX_NUM_SEQS=80
    echo "[sa-match] lmcache arm: gmu=$GPU_MEM_UTIL mns=$MAX_NUM_SEQS backend=${KV_OFFLOAD_BACKEND:-vllm-simple}"
fi
# T234: bare images only -- see the gmu-override block above. Patched images
# keep 0.9 so every number in the ledger stays comparable.
if [ -n "${GPU_MEM_UTIL_OVERRIDE:-}" ]; then GPU_MEM_UTIL="$GPU_MEM_UTIL_OVERRIDE"; fi
echo "[gmu] gpu_memory_utilization=$GPU_MEM_UTIL conc=$CONC"

VLLM_CMD=(
    vllm serve "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    --moe-backend auto
    --tensor-parallel-size "$TP"
    --load-format "$LOAD_FORMAT"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --language-model-only
    --max-num-seqs "$MAX_NUM_SEQS"
    --enable-auto-tool-choice
    --tool-call-parser kimi_k3
    --reasoning-parser kimi_k3
    --max-model-len 1048576
    --enable-prefix-caching
    --enable-prompt-tokens-details
    --kv-cache-dtype "$KV_CACHE_DTYPE"
    "${CHUNKED_PREFILL_ARGS[@]}"
    "${OFFLOAD_ARGS[@]}"
    "${CP_ARGS[@]}"
    "${EP_ARGS[@]}"
    "${SPEC_ARGS[@]}"
    "${ASYNC_SCHED_ARGS[@]}"
    "${MLA_PREFILL_ARGS[@]}"
    "${COMPILATION_CONFIG_ARGS[@]}"
)

for _a in CP_ARGS SPEC_ARGS CHUNKED_PREFILL_ARGS ASYNC_SCHED_ARGS MLA_PREFILL_ARGS OFFLOAD_ARGS COMPILATION_CONFIG_ARGS; do
    grep -q "\${$_a\[@\]}" "$0" || echo "[orphan-check] WARNING: $_a is built but never passed to VLLM_CMD" >&2
done

printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"

${PROFILE_WRAP[@]+"${PROFILE_WRAP[@]}"} "${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

python3 - <<'CCDPY' > /tmp/ccdmap.txt 2>/dev/null || true
import subprocess, re, os, glob
def expand(s):
    v=[]
    for part in s.split(','):
        if '-' in part:
            a,b=part.split('-'); v+=list(range(int(a),int(b)+1))
        else: v.append(int(part))
    return v
def l3_domains():
    seen,out=set(),[]
    for c in sorted(int(re.search(r'cpu(\d+)$',x).group(1)) for x in glob.glob('/sys/devices/system/cpu/cpu[0-9]*')):
        f=f'/sys/devices/system/cpu/cpu{c}/cache/index3/shared_cpu_list'
        if not os.path.exists(f): continue
        d=open(f).read().strip()
        if d not in seen: seen.add(d); out.append(d)
    return out
def node_of(cpus):
    for n in glob.glob('/sys/devices/system/node/node[0-9]*'):
        nid=int(re.search(r'node(\d+)$',n).group(1))
        if cpus[0] in expand(open(f'{n}/cpulist').read().strip()): return nid
    return -1
topo=""
try: topo=subprocess.run(["rocm-smi","--showtoponuma"],capture_output=True,text=True).stdout
except Exception: pass
gpu_node={int(m.group(1)):int(m.group(2)) for m in re.finditer(r"GPU\[(\d+)\].*?Numa Node:\s*(\d+)",topo)}
if not gpu_node: raise SystemExit
by={}
for d in l3_domains(): by.setdefault(node_of(expand(d)),[]).append(d)
for n in by: by[n].sort(key=lambda d: expand(d)[0])
for n in sorted(by):
    for i,g in enumerate(sorted(k for k,v in gpu_node.items() if v==n)):
        if i < len(by[n]): print(f"{g} {by[n][i]}")
CCDPY

# T249: ON. Never tested on a numa_balancing=0 node, and it targets exactly the
# host-side CPU/NUMA locality that turned out to matter this session: 2 NUMA
# nodes, GPUs split 4/4, remote distance 32, ~1.8 TB of host memory pinned for
# GPU DMA. Baseline is T247 = 11,006 on this same image; one variable.
# Pins AFTER wait_for_server_ready -- pinning before load cost 2008 s in T160.
PIN_CCD="${PIN_CCD:-0}"
pin_workers_to_ccd() {
    [ "$PIN_CCD" = "1" ] || return 0
    [ -s /tmp/ccdmap.txt ] || return 0
    local pinned=0
    while read -r _g _cpus; do
        for _p in $(pgrep -f "VLLM::Worker_TP${_g}([^0-9]|$)" 2>/dev/null); do
            for _t in /proc/$_p/task/*; do
                taskset -pc "$_cpus" "${_t##*/}" >/dev/null 2>&1 && pinned=$((pinned+1)) || true
            done
        done
    done < /tmp/ccdmap.txt
    echo "[pin-ccd] pinned $pinned threads"
}

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# Pin AFTER the server is ready, never before. T160 measured 2008.62 s to load
# weights against 576-681 s unpinned: pinning during the load confines ~190
# loader threads per worker to one CCD's 8 physical cores. Weight load and
# cudagraph capture are one-time and want every core; only steady-state serving
# wants L3 locality. One shot, no background loop.
pin_workers_to_ccd || true

# TEST=1 -> fixed-length serving benchmark instead of the agentic replay.
#
# Why this gate exists: a fixed-len cell is ~10 min against ~1h12m for the
# agentic replay, and it exercises the same server on the same config. Nineteen
# consecutive C1 agentic runs were spent discovering the engine had died -- at
# roughly an hour each. Fixed-len answers "does this engine survive traffic at
# all" for a tenth of the cost.
#
# Intended use: run TEST=1 for C1 and C52 first. If the engine is clean -- no
# EngineDeadError, no memory-access fault, error rate under the threshold --
# only then spend an hour on the agentic replay. If it is not clean, the
# agentic run cannot produce a number and should not be dispatched.
#
# There is no kimik3 script under benchmarks/single_node/fixed_seq_len/, and the
# runner resolves single-node non-disagg to that directory, so a `fixed-seq-len`
# yaml scenario would fail to launch. Branching inside this launcher keeps the
# whole thing in the one file and needs no new script.
if [ "${EVAL_ONLY:-false}" = "true" ]; then
    run_eval --port "$PORT"
# Default flipped to 1: the runner invokes this script directly and there is no
# env passthrough from the yaml, so TEST cannot be set per-dispatch from the
# workflow. Fixed-len is the default health probe now; set TEST=0 in the script
# to go back to the agentic replay once an engine is proven clean.
# T246: back to 0 (agentic replay). The fixed-len gate has served its purpose --
# T245 proved the engine healthy (893/893, 914k tokens) once NUMA was off and
# the probe was the right one. Next question is whether the agentic C72 number
# reproduces post-reboot, and that needs the replay, not fixed-len.
# Set TEST=1 again for a ~10 min health check before spending an hour, and when
# you do, check the [test-mode] line in the log -- `func` is NOT a light canary.
elif [ "${TEST:-0}" = "1" ]; then
    # ISL/OSL/ratio defaults, and why they are what they are.
    #
    # range_ratio in this repo is NOT +/-ratio. benchmark_serving.py:248 does
    #     lower = int(seq_len * range_ratio); upper = seq_len
    #     seq_lens = randint(lower, upper+1)
    # so lengths are uniform on [isl*ratio, isl]. ratio=1.0 is FIXED length;
    # ratio=0.0 is uniform 0..isl. Higher ratio = tighter, not wider.
    #
    # Measured agentic distribution, from the T170 C72 aiperf tables:
    #   prompt tokens     min 354  p50 80,155  mean 111,750  p90 214,305  max 735,739
    #   completion tokens min 11   p50 238     mean 369      p90 874      max 2,507
    #   prompt cache read 93.32% of prompt tokens
    #
    # Two honest limits on "just derive it from the agentic lengths":
    #  1. That prompt distribution is heavily right-skewed (mean >> median).
    #     A uniform cannot reproduce that shape -- matching the mean forces a
    #     span far wider than the actual IQR, matching the IQR loses the tail.
    #  2. The bigger mismatch is not length at all: agentic serves 93% of its
    #     prompt tokens from cache, so a fixed-len run at the same ISL does
    #     several times the prefill work. Length-matching does not fix that.
    #
    # Defaults now target the agentic band, not a synthetic 8k probe. T180
    # showed the 8k fixed-len probe is too easy to be a functionality test: C1
    # served 10/10 at 8k while the agentic replay has failed 19 straight times.
    # A probe that passes when the real workload fails tells us nothing.
    #
    #   ISL 214000 with ratio 0.37 -> uniform[79,180, 214,000] = agentic p50..p90
    #   OSL 874    with ratio 0.37 -> uniform[323, 874]        ~ agentic p50..p90
    #
    # One ratio drives both input and output (confirmed in T180: isl 8192 /
    # ratio 0.8 gave 6,830-7,936 and osl 1024 gave 858-1,012), so the output
    # band is a compromise -- its mean lands ~600 against the agentic 369.
    #
    # NOT overridden from RANDOM_RANGE_RATIO any more. The workflow exports
    # RANDOM_RANGE_RATIO=0.8, which silently won in T180 and gave ratio 0.8
    # where the comment claimed 1.0. Default is now literal.
    #
    # Still does NOT reproduce the 93.3% prefix-cache hit rate, so this does
    # several times the prefill work per token that agentic does. It is a
    # functionality test, not a throughput comparison -- do not read tok/s from
    # it as comparable to the agentic ledger.
    #
    # Iterations cut to CONC*2 (floor 4): at 214k tokens per prompt, CONC*10
    # would be ~11M input tokens at C52 before decode.
    # TEST_MODE=func (default) -> agentic-band functionality test, few iterations.
    # TEST_MODE=perf           -> 8k/1k fixed length, sized to ~15 min.
    # TEST_MODE=both (default): run the functionality pass AND the perf pass
    # against the SAME server, so we pay the ~2.5 min weight load once instead of
    # twice. func runs first and is short; if the engine is broken we find out in
    # minutes. perf writes the official result file the CI wrapper looks for.
    # T245: default flipped func -> perf. `func` generates agentic-band prompts
    # (mean 152k tokens, max 214k) at full CONC -- one of the heaviest workloads
    # we have, NOT a cheap canary. T243/T244 both failed it and that told us
    # little, because no baseline exists for it. `perf` is fixed 8192/1024,
    # ratio 1.0, run-to-run comparable, and comparable to the prior fixed-len
    # numbers (T180: 10/10, TPOT 7.41 ms). Use perf to answer "does the engine
    # serve at all"; switch back to func only once perf is clean.
    # T250: both = Function pass then Fixed Perf pass on one server.
    TEST_MODE="${TEST_MODE:-both}"
    if [ "$TEST_MODE" = "both" ]; then
        echo "[test-mode] both: functionality pass then perf pass on one server"
        FUNC_ISL="${FUNC_ISL:-214000}"; FUNC_OSL="${FUNC_OSL:-874}"
        FUNC_RATIO="${FUNC_RATIO:-0.37}"; FUNC_PROMPTS="${FUNC_PROMPTS:-$(( CONC * 2 ))}"
        if [ "$FUNC_PROMPTS" -lt 4 ]; then FUNC_PROMPTS=4; fi
        echo "[test] FUNC pass: isl=$FUNC_ISL osl=$FUNC_OSL ratio=$FUNC_RATIO prompts=$FUNC_PROMPTS conc=$CONC"
        run_benchmark_serving \
            --model "$MODEL" --port "$PORT" --backend vllm \
            --input-len "$FUNC_ISL" --output-len "$FUNC_OSL" \
            --random-range-ratio "$FUNC_RATIO" \
            --num-prompts "$FUNC_PROMPTS" --max-concurrency "$CONC" \
            --result-filename "${RESULT_FILENAME}_func" \
            --result-dir "${INFMAX_CONTAINER_WORKSPACE:-/workspace}" \
            --trust-remote-code || echo "[test] FUNC pass FAILED (continuing to perf pass)"
        echo "[test] FUNC pass done"
        TEST_MODE=perf
    fi
    if [ "$TEST_MODE" = "perf" ]; then
        # Fixed 8k/1k. ratio 1.0 = exactly fixed, so run-to-run comparable.
        TEST_ISL="${TEST_ISL:-8192}"
        TEST_OSL="${TEST_OSL:-1024}"
        TEST_RANGE_RATIO="${TEST_RANGE_RATIO:-1.0}"
        # benchmark_serving.py has NO duration flag -- it takes --num-prompts and
        # reports the duration it happened to take. So "15 minutes" has to be
        # converted to a prompt count, and that needs a per-request latency.
        #
        # TEST_EST_REQ_SECONDS is that estimate: seconds per request at this
        # concurrency. C1 is measured -- T180 did 10 prompts in 80.72 s at
        # isl 8192 / osl 1024, i.e. 8.07 s/req. Above C4 there is NO measurement
        # at 8k/1k, so 13 s is a guess: decode slows per-user as the batch fills.
        # The run will therefore NOT be exactly 15 min. Read the reported
        # "Benchmark duration (s)" and set TEST_EST_REQ_SECONDS from it to make
        # the next one land.
        TEST_TARGET_SECONDS="${TEST_TARGET_SECONDS:-900}"
        if [ -z "${TEST_EST_REQ_SECONDS:-}" ]; then
            if [ "$CONC" -le 4 ]; then TEST_EST_REQ_SECONDS=8.07; else TEST_EST_REQ_SECONDS=72.57; fi
        fi
        TEST_NUM_PROMPTS="${TEST_NUM_PROMPTS:-$(awk -v t="$TEST_TARGET_SECONDS" -v c="$CONC" \
            -v l="$TEST_EST_REQ_SECONDS" 'BEGIN{n=int(t*c/l+0.5); if(n<1)n=1; print n}')}"
        echo "[test-mode] perf: target=${TEST_TARGET_SECONDS}s est=${TEST_EST_REQ_SECONDS}s/req -> prompts=$TEST_NUM_PROMPTS (duration is an ESTIMATE, not enforced)"
    else
        TEST_ISL="${TEST_ISL:-214000}"
        TEST_OSL="${TEST_OSL:-874}"
        TEST_RANGE_RATIO="${TEST_RANGE_RATIO:-0.37}"
        TEST_NUM_PROMPTS="${TEST_NUM_PROMPTS:-$(( CONC * 2 ))}"
        if [ "$TEST_NUM_PROMPTS" -lt 4 ]; then TEST_NUM_PROMPTS=4; fi
        echo "[test-mode] func: agentic-band lengths, $TEST_NUM_PROMPTS prompts"
    fi
    echo "[test] fixed-len serving: isl=$TEST_ISL osl=$TEST_OSL ratio=$TEST_RANGE_RATIO prompts=$TEST_NUM_PROMPTS conc=$CONC (agentic replay SKIPPED)"
    run_benchmark_serving \
        --model "$MODEL" \
        --port "$PORT" \
        --backend vllm \
        --input-len "$TEST_ISL" \
        --output-len "$TEST_OSL" \
        --random-range-ratio "$TEST_RANGE_RATIO" \
        --num-prompts "$TEST_NUM_PROMPTS" \
        --max-concurrency "$CONC" \
        --result-filename "$RESULT_FILENAME" \
        --result-dir "${INFMAX_CONTAINER_WORKSPACE:-/workspace}" \
        --trust-remote-code
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi

# Release LMCache's DMA registrations while the vLLM server is still up. Doing
# this here rather than in the EXIT trap is the T239 fix -- see lmcache_shutdown.
lmcache_shutdown
