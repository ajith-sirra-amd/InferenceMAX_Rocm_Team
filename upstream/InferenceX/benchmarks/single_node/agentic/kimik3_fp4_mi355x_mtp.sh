#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for Kimi-K3 MXFP4 on MI355X / MI350X (gfx950)
# using vLLM.
#
# The server command is the AMD reference `vllm serve` for this model, i.e. the
# upstream vLLM recipe's amd block (vllm-project/recipes,
# https://recipes.vllm.ai/moonshotai/Kimi-K3) as run in practice:
#
#   --trust-remote-code --moe-backend auto --tensor-parallel-size 8
#   --load-format auto --gpu-memory-utilization 0.95 --mm-encoder-tp-mode data
#   --max-num-seqs 128 --max-num-batched-tokens 4096 --enable-auto-tool-choice
#   --tool-call-parser kimi_k3 --reasoning-parser kimi_k3
#
# with env VLLM_ROCM_USE_AITER=1 SAFETENSORS_FAST_GPU=1 AITER_SITUV2_A8W4=1
# AITER_BF16_FP8_MOE_BOUND=0 VLLM_USE_BREAKABLE_CUDAGRAPH=0.
#
# K3 is a 2.8T-parameter natively-multimodal MoE (896 routed experts, 16/token
# plus shared) on Kimi Delta Attention, gated MLA and Attention Residuals, with
# a 1M-token native context.
#
# TP=8 ONLY. The MXFP4 checkpoint is 1.561 TB decimal (1.420 TiB, 96
# safetensors), ~195 GB/GPU across 8 GPUs of the 288 GB part; TP=4 would need
# ~390 GB/GPU and cannot load. Upstream strategy_min_gpus agrees (single_node_tp
# and multi_node_tep both 8, DEP 16+), which is why there is no DP-attention arm.
#
# Required env vars:
#   MODEL, TP, CONC, KV_OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR, DURATION,
#   EP_SIZE
#
# Perf-search knobs. Each defaults to the reference command's value, so an
# otherwise-unset run reproduces the reference exactly:
#   GPU_MEM_UTIL             0.95   (reference)
#   MAX_NUM_BATCHED_TOKENS   8192   (default)
#   AITER_A8W4               1      (reference; 0 = aiter a16w4 MoE path)
#   LANGUAGE_MODEL_ONLY      true   
#   KV_CACHE_DTYPE           fp8    (default for every arm; =auto for a bf16 A/B)
#   KV_BLOCK_SIZE            unset  (unset -> vLLM sizes the page; 128 under fp8)
#   MAX_MODEL_LEN            1M     
#   SPEC_DECODE              true   (this is the _mtp DSpark recipe; =false for a no-spec A/B)
#   SPEC_NUM_TOKENS          2      (DSpark draft length; validated by the _mtp config)

source "$(dirname "$0")/../../benchmark_lib.sh"

wait_for_amd_gpu_clean

# ACCURACY RUN. Set to "false" to go back to the throughput/agentic-replay arm.
# Note EVAL_ONLY=true also flips the speculative config from
# rejection_sample_method "synthetic" to "block" (see the SPEC_ARGS branch
# below), i.e. draft tokens are actually verified against the target, so this
# is the only arm whose generated text is valid. It therefore doubles as the
# correctness check for the triton_mla cudagraph patch.
# Env-overridable so the accuracy arm can be selected per-run (EVAL_ONLY=true)
# without editing this file -- the DCP/LSE work needs to flip between the
# throughput and correctness arms repeatedly.
# T23: CORRECTNESS GATE on the best DCP config (T18). Never passed on the DCP
# path -- T6 timed out because decode ran at 3.5-6.8 tok/s with no KV offload.
# With offload T18 decodes ~4x faster, so 1319 GSM8K questions should now fit.
# Baseline to match: 0.9651. Flip to false to return to the throughput arm.
PROFILE_DECODE="${PROFILE_DECODE:-1}"   # T35: decode-only torch trace to localise the 122 ms
export PROFILE_DECODE
EVAL_ONLY="${EVAL_ONLY:-false}"
export EVAL_FRAMEWORK="lm-eval"

# Fast iteration mode. benchmark_lib.sh's run_agentic_replay honours
# AIPERF_EXPERIMENTAL_FAST=1 by advancing each trajectory lane only once
# (warmup_requests_per_lane 10 -> 1) and capping profiling at 1200 s.
# Trial 4 (32043813560) showed full-fidelity warmup is 707 requests and was
# only 189 done after 54 min -- ~4 h per trial, which allows two experiments in
# a working day. Fast mode brings that to ~40 min. Numbers from fast runs are
# directionally valid for tok/s but come from a 20-minute profile, so the
# headline result must be reproduced with AIPERF_EXPERIMENTAL_FAST=0.
# T12: keep the SHORT WARMUP but restore the FULL PROFILE. FAST=1 does both
# (warmup 10->1 per lane AND duration->1200 s), and the 1200 s window is too
# short for this trace: TTFT alone averages ~163 s and e2e ~400 s, so few
# requests complete inside the profile and throughput is understated. Setting
# the warmup knob directly gives ~20 min of warmup plus the matrix duration.
export AIPERF_EXPERIMENTAL_FAST="${AIPERF_EXPERIMENTAL_FAST:-0}"
export AIPERF_WARMUP_REQUESTS_PER_LANE="${AIPERF_WARMUP_REQUESTS_PER_LANE:-1}"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

if [ "$TP" -ne 8 ]; then
    echo "Error: Kimi-K3 MXFP4 is a 1.56 TB checkpoint and only fits at TP=8 on" >&2
    echo "       288 GB gfx950 parts (~195 GB/GPU). Got TP=$TP." >&2
    exit 1
fi

# ROCR/HIP visibility for vLLM 0.14+
if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

# `hf download` creates the target dir if missing and is itself idempotent. The
# 1.56 TB checkpoint is normally pre-staged, so these calls are a no-op there.
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

# ---- Resolve traces and install deps ----------------------------------------
resolve_trace_source
install_agentic_deps

# ---- In-container patches ----------------------------------------------------
# Three fixes, all confined to this container's site-packages, all idempotent
# and all self-disabling once the image ships them:
#   [1] aiter pybind11 internals mismatch  -> unblocks ROCM_AITER_FA prefill
#   [2] TritonMLA cudagraph support        -> FULL cudagraphs for DSpark (5.52x TPOT)
#   [3] KV block-pool negative-count clamp -> stops the mid-run engine crash
# Set SKIP_KIMI_PATCHES=1 to run stock.
# Only meaningful on the stock ROCm nightly. The unified image already carries
# the PR2585 bundle and a different vLLM, so the anchors would not match there.
if [ ! -f /opt/aiter-local/aiter/configs/merged_bf16_tuned_gemm.csv ]; then
    bash "$(dirname "$0")/apply_kimi_k3_patches.sh" || true
fi

# ---- Reference env block ----------------------------------------------------
# Keep ALL of these. Commenting them out does not avoid the AITER FMHA crash:
# that crash is gated on VLLM_ROCM_USE_AITER alone (AiterFlashAttnPrefillBackend
# .is_available() consults only rocm_aiter_ops.is_enabled()), so disabling the
# others just loses the MoE kernels while keeping the failure.
#
# These were commented out when we moved to the unified image, which exports its
# own equivalents from the block further down. But on the STOCK NIGHTLY nothing
# sets them, and run 32009028600 showed exactly what the comment above warns
# about -- AITER was off end to end:
#     Using 'EMULATION' Mxfp4 MoE backend    <- MXFP4 dequantized to generic GEMMs
#     Using TRITON_MLA backend               <- not ROCM_AITER_MLA
#     Using FLASH_ATTN MLA prefill backend   <- not ROCM_AITER_FA
# EMULATION on a MoE model is catastrophic and is why warmup crawled at 9/796.
# So: export them whenever the unified image is NOT present.
if [ ! -f /opt/aiter-local/aiter/configs/merged_bf16_tuned_gemm.csv ]; then
    export VLLM_ROCM_AITER_MLA_ASM_PADDING=asm
    export VLLM_ROCM_USE_AITER=1
    export SAFETENSORS_FAST_GPU=1
    export VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1
    export AITER_BF16_FP8_MOE_BOUND=0
    # REQUIRED on ROCm per the upstream recipe: the build auto-enables this to 1.
    export VLLM_USE_BREAKABLE_CUDAGRAPH=0
    # --- parity with the unified image's own env block (validate_live_graph_capture.sh)
    # This path was setting only 6 of its 13 vars. Two look load-bearing:
    #  * HSA_NO_SCRATCH_RECLAIM=1 disables ROCm scratch-memory reclaim, a known
    #    source of HSA_STATUS_ERROR_EXCEPTION -- which is exactly how run
    #    32012699807 died (code 0x1016 on all 8 queues).
    #  * AITER_SITUV2_A8W4 + VLLM_ROCM_USE_AITER_MOE: we set only the
    #    VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4 spelling, and that run selected
    #    'AITER_MXFP4_BF16' MoE (BF16 activations) rather than the A8W4 path.
    # VLLM_K3_KDA_SAFE_STAGES matters because Kimi-K3's hybrid KDA layers are
    # where DCP sharding is least exercised.
    # Not copied: AITER_CONFIG_GEMM_BF16, which points at the unified image's
    # merged tuned-GEMM table and has no equivalent file here.
    export GPU_ARCHS=gfx950
    export VLLM_ROCM_USE_AITER_MOE=1
    export AITER_SITUV2_A8W4=1
    export HSA_NO_SCRATCH_RECLAIM=1
    export VLLM_K3_KDA_SAFE_STAGES=1
    export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
    echo "AITER env: enabled (stock nightly path, unified-parity set)"
fi

# Workaround for MEC FW <177 RCCL memory reclaim issue (shared with the other
# gfx950 recipes in this tree).
mec_version=$(rocm-smi --showfw 2>/dev/null | grep MEC | head -n 1 | awk '{print $NF}')
if [[ "$mec_version" == "" || ${mec_version:-0} -lt 177 ]]; then
    export HSA_NO_SCRATCH_RECLAIM=1
fi

# 2.8T of weights off a shared/NFS mount takes far longer than the default.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}"

# Long agentic turns against a 1M context: keep the client from timing out
# mid-request while the server is prefill-bound.
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000

# ---- Server config ----------------------------------------------------------
SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

SERVER_PID=""

cleanup_agentic_services() {
    local exit_code=$?
    trap - EXIT INT TERM
    set +e
    stop_background_process_tree "$SERVER_PID" "vLLM server" 60
    exit "$exit_code"
}
trap cleanup_agentic_services EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ---- KV offload -------------------------------------------------------------
# TOTAL_CPU_DRAM_GB is the aggregate host-DRAM budget the matrix generator
# derives from dram-utilization and the runner's available-cpu-dram-mib, capped
# at the 3,095,781 MiB (3 TB decimal) agentic limit. Per
# benchmarks/single_node/agentic/README.md it must be consumed as given and
# never replaced with a model-specific constant.
OFFLOAD_ARGS=()

if agentic_kv_offload_enabled; then
case "${KV_OFFLOAD_BACKEND:-}" in
  vllm-simple)
    require_agentic_kv_offload_backend "$KV_OFFLOAD_BACKEND"
    CPU_BYTES_PER_RANK=$(( TOTAL_CPU_DRAM_GB * 1000 * 1000 * 1000 / TP ))
    # Identical prefixes must hash to identical block keys across ranks.
    export PYTHONHASHSEED=42
    SIMPLE_LAZY_OFFLOAD="${SIMPLE_LAZY_OFFLOAD:-false}"
    OFFLOAD_ARGS=(
        --kv-transfer-config
        "{\"kv_connector\":\"SimpleCPUOffloadConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use_per_rank\":$CPU_BYTES_PER_RANK,\"lazy_offload\":$SIMPLE_LAZY_OFFLOAD}}"
    )
    echo "SimpleCPUOffloadConnector: ${CPU_BYTES_PER_RANK} B/rank x ${TP} ranks, lazy_offload=$SIMPLE_LAZY_OFFLOAD"
    ;;
esac
fi

# ---- LLM server  ------------------------------------------------------------


# ---- Parallelism ------------------------------------------------------------
EP_ARGS=()
if [ "$EP_SIZE" -gt 1 ]; then
    EP_ARGS=(--enable-expert-parallel)
fi

# ---- Decode context parallel (DCP) ------------------------------------------
# OFF by default (DCP_SIZE=1): everything below is inert unless explicitly asked
# for, so the working DSpark recipe is untouched.
#
# Why it is interesting: MLA has ONE latent KV head, so TP cannot shard the KV
# cache -- every TP rank holds a full copy. DCP shards KV along the sequence
# dimension instead, making the pool the sum of all 8 GPUs. The B300 reference
# (run 31893747354, c70) measures the effect exactly:
#     ours  TP8, no DCP : GPU KV  2,646,487 tokens ->  2.52x max-model-len
#     B300  TP8, DCP=8  : GPU KV 21,564,193 tokens -> 20.57x
# i.e. 8.2x more KV, which is the wall we hit every time concurrency scales.
#
# THREE ROCm-SPECIFIC CATCHES, all verified at nightly commit 311b3513:
#  1. platforms/rocm.py:905 force-downgrades cudagraph_mode to PIECEWISE when
#     DCP>1. platforms/cuda.py has NO such gate -- this is ROCm-only, and it
#     silently undoes the FULL cudagraph win the triton_mla patch buys us.
#  2. mla/triton_mla.py:61 sets supports_draft_decode_metadata_update to
#     (dcp_world_size == 1), so at DCP=8 fused multi-step draft decode is off
#     and attention metadata is rebuilt between every draft step.
#  3. dcp_comm_backend is read only by flashinfer.py / flash_attn.py (CUDA-only).
#     Neither rocm_aiter_mla.py nor triton_mla.py references it, so "a2a" is a
#     no-op for us and the generic ag_rs path is what actually runs. We still
#     pass it to stay literally comparable with the B300 command.
# All three are about the drafter, which is why the B300 run turned spec
# decoding OFF at c70 (NUM_SPEC_TOKENS=0, speculative_config=None). DISABLE_SPEC
# below mirrors that: it is the config that actually produced their number.
# Auto-enable only at concurrencies no existing config uses (our conc-list tops
# out at 24), so every current run is bit-for-bit unaffected. The B300 script
# gates the same way -- it disables spec decode above conc 16, which is how c70
# ended up with NUM_SPEC_TOKENS=0.
# T17: threshold 64 -> 20 so DCP engages at the reference concurrency. The
# 5,388 tok/s/GPU reference ran conc 20 with kv-offloading dram; matching it
# makes DCP the only variable in the comparison.
# T26 measured c8: 969 tok/s/GPU, -52% vs c20 for +11% interactivity. The
# concurrency axis is closed and c20 is the DCP optimum -- back to 20.
DCP_AUTO_CONC_THRESHOLD="${DCP_AUTO_CONC_THRESHOLD:-20}"
if [ "$CONC" -ge "$DCP_AUTO_CONC_THRESHOLD" ]; then
    # HISTORY, not current config: DCP is 8 (see DCP_SIZE below). This block
    # once forced DCP=4 as a workaround; patch [6] fixed the underlying bug and
    # T18/T19 both ran DCP=8 cleanly (confirmed decode_context_parallel_size=8
    # in their server logs). Kept because the diagnosis is worth preserving.
    #
    # Run 32025696861 died with HSA_STATUS_ERROR_EXCEPTION 0x1016
    # on all 8 queues, and the crash dump pins it exactly:
    #   num_computed_tokens=126720, num_scheduled_tokens=7680, num_output_tokens=0
    # i.e. a chunked-PREFILL continuation, no decode in flight. vLLM narrows every
    # DCP block table to max_model_len/dcp_world_size
    # (kv_cache_interface.py:253-256 and :293-295), so at DCP=8 the budget is
    #   1048576 / 8 = 131072 tokens
    # and that request crossed it: 126720 fits, 126720+7680 = 134400 does not.
    # A group replicated across ranks rather than sharded then indexes past its
    # block table -> OOB read -> 0x1016. vllm-project/vllm#51705 fixes this
    # properly with cp_exempt_groups; it is still OPEN and not in any nightly.
    # DCP=4 raises the budget to 1048576/4 = 262144, clear of our ~134k
    # sequences. It masks the bug rather than fixing it -- revisit at DCP=8 once
    # #51705 lands. max_model_len cannot go the other way: mpe is 1048576 and
    # DCP=8 would need >= 1075200.
    # Verified locally that the 48-head gathered shape (12 local x DCP4) still
    # resolves a CP+LSE kernel -- mla_a16w16_qh64_qseqlen1_gqaratio64_lse_cprr_v3_ps
    # -- and merges to rel 3.1e-03, despite gqa=48 being excluded from aiter's
    # first normalization branch.
    # Back to 8 under PR #51705, which is supposed to fix the block-table
    # narrowing that forced DCP=4. If 0x1016 returns at 1048576/8 = 131072 the
    # PR's cp_exempt_groups did not cover our (spec-off) group set.
    # T24: DCP=2, not 8. The decode penalty is per-layer collectives across the
    # DCP group (T21 proved it is not the kernel: swapping ROCM_AITER_MLA ->
    # TRITON_MLA moved nothing). Collective traffic scales with world size, and
    # we do not need the capacity: kv_usage peaked at 15% of DCP=8's 31.22x.
    # With spec OFF there is no replicated draft group, so capacity is simply
    # W x M/b -- DCP=2 still gives ~7.8M tokens (~7.5x model-len), far above the
    # ~4.9M we actually touch, while cutting collective traffic ~4x.
    # (This is the inverse of the DCP=2 idea I floated for MTP, which was wrong:
    # with a draft group present, capacity saturates at M/d and lower DCP loses.)
    # T25: DCP=4. T24 tried 2 and died at init on
    #   assert AiterMLAHelper.is_valid_num_heads(num_heads)
    # which requires num_heads < 16 OR num_heads % 16 == 0. DCP gathers
    # 12 heads/rank x W, so W=2 gives 24 -- neither. Valid sizes for this model
    # are 1, 4, 8 (12, 48, 96 heads). PR #51705 applies that check to the
    # GATHERED count, which is what fires.
    # 4 still halves collective traffic vs 8 and leaves ~16.4M KV tokens against
    # the ~4.9M we touch.
    DCP_SIZE="${DCP_SIZE:-4}"
    # T14: spec decoding ON under DCP. This only became possible with the
    # colleague image: DSpark draft verify under DCP now supports both ASM and
    # Gluon there. On our nightlies aiter is pinned to v0.1.19, whose mla_gluon
    # asserts "nhead <= 16 or nhead in (64,128)" -- no kernel for the 96 heads a
    # TP8/DCP8 gather produces -- and the ASM path has no gqa=64 kernel past
    # qseqlen 1, so qlen>1 verify was unreachable. Per the colleague: ASM must
    # slice 96->12 then pad to 16 and cannot multi-token verify (token-by-token
    # at qlen=1 only); Gluon supports qh96 natively but must flatten rather than
    # use native 4-D MTP. Either way MTP under DCP is finally testable.
    # Set DISABLE_SPEC=1 to get the previous spec-off behaviour back.
    # T15: spec OFF again. T14 proved MTP-under-DCP is reachable on the
    # colleague image -- it routed to Gluon without complaining about nhead, so
    # that image does carry a post-aiter#4412 build -- but it died with
    #   mla_gluon[bh16bn128] requires batch_size=1, got 64
    # i.e. the flattened MTP verify path the colleague described is single-batch
    # only, so it cannot serve a concurrent benchmark. Set DISABLE_SPEC=0 to
    # reproduce that, ideally at conc 1.
    # T20: MTP ON under DCP. Re-reading the code, my earlier claim that MTP is
    # impossible under DCP was WRONG for TRITON_MLA. The gate
    #   triton_mla.py:61  supports_draft_decode_metadata_update = dcp_world_size == 1
    # only disables FUSED multi-step draft decode; speculator.py:112 falls back to
    # "rebuilding attention metadata between draft steps" -- slower, not disabled.
    # The hard blocker was config/speculative.py's ValueError ("MLA DSpark does
    # not currently support decode context parallelism"), and PR #51705 (patch
    # [5]) deletes it. The two REAL kernel gaps -- ASM has no gqa=64 kernel past
    # qseqlen 1, Gluon's bh16bn128 needs batch_size=1 -- are both in the AITER
    # backend, which we are no longer using for decode.
    # Worth ~2.5x on TPOT (DSpark acceptance ~2.51), the single biggest remaining
    # lever. Set DISABLE_SPEC=1 to revert.
    # T21: spec OFF again. T20 (TRITON_MLA + MTP) degraded steadily --
    #   08:08 profiling: kv_usage 81.2% cache 5.6% tput_in_srv 5,062/s
    #   23:10 profiling: kv_usage 81.5% cache 3.7% tput_in_srv 2,897/s
    # vs T18's kv_usage ~15%, cache 87%, tput_in_srv ~17,000/s. Enabling spec
    # loads the DSpark drafter as a second model with its own KV, which ate the
    # headroom that made T18 work; the prefix cache then collapsed and we went
    # back to recomputing ~100k-token prefills. itl p50 183-193 ms was also no
    # better than T18's 174 ms, so MTP bought nothing even before the KV cost.
    # Conclusion: MTP is off the table under DCP for capacity reasons, not just
    # kernel ones.
    DISABLE_SPEC="${DISABLE_SPEC:-1}"
    # NOTE: run 32005332130 died with
    #   ValueError: Selected MLA prefill backend ROCM_AITER_FA is not valid ...
    #   Reason: ['required dependencies not available']
    # and I wrongly blamed a missing triton_kernels.matmul_ogs in the image.
    # The real cause: aiter_flash_attn.py:42 is_available() returns
    # rocm_aiter_ops.is_enabled(), i.e. VLLM_ROCM_USE_AITER -- which was unset
    # because the reference env block above was commented out. base.py:105 then
    # reports the generic "required dependencies not available". With the env
    # block restored, ROCM_AITER_FA is available again, so keep the pin.
    echo "DCP: CONC=$CONC >= $DCP_AUTO_CONC_THRESHOLD -> B300-style config (DCP=8, spec decode off)"
fi
DCP_SIZE="${DCP_SIZE:-8}"
# fp8 KV everywhere except the DCP path, which overrides this to bf16 below --
# every measured number to date (c12=4431 ... c20=5022) is on fp8, so the
# non-DCP arms must stay bit-for-bit unchanged.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
CP_ARGS=()
if [ "$DCP_SIZE" -gt 1 ]; then
    # Decode MUST be TRITON_MLA under DCP. Run 32011858320 died at init with
    #   RuntimeError: Decode Context Parallelism (DCP) requires attention
    #   implementations to return the softmax LSE during decode, but
    #   AiterMLAImpl does not.
    # ISOLATION (route A) for run 32012699807, which died ~10 min in with
    #   HSA_STATUS_ERROR_EXCEPTION: An HSAIL operation resulted in a hardware
    #   exception. code: 0x1016
    # on all 8 queues. The GPU coredump failed to write, so the faulting kernel
    # is unattributed. The only variable between that run and 32009028600 (which
    # did not fault) is AITER. Neither MLA prefill backend has ANY DCP awareness
    # -- 0 references to dcp/context_parallel in aiter_flash_attn.py and
    # flash_attn.py -- yet under DCP they receive tensors whose context length
    # and head count are scaled by dcp_world_size. So: fall back to FLASH_ATTN
    # prefill while KEEPING AITER for MoE. If this survives, the fault is in
    # ROCM_AITER_FA under DCP. Set DCP_PREFILL_BACKEND=ROCM_AITER_FA to undo.
    # Route A (FLASH_ATTN isolation) is superseded: HSA_NO_SCRATCH_RECLAIM=1 is a
    # much stronger candidate for a hardware exception than the prefill backend,
    # and it was missing. Keep ROCM_AITER_FA so that if the env parity fixes the
    # fault we get the fast prefill path too -- still a clean A/B against
    # 32012699807, which had identical settings minus the six new vars.
    MLA_PREFILL_BACKEND="${DCP_PREFILL_BACKEND:-ROCM_AITER_FA}"
    # Decode is now ROCM_AITER_MLA, not TRITON_MLA. The claim above -- that
    # "ROCM_AITER_MLA and DCP are mutually exclusive" because the ASM kernel
    # cannot emit a per-shard log-sum-exp -- was wrong. aiter's mla_decode_fwd
    # has taken return_lse/cp_world_size/cp_rank/g_kv_indptr all along and
    # gfx950 ships the CP round-robin kernels; only vLLM's wrapper was missing.
    # apply_kimi_k3_patches.sh patch [4] (KIMI-PATCH-DCP-LSE) plumbs it through,
    # which both clears the cp_utils.py:46 assert and sidesteps the TRITON_MLA
    # HSA 0x1016 fault entirely. Set DCP_ATTN_BACKEND=TRITON_MLA to go back.
    #
    # The patch hard-requires qlen==1 under DCP (aiter has no gqa=64 CP kernel
    # past qseqlen 1), which the DISABLE_SPEC=1 above already gives us.
    #
    # DCP also forces a bf16 KV cache. Every CP round-robin row in aiter's
    # kernel table (aiter_meta/hsa/gfx950/mla/mla_asm.csv) is bf16/bf16 -- there
    # is simply no cprr kernel for an fp8 KV cache, and a table miss aborts the
    # process rather than raising. This is a REAL COST, not a formality: bf16
    # doubles the bytes per KV token, so DCP=8 buys 8x/2 = 4x effective KV
    # capacity over the fp8 non-DCP baseline, not the 8x a naive reading gives.
    # Set DCP_KV_CACHE_DTYPE=fp8 to observe the guard fire.
    # fp8, per PR #52248's tested config. The bf16 requirement belongs to OUR
    # patch [4], which drives aiter's bf16-only cprr kernels; PR #51705 calls
    # mla_decode_fwd without cp_world_size/g_kv_indptr, so it uses the plain LSE
    # kernels, which do have fp8 variants. Doubles KV capacity vs bf16.
    KV_CACHE_DTYPE="${DCP_KV_CACHE_DTYPE:-fp8}"
    echo "DCP: kv-cache-dtype=$KV_CACHE_DTYPE (fp8 has no CP kernel in aiter)"
    CP_ARGS=(--decode-context-parallel-size "$DCP_SIZE"
             # T28 measured both DCPCommBackend values on ROCm and a2a won on
             # every metric (2,034 vs 1,978 tok/s/GPU, TPOT 0.167 vs 0.184), so
             # the upstream default ag_rs is the worse choice here. Back to a2a.
             --dcp-comm-backend "${DCP_COMM_BACKEND:-a2a}"
             # Back to ROCM_AITER_MLA: T21 showed TRITON_MLA is within noise
             # (1,948 vs 1,990 tok/s/GPU, TPOT 0.186 vs 0.174), so gate the
             # configuration that actually produced the best DCP number.
             --attention-backend "${DCP_ATTN_BACKEND:-ROCM_AITER_MLA}")
    # Env from vllm-project/vllm#52248's tested DCP config. The four
    # VLLM_USE_DIRECT_DCP_* / VLLM_DCP_Q_REPLICATE disables turn off the
    # symmetric-memory direct DCP paths; that is very likely why upstream could
    # capture cudagraphs under DCP where dcp_utils' all_gather(query) deadlocked
    # for us. AITER_DISABLE_FMHA_OPUS avoids the fmha_fwd_bf16_opus path.
    export VLLM_ROCM_USE_AITER_MLA=1
    export AITER_SITUV2_A8W4=1
    export VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1
    export AITER_BF16_FP8_MOE_BOUND=0
    export AITER_DISABLE_FMHA_OPUS=1
    # T27 flipped all three to 1 and the engine died at startup:
    # torch.ops._C has no attribute 'direct_dcp_a2a_lse_reduce'. The Python call
    # site ships in vllm/v1/attention/ops/dcp_utils.py:262 but upstream compiles
    # csrc/libtorch_stable/attention/dcp_utils/*.cu ONLY inside
    # if(VLLM_GPU_LANG STREQUAL "CUDA"), so the ROCm build has no direct_dcp ops.
    #
    # T31: that is now fixed for the a2a combine ONLY. Reading the sources shows
    # the three kernels are not equally portable:
    #   q_gather / kv_gather -> multimem.st.* PTX under #if __CUDA_ARCH__ >= 900,
    #     i.e. NVIDIA NVLink hardware multicast. No AMD equivalent. NOT ported.
    #   a2a_lse_reduce       -> zero multimem; only st.global.release.sys.u32 and
    #     ld.global.acquire.sys.u32, which map exactly onto __hip_atomic_store /
    #     __hip_atomic_load at __ATOMIC_RELEASE/ACQUIRE + __HIP_MEMORY_SCOPE_SYSTEM.
    # The ported kernel is built into the image as
    # /opt/dcp/vllm_dcp_direct_rocm.so and verified to resolve:
    #   torch.ops._C.direct_dcp_a2a_lse_reduce -> _C.direct_dcp_a2a_lse_reduce
    #
    # Why the combine is the right target: T25 (world size 8->4, -4%), T26
    # (batch 20->8, -2%) and T28 (a2a vs ag_rs, within 10%) all showed the
    # combine is a FIXED per-decode-step cost. Both ROCm combine paths go via
    # RCCL; a direct peer-to-peer combine is the only untried mechanism.
    #
    # Q_GATHER/KV_GATHER stay 0 -- their kernels are absent and would trap.
    export VLLM_USE_DIRECT_DCP_A2A="${VLLM_USE_DIRECT_DCP_A2A:-0}"  # T31b: measured -0.9% tput / +2.4% TPOT vs RCCL a2a. Kernel works; it just does not help. Set 1 to re-enable.
    export VLLM_USE_DIRECT_DCP_Q_GATHER=0
    export VLLM_USE_DIRECT_DCP_KV_GATHER=0
    export VLLM_DCP_Q_REPLICATE=0
    # T29: KV shard granularity 1 -> 16. This is the last live, untested DCP
    # knob on our code path (mla_attention.py:2099 dcp_local_block_size).
    #   interleave=1  : token-level round-robin -- every rank's shard is strided
    #                   across the whole sequence, worst locality for the decode
    #                   read that feeds the combine.
    #   interleave=16 : 16-token contiguous runs per rank.
    # Constraint (config/parallel.py:371-372): interleave <= block_size and
    # block_size % interleave == 0. block_size is the platform default here, so
    # 16 is valid for any block_size in {16,32,64,128}; an invalid value is
    # rejected at startup, which is a cheap failure.
    # This does not change the collective's volume -- T27/T28 showed that is
    # fixed -- it changes how efficiently each rank reads its own shard.
    CP_ARGS+=(--cp-kv-cache-interleave-size "${CP_INTERLEAVE:-1}")
    echo "DCP: decode-context-parallel-size=$DCP_SIZE comm-backend=${DCP_COMM_BACKEND:-a2a} interleave=${CP_INTERLEAVE:-16} kv=$KV_CACHE_DTYPE"
    echo "DCP: expect a PIECEWISE downgrade warning from platforms/rocm.py -- that is the known ROCm gate."
fi

# ---- Speculative ------------------------------------------------------------
SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-2}"
SYNTHETIC_ACCEPT_LEN=2.51

if [ "${DISABLE_SPEC:-0}" = "1" ]; then
    # Mirrors the B300 c70 path, which fell through to NUM_SPEC_TOKENS=0.
    # Note this makes throughput real rather than synthetic: with no drafter
    # there is no rejection_sample_method="synthetic" inflating the token count.
    SPEC_ARGS=()
    echo "spec decoding: DISABLED (DISABLE_SPEC=1) -- matches the B300 c70 reference"
elif [ "${EVAL_ONLY:-false}" = "true" ]; then
    SPEC_ARGS=(
        --speculative-config
        "{\"model\":\"Inferact/Kimi-K3-DSpark\",\"num_speculative_tokens\":$SPEC_NUM_TOKENS,\"method\":\"dspark\",\"attention_backend\":\"TRITON_MLA\",\"kv_cache_dtype\":\"auto\",\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\": \"block\"}"
    )
else
    SPEC_ARGS=(
        --speculative-config
        "{\"model\":\"Inferact/Kimi-K3-DSpark\",\"num_speculative_tokens\":$SPEC_NUM_TOKENS,\"method\":\"dspark\",\"attention_backend\":\"TRITON_MLA\",\"kv_cache_dtype\":\"auto\",\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\": \"synthetic\", \"synthetic_acceptance_length\": $SYNTHETIC_ACCEPT_LEN}"
    )
fi

# ---- Chunked prefill sizing --------------------------------------------------
# Chunked prefill is ON by default and max_num_batched_tokens defaults to 16384
# (confirmed in the c12 server log; neither was previously set by this recipe).
# With ~99k-token mean inputs that is ~6 prefill chunks per request.
# 8192 doubles that to ~12 smaller chunks: finer interleaving with decode, at the
# cost of more per-chunk boundaries. Note vLLM PR #51862 (in the new image) exists
# specifically to remove a per-chunk KDA prefill stall, so more chunks may cut
# against that gain -- this run measures which effect dominates.
# B300 does not set this flag at all, so we are moving off their config here.
# T13: 8192 -> 32768. This workload is ~99% prefill by token count (T12: input
# 7,944 tok/s vs output 50), so total throughput IS prefill throughput. At 8192
# a ~137k-token prompt is ~17 separate chunks, and on this nightly every one of
# those GEMMs logs "not found tuned config ... using default config" because the
# image ships no merged_bf16_tuned_gemm.csv. Untuned GEMMs are most penalised at
# small M, so fewer/larger chunks should recover some of that.
# Concurrency is NOT the lever here: T12's TTFT was 396 s, i.e. already deeply
# queued, so raising conc adds latency without adding prefill rate.
# T18: back to 8192. The 32768 experiment (T13/T16) was never completed, and
# leaving it set broke T17: DCP's 31x KV pool at gpu-memory-utilization 0.9 plus
# 32k-token activation tensors (aiter logged M:32755, N:8448, K:7168) left only
# 2408 MB free and the run died with HSA_STATUS_ERROR_OUT_OF_RESOURCES. 8192 is
# also what the 5,388 reference uses, so this keeps DCP the only variable.
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
CHUNKED_PREFILL_ARGS=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")

# ---- Async scheduling / KV block-pool stability ------------------------------
# DSpark is the ONLY spec method exempted from vLLM's async-scheduling disable
# list (config/vllm.py:1181), so async_scheduling resolves True here. That gives
# max_concurrent_batches = pp_size + 1 = 2 (vllm.py:563-569), and with
# kv_role=kv_both (is_kv_consumer=True) the scheduler sets defer_block_free=True
# (sched/scheduler.py:155-157). Its own comment: "a step may still be writing a
# freed request's KV blocks. A consumer KV Connector can reallocate and fill
# those blocks via a load that isn't ordered against that write."
#
# That limbo state matches our crash signature exactly -- the engine dies with
#   block_pool.py:667  assert block.ref_cnt == 0
# i.e. a block sitting on the FREE list that is still referenced. Crash time
# scales inversely with concurrency: c10 survived 3612 s, c12 died at 487 s,
# c16 at 354 s. Note vLLM already disables async scheduling for ROCm DeepEP DBO
# because "that combination can corrupt" state.
#
# Setting max_concurrent_batches back to 1 makes defer_block_free unreachable.
# Cost: async scheduling exists to fill GPU-utilisation gaps, so expect to give
# some throughput back. Set ASYNC_SCHEDULING=1 to restore the default.
ASYNC_SCHED_ARGS=()
if [ "${ASYNC_SCHEDULING:-0}" != "1" ]; then
    ASYNC_SCHED_ARGS=(--no-async-scheduling)
fi

# ---- MLA prefill backend -----------------------------------------------------
# On ROCm the prefill priority is [ROCM_AITER_FA, FLASH_ATTN]. ROCM_AITER_FA
# JIT-builds module_fmha_fwd_bf16_opus at runtime; that module registers its own
# aiter_tensor_t, distinct from the one in the prebuilt module_aiter_core, so the
# first call dies with:
#   TypeError: fmha_fwd_bf16_opus_fwd(): incompatible function arguments
# during compile_or_warm_up_model -> _dummy_run, before the server binds.
# Pinning FLASH_ATTN keeps every AITER MoE kernel (and its throughput) while
# skipping only the broken FMHA prefill path.
# UPDATE: the AITER packaging issue is now fixed at source by
# apply_kimi_k3_patches.sh (run above), so ROCM_AITER_FA is usable again and
# is the default. Measured on 8x MI355X / Kimi-K3 MXFP4 TP8, cold prefill:
#   ~24k ctx  FLASH_ATTN 12,953 -> AITER 13,524 tok/s  (+4.4%)
#   ~93k ctx  FLASH_ATTN 11,174 -> AITER 13,423 tok/s  (+20.1%)
# This workload averages ~99k input tokens, so the ~93k figure is the relevant
# one. Set MLA_PREFILL_BACKEND=FLASH_ATTN to fall back if AITER regresses.
# Note "-" not ":-": an explicitly-empty MLA_PREFILL_BACKEND means "let vLLM
# auto-select" and must survive. Newer images (the unified one, and nightly
# 311b3513) ship a Triton without triton_kernels.matmul_ogs, so pinning
# ROCM_AITER_FA there dies at init with
#   ValueError: Selected MLA prefill backend ROCM_AITER_FA is not valid for
#   this configuration. Reason: ['required dependencies not available']
MLA_PREFILL_BACKEND="${MLA_PREFILL_BACKEND-ROCM_AITER_FA}"
MLA_PREFILL_ARGS=()
if [ -n "$MLA_PREFILL_BACKEND" ]; then
    MLA_PREFILL_ARGS=(
        --attention-config
        "{\"mla_prefill_backend\":\"$MLA_PREFILL_BACKEND\"}"
    )
fi

# ---- HIP graph ------------------------------------------------------------
# max_num_seqs scales with concurrency, matching what the B300 reference actually
# does (read out of its vllm_command.txt): conc 1/2/4/8/16 -> max_num_seqs
# 2/4/8/16/32, i.e. 2 x CONC. We had this hardcoded at 20 for every conc, which
# was tight at c16 and over-provisioned at c1 -- neither like-for-like.
#
# Headroom matters because in-flight sequences EXCEED nominal concurrency: the
# agentic harness branches, and c12 was measured at peak 14 running vs nominal
# 12 (1.17x). 2x covers that comfortably.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-$(( CONC * 2 ))}"

# INVARIANT: with DSpark a decode step is num_seqs * (1 + num_speculative_tokens)
# = num_seqs * 3 tokens, so the capture ceiling must be >= 3 * max_num_seqs or
# large decode batches fall out of the captured graphs and attention runs eager
# every step (the get_mla_metadata_v1 host bubble, ~75 ms ITL).
# NOTE: 3 * CONC would be WRONG here -- at c12 that is 36, but the real peak was
# 14 seqs = 42 tokens, which would have escaped capture. Derive from
# max_num_seqs, not from CONC.
# The list is dense, so every 3*C is an exact captured size (no rounding up) and
# there is no need to hand-add sizes the way a sparse ladder would.
# Cost measured on this recipe: 1.07 GiB and ~54 s to capture, so headroom is cheap.
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-$(( MAX_NUM_SEQS * 3 ))}"

if [ "$DCP_SIZE" -gt 1 ]; then
    # Sparse ladder, as the B300 reference uses (max_num_seqs 140 with
    # [1,2,...,128,256,...,8192] rather than a dense list). The dense 1..3N list
    # above is only affordable because N is small: at c72 it would be 1..432,
    # which is minutes of capture for sizes a decode batch never actually hits.
    # The dense list also exists to guarantee every 3*C is an exact captured
    # size for the DSpark drafter -- irrelevant here, since DCP forces PIECEWISE
    # and DISABLE_SPEC removes the drafter entirely.
    CUDAGRAPH_CAPTURE_SIZES="1,2,4,8,16,24,32,48,64,96,128,160,192,256,320,384,512"
    MAX_CUDAGRAPH_CAPTURE_SIZE=512
    # Run 32005837765: PIECEWISE capture DEADLOCKED at graph 3/17 for 30 min, then
    #   query = self.group.all_gather(query, dim=...)
    #   DistBackendError: NCCL error ... unhandled cuda error, NCCL version 2.27.7
    # That is the DCP query-gather for MLA. On NVIDIA this runs as "direct
    # symmetric-memory DCP query gather"; on ROCm no backend implements it, so it
    # degrades to a plain RCCL all_gather -- and capturing a collective in a HIP
    # graph hangs. rocm.py's FULL->PIECEWISE downgrade is not enough, because
    # PIECEWISE still captures. NONE is the only mode that avoids capture entirely.
    # Cost is real but bounded here: this workload is ~99% prefill by token count
    # (B300 c70: 99,539 input tok/s vs 698 output tok/s) and prefill does not use
    # decode cudagraphs anyway.
    # T7: NONE was costing us everything on decode. Trial 5 measured TPOT 663 ms
    # and trial 6's GSM8K could not finish -- the server generated at 3.5-6.8
    # tok/s with one request in flight, so 1319 questions would take ~15 h.
    # Patch [2]'s own measurement was 14.05 -> 77.65 tok/s (ITL 71.16 -> 12.88 ms)
    # from cudagraphs alone, so eager decode plausibly explains most of the gap.
    #
    # The historical objection was run 32005837765, where PIECEWISE capture
    # deadlocked at graph 3/17 in dcp_utils' all_gather(query). Since then we
    # export VLLM_USE_DIRECT_DCP_{A2A,Q_GATHER,KV_GATHER}=0 and
    # VLLM_DCP_Q_REPLICATE=0 from PR #52248, whose author ran PIECEWISE under
    # DCP8 to completion, as did the colleague recipe. Small capture ladder to
    # keep capture time bounded and to stay near their cap of 16.
    # Set DCP_CUDAGRAPH_MODE=NONE to revert if capture hangs again.
    CUDAGRAPH_CAPTURE_SIZES="1,2,4,8,16,24,32,48,64"
    MAX_CUDAGRAPH_CAPTURE_SIZE=64
    CUDAGRAPH_MODE_OVERRIDE="${DCP_CUDAGRAPH_MODE:-PIECEWISE}"
    echo "cudagraph sizing: CONC=$CONC max_num_seqs=$MAX_NUM_SEQS DCP mode -> cudagraph_mode=$CUDAGRAPH_MODE_OVERRIDE capture<=64"
else
    CUDAGRAPH_CAPTURE_SIZES="$(seq -s, 1 "$MAX_CUDAGRAPH_CAPTURE_SIZE")"
    echo "cudagraph sizing: CONC=$CONC max_num_seqs=$MAX_NUM_SEQS capture<=$MAX_CUDAGRAPH_CAPTURE_SIZE (3x max_num_seqs)"
fi
CUDAGRAPH_MODE="${CUDAGRAPH_MODE_OVERRIDE:-FULL_AND_PIECEWISE}"
if [ "$CUDAGRAPH_MODE" = "NONE" ]; then
    # No capture, so the size list is meaningless -- omit it rather than ship a
    # list vLLM will ignore.
    COMPILATION_CONFIG_ARGS=(--compilation-config "{\"mode\":3,\"cudagraph_mode\":\"NONE\",\"custom_ops\":[\"+fused_rms_norm_gated\"]}")
else
    COMPILATION_CONFIG_ARGS=(--compilation-config "{\"mode\":3,\"cudagraph_mode\":\"$CUDAGRAPH_MODE\",\"max_cudagraph_capture_size\":$MAX_CUDAGRAPH_CAPTURE_SIZE,\"custom_ops\":[\"+fused_rms_norm_gated\"],\"cudagraph_capture_sizes\":[$CUDAGRAPH_CAPTURE_SIZES]}")
fi

GPU_MEM_UTIL="0.9"

echo "Starting vllm server..."
export PYTHONNOUSERSITE=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1200}"

{ set +x; } 2>/dev/null
# ---- Unified-image mode (aigmkt/k3-unified-v2-from-cb810) --------------------
# Self-detecting: the unified image ships /opt/aiter-local with a merged tuned
# GEMM table. On the stock nightly that path does not exist and this whole block
# is skipped, so the working recipe is untouched.
#
# Everything below mirrors validate_live_graph_capture.sh from the image's own
# build context. Do NOT mix these with our in-container patches -- the image
# already carries the PR2585 bundle, and our ROCM_AITER_FA prefill pin is exactly
# what made the first smoke test die with
#   ValueError: Selected MLA prefill backend ROCM_AITER_FA is not valid ...
#               Reason: ['required dependencies not available']
# (the image ships a different Triton, so triton_kernels.matmul_ogs is absent).
UNIFIED_GEMM_CSV=/opt/aiter-local/aiter/configs/merged_bf16_tuned_gemm.csv
if [ -f "$UNIFIED_GEMM_CSV" ]; then
    echo "=== unified image detected: applying its documented runtime config ==="
    export GPU_ARCHS=gfx950
    export VLLM_ROCM_USE_AITER=1
    export VLLM_ROCM_USE_AITER_MOE=1
    export AITER_SITUV2_A8W4=1
    export VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1
    export AITER_BF16_FP8_MOE_BOUND=0
    export VLLM_USE_BREAKABLE_CUDAGRAPH=0
    export SAFETENSORS_FAST_GPU=1
    export VLLM_ROCM_AITER_MLA_ASM_PADDING=asm
    export AITER_CONFIG_GEMM_BF16="$UNIFIED_GEMM_CSV"
    export HSA_NO_SCRATCH_RECLAIM=1
    export VLLM_K3_KDA_SAFE_STAGES=1
    export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1

    # apply_dspark_fp8asm.sh step 2/7 forces the DSpark draft causal so that
    # ROCM_AITER_MLA is a legal draft backend (otherwise vLLM rejects it with
    # "non-causal attention not supported"). Their script hardcodes
    # /dev/shm/hf-cache, which does not exist here, so it would silently no-op
    # and leave the draft non-causal. Search the real cache instead.
    python3 - <<'PYFORCE'
import glob, json, os, shutil
roots = [os.environ.get("HF_HUB_CACHE",""), os.environ.get("HF_HOME",""),
         "/mnt/hf_hub_cache", "/home/models", "/dev/shm/hf-cache", "/root/.cache/huggingface/hub"]
pats = []
for r in roots:
    if r:
        pats += [f"{r}/models--Inferact--Kimi-K3-DSpark/snapshots/*/config.json",
                 f"{r}/**/models--Inferact--Kimi-K3-DSpark/snapshots/*/config.json"]
hits = []
for p in pats:
    hits += glob.glob(p, recursive=True)
if not hits:
    print("  !! DSpark draft config NOT found -- draft stays non-causal; "
          "ROCM_AITER_MLA will be rejected for the draft")
for f in sorted(set(hits)):
    c = json.load(open(f))
    d = c.setdefault("dflash_config", {})
    if d.get("causal") is True:
        print("  draft already causal:", f)
    else:
        if not os.path.exists(f + ".orig.bak"):
            shutil.copy2(f, f + ".orig.bak")
        d["causal"] = True
        json.dump(c, open(f, "w"), indent=2)
        print("  forced causal:", f)
PYFORCE

    # Their serve flags. ROCM_AITER_MLA everywhere; no separate prefill pin.
    MLA_PREFILL_ARGS=()
    ASYNC_SCHED_ARGS=(--async-scheduling)
    CHUNKED_PREFILL_ARGS=()
    UNIFIED_ARGS=(
        --attention-backend ROCM_AITER_MLA
        --distributed-executor-backend mp
        --enable-prompt-tokens-details
        --no-disable-hybrid-kv-cache-manager
        --disable-uvicorn-access-log
    )
    UNIFIED_MOE_BACKEND=aiter
    UNIFIED_LOAD_FORMAT=auto
    SPEC_ARGS=(
        --speculative-config
        "{\"method\":\"dspark\",\"model\":\"Inferact/Kimi-K3-DSpark\",\"num_speculative_tokens\":2,\"attention_backend\":\"ROCM_AITER_MLA\",\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"synthetic\",\"synthetic_acceptance_length\":2.51}"
    )
    # Their capture ladder (sparse, to 384) rather than our dense 1..N.
    CUDAGRAPH_CAPTURE_SIZES="1,2,4,8,12,16,24,32,36,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256,272,288,304,320,336,352,368,384"
    COMPILATION_CONFIG_ARGS=(--compilation-config "{\"mode\":3,\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"custom_ops\":[\"+fused_rms_norm_gated\"],\"cudagraph_capture_sizes\":[$CUDAGRAPH_CAPTURE_SIZES]}")
else
    UNIFIED_ARGS=()
    UNIFIED_MOE_BACKEND=auto
    UNIFIED_LOAD_FORMAT=fastsafetensors
fi

VLLM_CMD=(
    vllm serve "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    --moe-backend "$UNIFIED_MOE_BACKEND"
    --tensor-parallel-size "$TP"
    --load-format "$UNIFIED_LOAD_FORMAT"
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
    "${CP_ARGS[@]}"
    "${UNIFIED_ARGS[@]}"
    "${ASYNC_SCHED_ARGS[@]}"
    "${MLA_PREFILL_ARGS[@]}"
    "${COMPILATION_CONFIG_ARGS[@]}"
    "${SPEC_ARGS[@]}"
    "${OFFLOAD_ARGS[@]}"
)
# vLLM only registers /start_profile and /stop_profile when this is set at boot,
# so it must be exported before the server launches, not after.
if [ "${PROFILE_DECODE:-0}" = "1" ]; then
    export VLLM_TORCH_PROFILER_DIR="${VLLM_TORCH_PROFILER_DIR:-$RESULT_DIR/torch_profile}"
    mkdir -p "$VLLM_TORCH_PROFILER_DIR"
    echo "[profile-decode] VLLM_TORCH_PROFILER_DIR=$VLLM_TORCH_PROFILER_DIR"
fi
printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# ---- PROFILE_DECODE: capture a DECODE-ONLY torch trace -----------------------
# Why this exists. The DCP decode penalty is 124 ms/step (167 vs 43 non-DCP), but
# the communication it performs is ~73 MB/step across 24 MLA layers -- about
# 1.5 ms at 50 GB/s, or 1.4 ms if you cost it as 48 latency-bound collectives.
# So ~122 ms is unaccounted for, and eleven black-box config sweeps (world size,
# batch, algorithm, mechanism, granularity, backend, cudagraphs...) all moved it
# <=4%. A sweep cannot localise this; a trace can.
#
# The agentic replay path has no profiling hook and an hour-long trace would be
# dominated by prefill anyway, so this drives a small decode-only window:
# a few long-output requests to reach steady-state decode, then
# /start_profile .. /stop_profile around it. Minutes of GPU, not an hour.
if [ "${PROFILE_DECODE:-0}" = "1" ]; then
    PROF_DIR="${VLLM_TORCH_PROFILER_DIR:-$RESULT_DIR/torch_profile}"
    mkdir -p "$PROF_DIR"
    echo "[profile-decode] dir=$PROF_DIR"

    # Load generator: long outputs, short prompts -> almost pure decode.
    PROF_CONC="${PROFILE_DECODE_CONC:-$CONC}"
    for i in $(seq 1 "$PROF_CONC"); do
        curl -s -m 600 "http://0.0.0.0:$PORT/v1/completions" \
            -H 'Content-Type: application/json' \
            -d "{\"model\":\"$MODEL\",\"prompt\":\"Count upward from $i, one number per line.\",\"max_tokens\":4096,\"temperature\":0,\"stream\":false}" \
            > /dev/null 2>&1 &
    done
    PROF_LOAD_PIDS=$(jobs -p | tr '\n' ' ')

    # Let the batch settle into decode before the trace opens.
    sleep "${PROFILE_DECODE_WARM_S:-25}"
    echo "[profile-decode] starting trace"
    curl -s -m 60 -X POST "http://0.0.0.0:$PORT/start_profile" || true
    sleep "${PROFILE_DECODE_WINDOW_S:-15}"
    curl -s -m 300 -X POST "http://0.0.0.0:$PORT/stop_profile" || true
    echo "[profile-decode] trace stopped; waiting for writer flush"
    sleep "${PROFILE_DECODE_FLUSH_S:-60}"

    for p in $PROF_LOAD_PIDS; do kill "$p" 2>/dev/null || true; done
    ls -la "$PROF_DIR" || true
    echo "[profile-decode] done; skipping the benchmark arm"
elif [ "${EVAL_ONLY}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi

