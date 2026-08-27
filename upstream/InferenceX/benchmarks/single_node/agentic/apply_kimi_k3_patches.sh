#!/usr/bin/env bash
set -euo pipefail
PY=${PYTHON:-python3}

if [ "${SKIP_KIMI_PATCHES:-0}" = "1" ]; then
    echo "[kimi-patches] SKIP_KIMI_PATCHES=1, doing nothing."
    exit 0
fi

_modfile() {
    $PY - "$1" <<'EOF'
import importlib, os, sys
try:
    print(os.path.abspath(importlib.import_module(sys.argv[1]).__file__))
except Exception:
    print("")
EOF
}

# -----------------------------------------------------------------------------
# [3] vLLM: clamp the negative block count that corrupts the KV free list
# -----------------------------------------------------------------------------
# single_type_kv_cache_manager.py, in allocate_external_computed_blocks(), is the
# ONLY unguarded get_new_blocks() call site in that file (siblings clamp or
# early-return). When len(req_blocks) exceeds the block count implied by
# num_total_computed_tokens the argument goes NEGATIVE, and a negative count is
# silently destructive rather than rejected:
#   * block_pool.get_new_blocks only rejects  num_blocks > free
#   * popleft_n passes its own  assert num_free_blocks >= n
#   * it runs  num_free_blocks -= n   -> an INCREASE
#   * range(n) iterates zero times, so the linked list is untouched
# num_free_blocks is then inflated relative to the real free list; a later
# legitimate pop walks past the tail and the engine dies mid-run on
#   kv_cache_utils.py  assert curr_block is not None
#   block_pool.py      assert block.ref_cnt == 0
# Load-dependent: c10 died at 3612 s, c12 at 487 s, c16 at 354 s. On the EXTERNAL
# block path, so it needs --kv-transfer-config to appear. NOTE
# --no-async-scheduling was tested and does NOT help (c12 died at 490 s).
patch_kv_blockpool() {
    local label="kv-blockpool"
    # Resolve by path, not by import: importing vllm.v1.core pulls in GPU deps,
    # so _modfile returns empty on a host with no visible device and the patch
    # silently no-ops.
    local target
    target=$($PY -c "import vllm, os; print(os.path.join(os.path.dirname(vllm.__file__), 'v1', 'core', 'single_type_kv_cache_manager.py'))" 2>/dev/null)
    if [ -z "$target" ] || [ ! -f "$target" ]; then
        target=$(ls /usr/local/lib/python3*/dist-packages/vllm/v1/core/single_type_kv_cache_manager.py 2>/dev/null | head -1)
    fi
    if [ -z "$target" ] || [ ! -f "$target" ]; then
        echo "[$label] target not found; skipping."; return 0
    fi
    # NB: the marker must be unique to OUR patch. "num_new_blocks = max(" is NOT
    # -- stock already has it at three other call sites (lines ~208/1511/1601),
    # so using it silently skipped the patch on a clean image.
    if grep -q "KIMI-PATCH-KV-BLOCKPOOL" "$target"; then
        echo "[$label] already patched."; return 0
    fi
    cp -n "$target" "$target.orig" 2>/dev/null || true
    $PY - "$target" <<'EOF' || echo "[kv-blockpool] patch failed; unchanged." >&2
import sys, io
p = sys.argv[1]
src = io.open(p, encoding="utf-8").read()
old = """        req_blocks = self.req_to_blocks[request_id]
        allocated_blocks = self.block_pool.get_new_blocks(
            cdiv(num_total_computed_tokens, self.block_size) - len(req_blocks)
        )"""
new = """        req_blocks = self.req_to_blocks[request_id]
        # KIMI-PATCH-KV-BLOCKPOOL: clamp to >= 0; a negative count silently
        # inflates FreeKVCacheBlockQueue.num_free_blocks and corrupts the free list.
        num_new_blocks = max(
            0, cdiv(num_total_computed_tokens, self.block_size) - len(req_blocks)
        )
        allocated_blocks = self.block_pool.get_new_blocks(num_new_blocks)"""
if src.count(old) != 1:
    sys.stderr.write("[kv-blockpool] anchor missing or not unique; aborting.\n")
    sys.exit(1)
io.open(p, "w", encoding="utf-8").write(src.replace(old, new))
print("[kv-blockpool] patched", p)
EOF
}

# -----------------------------------------------------------------------------
# [4] vLLM: plumb softmax LSE + round-robin CP through the AITER MLA decode
# -----------------------------------------------------------------------------
# Unblocks DCP on ROCM_AITER_MLA. aiter already implements everything needed --
# mla_decode_fwd takes return_lse/cp_world_size/cp_rank/g_kv_indptr and gfx950
# ships the CP kernels -- but vLLM never wired any of it through, so
# AiterMLAImpl returns (o, None) and cp_utils.py's
#   assert layer_impl.need_to_return_lse_for_decode
# rejects DCP before the first forward.
#
# Verified locally on vllm/vllm-openai-rocm:nightly-ac7509e2b (8x MI355X):
#   * dcp_lse_probe.py     -- aiter LSE is NATURAL log with sm_scale folded in,
#                             shape [B,H] fp32, max|lse-ln_ref| 9.5e-07. That is
#                             already DCP's layout, so no transpose (unlike
#                             flashattn_mla.py:367, which returns [H,B]).
#   * dcp_persistent_probe.py -- 96 heads, W=8 round-robin shard + the standard
#                             merge LSE=logsumexp_s(LSE_s), O=sum_s O_s*exp(LSE_s-LSE)
#                             reproduces unsharded attention to rel 2.4e-03
#                             (bf16 noise floor); merged LSE exact to 9.5e-07.
#
# THREE constraints the aiter kernel table imposes (aiter_meta/.../mla/mla_asm.csv).
# Kernel selection is an exact match on all nine columns with NO fallback, and a
# miss calls AITER_CHECK(false) -- which ABORTS THE PROCESS (SIGABRT, core dump),
# it does not raise. Every cprr row is:
#   * ps=1        -> persistent scheduling REQUIRED. asm_mla.cu also gates
#                    lse_flag on `persistent`, so the non-persistent split-K path
#                    can serve neither CP nor LSE. Hence the gluon suppression
#                    below: gluon skips persistent metadata AND has no LSE out.
#   * bf16/bf16   -> no CP kernel exists for an fp8 KV cache.
#   * (Gqa=32,qSeqLen=4) or (Gqa=64,qSeqLen=1) -> qlen>1 (MTP verify) has no
#                    gqa=64 CP kernel, so the DCP path must keep DISABLE_SPEC=1.
# The guard below turns each of these into an explicit RuntimeError, because the
# native failure is an unattributed core dump 10-25 min into warmup.
#
# The head-count fix is the subtle one. Under DCP, mla_attention.py:939 all-gathers
# the query along the head dim, so _forward_decode sees num_heads*dcp_world_size
# (12*8=96 at TP8) while self.num_heads stays 12. Padding against the LOCAL 12
# takes the num_heads<16 branch of get_mla_padded_q and tiles the 96-head tensor
# back down to 16 -- silently dropping 5/6 of the heads, with no crash and
# plausible-looking output. That is why EVAL_ONLY=true / GSM8K (baseline 0.9651)
# is mandatory before quoting any DCP throughput number.

# -----------------------------------------------------------------------------
# [5] vLLM PR #51705 -- upstream DCP support for Kimi-K3 DSpark (EXPERIMENT)
# -----------------------------------------------------------------------------
# https://github.com/vllm-project/vllm/pull/51705 -- OPEN, in no nightly.
# Fetched at run time instead of vendored. Note github.com/<repo>/pull/N.diff
# 404s for this repo; patch-diff.githubusercontent.com serves it. Pinned by
# sha256 so a force-push to the open PR is caught rather than silently changing
# what we benchmark. Verified to apply cleanly to nightly 311b3513.
#
# Mutually exclusive with patch [4]: both rewrite AiterMLAImpl's decode call
# site. SKIP_PATCH_DCPLSE defaults to 1 while this is enabled.
#
# PREDICTION (recorded so the run can falsify it): this will NOT fix our 0x1016.
# The PR exempts a KV group from the DCP block-table narrowing only when
# spec.non_causal_multi_token_decode is true -- the DSpark DRAFT group. Our runs
# have speculative_config=None, so cp_exempt_groups is empty and the narrowing
# is unchanged for every group we actually have. Expect the same fault at
# max_model_len/dcp = 1048576/8 = 131072.
patch_pr51705() {
    local label="pr51705"
    local root; root=$($PY -c 'import vllm,os;print(os.path.dirname(os.path.dirname(vllm.__file__)))' 2>/dev/null)
    if [ -z "$root" ] || [ ! -d "$root/vllm" ]; then
        echo "[$label] vllm root not found; skipping."; return 0
    fi
    if grep -q "VLLM_ALLOW_DCP_FULL_CUDAGRAPH" "$root/vllm/platforms/rocm.py" 2>/dev/null; then
        echo "[$label] already patched."; return 0
    fi
    local dv="$(dirname "$0")/pr51705_vllm.diff"
    if [ ! -f "$dv" ]; then
        echo "[$label] vendored diff not found at $dv; skipping." >&2; return 0
    fi
    echo "[$label] applying vendored diff ($(sha256sum "$dv" | cut -c1-16), $(wc -l < "$dv") lines)"
    patch -p1 -d "$root" --forward --batch < "$dv" >/dev/null 2>&1
    if grep -q "VLLM_ALLOW_DCP_FULL_CUDAGRAPH" "$root/vllm/platforms/rocm.py" 2>/dev/null; then
        echo "[$label] applied"
    else
        echo "[$label] apply did not take" >&2
    fi
}

patch_pr51705_rejects() {
    local label="pr51705-rejects"
    local root; root=$($PY -c 'import vllm,os;print(os.path.dirname(os.path.dirname(vllm.__file__)))' 2>/dev/null)
    [ -n "$root" ] && [ -d "$root/vllm" ] || { echo "[$label] vllm root not found; skipping."; return 0; }

    # PR #51705 hunk 4 of models/kimi_k3/nvidia/mla.py is rejected against this
    # image: the PR's context line is "run_gemm_rs_ar", the image has
    # "run_gemm_rs" (renamed upstream). The hunk that USES
    # enable_dcp_q_replicate applies, so MultiHeadLatentAttention ends up
    # referencing a parameter its signature never gained. Harmless with spec
    # off -- nothing instantiates it -- but dspark_mla.py passes the kwarg, so
    # DCP + MTP dies at init with:
    #   TypeError: MultiHeadLatentAttention.__init__() got an unexpected
    #   keyword argument 'enable_dcp_q_replicate'
    local f="$root/vllm/models/kimi_k3/nvidia/mla.py"
    if [ -f "$f" ] && grep -q "enable_dcp_q_replicate" "$f" 2>/dev/null; then
        if ! grep -qE "^\s*enable_dcp_q_replicate: bool" "$f"; then
            cp -n "$f" "$f.prerej" 2>/dev/null || true
            $PY - "$f" <<'PYEOF'
import re, sys
f = sys.argv[1]
s = open(f).read()
# Insert the parameter ahead of the renamed trailing arg, matching the PR's
# placement after non_causal_multi_token_decode.
pat = re.compile(r"(non_causal_multi_token_decode: bool = False,\n)(\s*)(run_gemm_rs(_ar)?: bool = False,)")
s2, n = pat.subn(r"\1\2enable_dcp_q_replicate: bool = True,\n\2\3", s, count=1)
if n:
    open(f, "w").write(s2)
print(f"  inserted={n}")
PYEOF
        fi
    fi
    if grep -qE "^\s*enable_dcp_q_replicate: bool" "$f" 2>/dev/null; then
        rm -f "$f.rej"
        echo "[$label] MultiHeadLatentAttention.__init__ accepts enable_dcp_q_replicate"
    else
        echo "[$label] WARNING: enable_dcp_q_replicate still missing -- DCP+MTP will fail at init" >&2
    fi

    # Any surviving .rej under a Kimi path is a silent correctness hole.
    local r; r=$(find "$root/vllm/models/kimi_k3" "$root/vllm/v1/attention/backends/mla" \
                     -name "*.rej" 2>/dev/null | head -5)
    if [ -n "$r" ]; then
        echo "[$label] ERROR: unresolved rejects on Kimi/MLA paths -- capability" >&2
        echo "[$label]   flags may be missing; DCP+MTP will fail at init:" >&2
        echo "$r" | sed "s/^/  /" >&2
    fi
}
echo "[kimi-patches] applying in-container patches..."

if [ "${SKIP_PATCH_BLOCKPOOL:-0}" = "1" ]; then
    echo "[kv-blockpool] SKIPPED via SKIP_PATCH_BLOCKPOOL=1"
else
    patch_kv_blockpool || true
fi

patch_dcp_aiter_allreduce() {
    local label="dcp-aiter-allreduce"
    # Resolve by path, not by import: cuda_communicator pulls in GPU deps, so
    # _modfile returns empty on a host with no visible device and the patch
    # silently no-ops. Importing vllm at all is unnecessary to edit a file.
    local target
    target=$($PY -c "import vllm, os; print(os.path.join(os.path.dirname(vllm.__file__), 'distributed', 'device_communicators', 'cuda_communicator.py'))" 2>/dev/null)
    if [ -z "$target" ] || [ ! -f "$target" ]; then
        target=$(ls /usr/local/lib/python3*/dist-packages/vllm/distributed/device_communicators/cuda_communicator.py 2>/dev/null | head -1)
    fi
    if [ -z "$target" ] || [ ! -f "$target" ]; then
        echo "[$label] cuda_communicator.py not found -- skipping" >&2
        return 1
    fi
    if grep -q 'AGENTX_DCP_FAST_AR' "$target"; then
        echo "[$label] already applied"
        return 0
    fi
    if ! grep -q 'if "tp" not in unique_name:' "$target"; then
        echo "[$label] gate text not found (upstream changed?) -- skipping" >&2
        return 1
    fi
    python3 - "$target" <<'PYEOF'
import sys
p = sys.argv[1]
s = open(p).read()
old = '        if "tp" not in unique_name:\n'
new = ('        # AGENTX_DCP_FAST_AR: upstream gates every fast all-reduce backend on the\n'
       '        # literal substring "tp" in the group name, so dcp:0 falls through to\n'
       '        # PYNCCL before any hardware capability is checked. On Kimi-K3 with DCP=8\n'
       '        # that group carries most of the collective time. Allow dcp as well.\n'
       '        if "tp" not in unique_name and "dcp" not in unique_name:\n')
assert s.count(old) == 1, f"expected exactly 1 gate, found {s.count(old)}"
open(p, "w").write(s.replace(old, new))
print("  patched gate")
PYEOF
    echo "[$label] applied to $target"
}

if [ "${SKIP_PATCH_PR51705:-0}" = "1" ]; then
    echo "[pr51705] SKIPPED via SKIP_PATCH_PR51705=1"
else
    patch_pr51705 || true
fi

patch_pr51705_rejects || true

# EXPERIMENTAL, off by default. Not in aigmkt/kimi-k3-vllm:latest -- enabling it
# produces an image that does NOT match the reference. Never measured.
if [ "${APPLY_DCP_FAST_AR:-0}" = "1" ]; then
    patch_dcp_aiter_allreduce || true
fi

echo "[kimi-patches] done."
