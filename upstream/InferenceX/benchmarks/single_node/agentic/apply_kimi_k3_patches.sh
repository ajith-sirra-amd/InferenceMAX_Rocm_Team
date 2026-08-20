#!/usr/bin/env bash
# =============================================================================
# Kimi-K3 / MI355X (gfx950) in-container patches — all three in one place.
#
# Everything here patches files inside the running container only
# (site-packages). Nothing outside the container is touched. Each patch is
# idempotent, verifies its own anchor, backs up to <file>.orig, and no-ops if
# the image already ships the fix. A failed anchor aborts that patch cleanly
# rather than corrupting the file, so a future image with different sources
# degrades to "unpatched", never to "broken".
#
#   [1] aiter pybind11 internals mismatch  -> unblocks ROCM_AITER_FA prefill
#   [2] TritonMLA cudagraph support        -> FULL cudagraphs for DSpark (5.52x TPOT)
#   [3] KV block-pool negative-count clamp -> stops the mid-run engine crash
#
# Env:
#   SKIP_KIMI_PATCHES=1   skip everything
#   PYTHON=...            interpreter to use (default python3)
# =============================================================================
set -euo pipefail
PY=${PYTHON:-python3}

if [ "${SKIP_KIMI_PATCHES:-0}" = "1" ]; then
    echo "[kimi-patches] SKIP_KIMI_PATCHES=1, doing nothing."
    exit 0
fi

# Locate an installed module's file, or empty string if unimportable.
_modfile() {
    $PY - "$1" <<'EOF'
import importlib, os, sys
try:
    print(os.path.abspath(importlib.import_module(sys.argv[1]).__file__))
except Exception:
    print("")
EOF
}

# _patch <file> <already-patched-marker> <<'PYEOF' ... old/new python ... PYEOF
# The heredoc body must define OLD and NEW strings.
_patch() {
    local target="$1" marker="$2" label="$3"
    if [ -z "$target" ] || [ ! -f "$target" ]; then
        echo "[$label] target not found; skipping."
        return 0
    fi
    if grep -q "$marker" "$target"; then
        echo "[$label] already patched."
        return 0
    fi
    cp -n "$target" "$target.orig" 2>/dev/null || true
    if $PY - "$target" "$label"; then
        return 0
    else
        echo "[$label] patch failed; left unchanged." >&2
        return 0
    fi
}

# -----------------------------------------------------------------------------
# [1] aiter: JIT modules must use torch's bundled pybind11
# -----------------------------------------------------------------------------
# aiter/jit/utils/cpp_extension.py appends the STANDALONE pybind11 include via
# -I, which outranks the -isystem path carrying torch's bundled copy. The 117
# prebuilt aiter .so are built against torch's (PYBIND11_INTERNALS_VERSION 11);
# the standalone package here is version 12. pybind11 keeps a SEPARATE type
# registry per internals id, so a JIT-built module cannot see aiter_tensor_t
# registered by the prebuilt core and the first call dies during warmup with
#   TypeError: fmha_fwd_bf16_opus_fwd(): incompatible function arguments
# even though arity and types match exactly.
patch_aiter_pybind11() {
    local label="aiter-pybind11"
    local target; target=$(_modfile aiter.jit.utils.cpp_extension)
    if [ -z "$target" ] || [ ! -f "$target" ]; then
        echo "[$label] aiter not present; skipping."; return 0
    fi

    # Only act if the two pybind11s actually disagree.
    local need; need=$($PY - <<'EOF'
import os, re
try:
    import torch, pybind11
except Exception:
    print("no"); raise SystemExit
def ver(p):
    f = os.path.join(p, "pybind11", "detail", "internals.h")
    if not os.path.isfile(f): return None
    m = re.search(r"define\s+PYBIND11_INTERNALS_VERSION\s+(\d+)", open(f).read())
    return int(m.group(1)) if m else None
t = ver(os.path.join(os.path.dirname(torch.__file__), "include"))
s = ver(pybind11.get_include())
print("yes" if (t is not None and s is not None and t != s) else "no")
EOF
)
    if [ "$need" != "yes" ]; then
        echo "[$label] pybind11 internals already agree; nothing to do."; return 0
    fi

    if grep -q "_use_torch_pybind11" "$target"; then
        echo "[$label] already patched."
    else
        cp -n "$target" "$target.orig" 2>/dev/null || true
        $PY - "$target" <<'EOF' || echo "[aiter-pybind11] patch failed; unchanged." >&2
import sys, io
p = sys.argv[1]
src = io.open(p, encoding="utf-8").read()
old = "        extra_include_paths.append(pybind11.get_include())\n"
new = (
    "        # PATCHED: prefer torch's bundled pybind11 so JIT modules land in the\n"
    "        # same pybind11 type registry as the prebuilt .so files.\n"
    "        _use_torch_pybind11 = False\n"
    "        if not torch_exclude:\n"
    "            _use_torch_pybind11 = os.path.isdir(\n"
    "                os.path.join(TORCH_INCLUDE_ROOT, \"pybind11\")\n"
    "            )\n"
    "        if not _use_torch_pybind11:\n"
    "            extra_include_paths.append(pybind11.get_include())\n"
)
if src.count(old) != 1:
    sys.stderr.write("[aiter-pybind11] anchor missing or not unique; aborting.\n")
    sys.exit(1)
io.open(p, "w", encoding="utf-8").write(src.replace(old, new))
print("[aiter-pybind11] patched", p)
EOF
    fi

    # Drop JIT artifacts built against the wrong pybind11 so they rebuild.
    # aiter honours AITER_JIT_DIR and falls back to ~/.aiter when dist-packages
    # is read-only, so ask aiter rather than deriving the path from $target.
    local jitdir
    jitdir=$($PY -c 'from aiter.jit.core import get_user_jit_dir; print(get_user_jit_dir())' 2>/dev/null || true)
    [ -n "$jitdir" ] && [ -d "$jitdir" ] || jitdir=$(dirname "$(dirname "$target")")
    shopt -s nullglob
    for so in "$jitdir"/*.so; do
        if grep -qa "__pybind11_internals_v12" "$so" 2>/dev/null; then
            rm -f "$so"; rm -rf "$jitdir/build/$(basename "${so%.so}")"
            echo "[$label] removed stale v12 module: $(basename "$so")"
        fi
    done
    shopt -u nullglob
}

# -----------------------------------------------------------------------------
# [2] vLLM: let DSpark spec-decode keep FULL cudagraphs
# -----------------------------------------------------------------------------
# TritonMLAMetadataBuilder._cudagraph_support = UNIFORM_SINGLE_TOKEN_DECODE caps
# min_cg_support below UNIFORM_BATCH, so config/compilation.py downgrades
# FULL_AND_PIECEWISE -> PIECEWISE under spec-decode. dflash/speculator.py then
# gives the DSpark drafter CUDAGraphMode.NONE -- fully eager -- and logs nothing.
# TRITON_MLA cannot be swapped out: it is the only ROCm MLA backend with
# supports_non_causal_multi_token_decode=True, which DSpark requires
# (ROCM_AITER_MLA fails with "non-causal attention not supported").
# The builder already calls _init_reorder_batch_threshold(1,
# supports_spec_as_decode=True) "so full-cudagraph capture admits it", so
# UNIFORM_BATCH is the self-consistent value.
# MEASURED 8x MI355X single stream, 600-token gens:
#   before 14.05 tok/s ITL 71.16 ms  ->  after 77.65 tok/s ITL 12.88 ms  (5.52x)
patch_triton_mla_cudagraph() {
    local label="triton-mla-cudagraph"
    local target; target=$(_modfile vllm.v1.attention.backends.mla.triton_mla)
    if [ -z "$target" ] || [ ! -f "$target" ]; then
        echo "[$label] target not found; skipping."; return 0
    fi
    if grep -q "AttentionCGSupport.UNIFORM_BATCH" "$target"; then
        echo "[$label] already patched."; return 0
    fi
    cp -n "$target" "$target.orig" 2>/dev/null || true
    $PY - "$target" <<'EOF' || echo "[triton-mla-cudagraph] patch failed; unchanged." >&2
import sys, io
p = sys.argv[1]
src = io.open(p, encoding="utf-8").read()
old = """    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )"""
new = """    # PATCHED: UNIFORM_SINGLE_TOKEN_DECODE forced a PIECEWISE downgrade under
    # spec-decode, which silently made the DSpark drafter fully eager.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH"""
if src.count(old) != 1:
    sys.stderr.write("[triton-mla-cudagraph] anchor missing or not unique; aborting.\n")
    sys.exit(1)
io.open(p, "w", encoding="utf-8").write(src.replace(old, new))
print("[triton-mla-cudagraph] patched", p)
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
    local target; target=$(_modfile vllm.v1.core.single_type_kv_cache_manager)
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
patch_dcp_lse() {
    local label="dcp-lse"
    local f_ops; f_ops=$(_modfile vllm._aiter_ops)
    local f_mla; f_mla=$(_modfile vllm.v1.attention.backends.mla.rocm_aiter_mla)
    if [ -z "$f_ops" ] || [ ! -f "$f_ops" ] || [ -z "$f_mla" ] || [ ! -f "$f_mla" ]; then
        echo "[$label] target not found; skipping."; return 0
    fi
    if grep -q "KIMI-PATCH-DCP-LSE" "$f_ops" && grep -q "KIMI-PATCH-DCP-LSE" "$f_mla"; then
        echo "[$label] already patched."; return 0
    fi
    cp -n "$f_ops" "$f_ops.orig" 2>/dev/null || true
    cp -n "$f_mla" "$f_mla.orig" 2>/dev/null || true
    # Two files, so every anchor is verified BEFORE anything is written -- a
    # half-applied patch across _aiter_ops and rocm_aiter_mla would be a
    # signature mismatch at the first decode.
    $PY - "$f_ops" "$f_mla" <<'EOF' || echo "[dcp-lse] patch failed; unchanged." >&2
import sys, io

p_ops, p_mla = sys.argv[1], sys.argv[2]
src_ops = io.open(p_ops, encoding="utf-8").read()
src_mla = io.open(p_mla, encoding="utf-8").read()

edits = []  # (which, old, new, tag)

# --- [4a] _rocm_aiter_mla_decode_fwd_impl: accept + forward the CP/LSE args ---
# Anchored on the `from aiter.mla import` line, which is unique to the real impl
# (the _fake below has an identical parameter list ending in `) -> None: pass`).
edits.append(("ops", """    reduce_partial_map: torch.Tensor | None = None,
) -> None:
    from aiter.mla import mla_decode_fwd
""", """    reduce_partial_map: torch.Tensor | None = None,
    # KIMI-PATCH-DCP-LSE: decode-context-parallel + softmax LSE passthrough.
    lse: torch.Tensor | None = None,
    cp_world_size: int = 1,
    cp_rank: int = 0,
    g_kv_indptr: torch.Tensor | None = None,
) -> None:
    from aiter.mla import mla_decode_fwd
""", "4a impl signature"))

edits.append(("ops", """    mla_decode_fwd(
        q,
        kv_buffer.view(-1, 1, 1, q.shape[-1]),
        o,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        max_seqlen_qo,
        **kwargs,
    )""", """    # KIMI-PATCH-DCP-LSE: ask aiter for the LSE and, under DCP, tell it the local
    # KV is a round-robin shard (global pos p -> rank p % W) so the kernel can
    # rebuild global positions as g(j) = j*W + r for the causal mask.
    if lse is not None:
        kwargs["return_lse"] = True
    if cp_world_size > 1:
        kwargs["cp_world_size"] = cp_world_size
        kwargs["cp_rank"] = cp_rank
        kwargs["g_kv_indptr"] = g_kv_indptr

    ret = mla_decode_fwd(
        q,
        kv_buffer.view(-1, 1, 1, q.shape[-1]),
        o,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        max_seqlen_qo,
        **kwargs,
    )
    if lse is not None:
        # aiter returns (splitData, lse) with lse [B,H] float32 -- already DCP's
        # layout. This op mutates its outputs rather than returning them, so the
        # LSE has to be copied into the caller's buffer.
        assert ret is not None, "return_lse=True but aiter returned None"
        lse.copy_(ret[-1].reshape(lse.shape))""", "4a impl body"))

# --- [4b] the fake impl must mirror the signature or torch.compile diverges ---
edits.append(("ops", """    reduce_partial_map: torch.Tensor | None = None,
) -> None:
    pass""", """    reduce_partial_map: torch.Tensor | None = None,
    # KIMI-PATCH-DCP-LSE: mirror the real impl's signature.
    lse: torch.Tensor | None = None,
    cp_world_size: int = 1,
    cp_rank: int = 0,
    g_kv_indptr: torch.Tensor | None = None,
) -> None:
    pass""", "4b fake signature"))

# --- [4c] registration: lse is an output buffer, so declare it as mutated ---
edits.append(("ops", """                op_name="rocm_aiter_mla_decode_fwd",
                op_func=_rocm_aiter_mla_decode_fwd_impl,
                mutates_args=["o"],""", """                op_name="rocm_aiter_mla_decode_fwd",
                op_func=_rocm_aiter_mla_decode_fwd_impl,
                # KIMI-PATCH-DCP-LSE: "lse" is written in place like "o".
                mutates_args=["o", "lse"],""", "4c registration"))

# --- [4d] the rocm_aiter_ops.mla_decode_fwd wrapper forwards the new args ---
edits.append(("ops", """        reduce_partial_map: torch.Tensor | None = None,
    ):
        torch.ops.vllm.rocm_aiter_mla_decode_fwd(""", """        reduce_partial_map: torch.Tensor | None = None,
        # KIMI-PATCH-DCP-LSE
        lse: torch.Tensor | None = None,
        cp_world_size: int = 1,
        cp_rank: int = 0,
        g_kv_indptr: torch.Tensor | None = None,
    ):
        torch.ops.vllm.rocm_aiter_mla_decode_fwd(""", "4d wrapper signature"))

edits.append(("ops", """            reduce_final_map=reduce_final_map,
            reduce_partial_map=reduce_partial_map,
        )

    @staticmethod
    def per_tensor_quant(""", """            reduce_final_map=reduce_final_map,
            reduce_partial_map=reduce_partial_map,
            # KIMI-PATCH-DCP-LSE
            lse=lse,
            cp_world_size=cp_world_size,
            cp_rank=cp_rank,
            g_kv_indptr=g_kv_indptr,
        )

    @staticmethod
    def per_tensor_quant(""", "4d wrapper call"))

# --- [4d2] the persistent SCHEDULE must be sized for the gathered head count ---
# Caught by comparing against vllm-project/vllm#52248 (closed as overlapping
# with #51705). Patching only the impl's forward path leaves the builder sizing
# get_mla_metadata_info_v1 / get_mla_metadata_v1 for max(16, 12) = 16 heads while
# the kernel is actually invoked with 96 -- a work/reduce schedule built for the
# wrong shape. Prime suspect for the local GSM8K stall at request 1318/1319.
# Read the DCP size from vllm_config rather than self.* so this cannot depend on
# __init__ ordering inside the builder.
edits.append(("mla", """        # For num_attention_heads < 16 (e.g. kimi-k2.5 head=8 with TP8),
        # make sure get_mla_metadata_info_v1 / get_mla_metadata_v1 are consistent
        # with the actual tensor shape passed to mla_decode_fwd.
        self._num_attention_heads = max(16, self.num_heads)""",
"""        # For num_attention_heads < 16 (e.g. kimi-k2.5 head=8 with TP8),
        # make sure get_mla_metadata_info_v1 / get_mla_metadata_v1 are consistent
        # with the actual tensor shape passed to mla_decode_fwd.
        # KIMI-PATCH-DCP-LSE: under DCP the query is all-gathered along the head
        # dim before decode, so the schedule must be built for the GATHERED count
        # (12*8=96 at TP8/DCP8), not the local TP shard. Identity when DCP is off.
        _dcp_size = vllm_config.parallel_config.decode_context_parallel_size
        self._decode_num_heads = self.num_heads * _dcp_size
        self._num_attention_heads = max(16, self._decode_num_heads)""",
"4d2 builder head count"))

# The persistent-metadata gate must ask the same question of the same head
# count, or the builder skips the schedule that the asm decode then requires.
edits.append(("mla", """                self.num_heads >= AiterMLAHelper._AITER_MIN_MLA_HEADS
                or max_qo_len <= AiterMLAHelper._ASM_PADDED_MAX_PS_QLEN""",
"""                self._decode_num_heads >= AiterMLAHelper._AITER_MIN_MLA_HEADS
                or max_qo_len <= AiterMLAHelper._ASM_PADDED_MAX_PS_QLEN""",
"4d3 persistent gate head count"))

# --- [4e] gluon has no LSE output and skips persistent metadata -> not DCP-safe ---
edits.append(("mla", """    @staticmethod
    def use_gluon_decode(num_heads: int, max_qo_len: int, kv_cache_dtype: str) -> bool:""",
"""    @staticmethod
    def _dcp_active() -> bool:
        # KIMI-PATCH-DCP-LSE: gluon returns (o, None) and, because the builder's
        # use_persistent_metadata excludes it, also skips the persistent schedule
        # that every cprr kernel requires. Under DCP force the ASM path. Read the
        # group directly so the builder and the impl cannot disagree.
        try:
            from vllm.distributed.parallel_state import get_dcp_group

            return get_dcp_group().world_size > 1
        except Exception:
            return False

    @staticmethod
    def use_gluon_decode(num_heads: int, max_qo_len: int, kv_cache_dtype: str) -> bool:
        if AiterMLAHelper._dcp_active():  # KIMI-PATCH-DCP-LSE
            return False""", "4e gluon decode gate"))

# --- [4f] AiterMLAImpl advertises LSE; the base class derives the rest ---
# backend.py:940 does need_to_return_lse_for_decode = dcp_world_size > 1 and
# can_return_lse_for_decode, so this single flag is the whole opt-in.
edits.append(("mla", """class AiterMLAImpl(MLACommonImpl[AiterMLAMetadata]):""",
"""class AiterMLAImpl(MLACommonImpl[AiterMLAMetadata]):
    # KIMI-PATCH-DCP-LSE: backend.py derives need_to_return_lse_for_decode from
    # this and dcp_world_size > 1, which is what clears cp_utils.py's DCP assert.
    can_return_lse_for_decode: bool = True
""", "4f can_return_lse flag"))

# --- [4g] effective head count under the DCP query all-gather ---
edits.append(("mla", """        mla_padded_q = AiterMLAHelper.get_mla_padded_q(self.num_heads, q)
        mla_num_heads = AiterMLAHelper.get_actual_mla_num_heads(self.num_heads)""",
"""        # KIMI-PATCH-DCP-LSE: mla_attention.py all-gathers the query along the
        # head dim under DCP, so this rank sees num_heads*dcp_world_size heads
        # (12*8=96 at TP8) while self.num_heads is still the local 12. Padding
        # against 12 would take the <16 branch and tile 96 heads back down to
        # 16, silently dropping 5/6 of them.
        _eff_num_heads = self.num_heads * self.dcp_world_size
        mla_padded_q = AiterMLAHelper.get_mla_padded_q(_eff_num_heads, q)
        mla_num_heads = AiterMLAHelper.get_actual_mla_num_heads(_eff_num_heads)""",
"4g effective head count"))

# --- [4h] allocate the LSE, pass the CP args, return (o, lse) ---
edits.append(("mla", """        rocm_aiter_ops.mla_decode_fwd(
            mla_padded_q,
            kv_buffer,
            o,
            self.scale,
            decode.qo_indptr,
            decode.max_qo_len,
            decode.paged_kv_indptr,
            decode.paged_kv_indices,
            decode.paged_kv_last_page_len,
            **mla_kwargs,
        )

        return AiterMLAHelper.get_mla_unpadded_o(self.num_heads, o), None""",
"""        # KIMI-PATCH-DCP-LSE: every cprr row in aiter's mla_asm.csv is ps=1 and
        # bf16/bf16, and a table miss is AITER_CHECK(false) -> SIGABRT rather than
        # an exception. Fail loudly here instead of core-dumping in warmup.
        lse = None
        if self.need_to_return_lse_for_decode:
            if not decode.has_persistent_metadata:
                raise RuntimeError(
                    "DCP on ROCM_AITER_MLA requires the persistent ASM decode "
                    "schedule (aiter serves CP round-robin only from ps=1 "
                    "kernels), but has_persistent_metadata is False."
                )
            # NB: _kv_cache_dtype_str lives on the metadata BUILDER, not on the
            # impl -- reaching for it here is an AttributeError at first decode.
            if is_quantized_kv_cache(self.kv_cache_dtype):
                raise RuntimeError(
                    "DCP on ROCM_AITER_MLA requires a bf16 KV cache; aiter ships "
                    f"no CP kernel for kv_cache_dtype={self.kv_cache_dtype}."
                )
            if decode.max_qo_len != 1:
                raise RuntimeError(
                    "DCP on ROCM_AITER_MLA requires qlen==1 (aiter has no "
                    f"gqa=64 CP kernel past qseqlen 1), got {decode.max_qo_len}. "
                    "Keep DISABLE_SPEC=1 on the DCP path."
                )
            lse = torch.empty(
                (o.shape[0], mla_num_heads), dtype=torch.float32, device=o.device
            )
            # aiter wants a cumulative GLOBAL kv indptr; dcp_tot_seq_lens holds
            # per-request global lengths (flashattn takes those directly as
            # cp_tot_seqused_k, so this conversion is aiter-specific).
            tot = decode.dcp_tot_seq_lens
            assert tot is not None, "DCP decode without dcp_tot_seq_lens"
            g_kv_indptr = torch.zeros(
                tot.numel() + 1, dtype=torch.int32, device=tot.device
            )
            torch.cumsum(tot, 0, out=g_kv_indptr[1:])
            mla_kwargs.update(
                lse=lse,
                cp_world_size=self.dcp_world_size,
                cp_rank=self.dcp_rank,
                g_kv_indptr=g_kv_indptr,
            )

        rocm_aiter_ops.mla_decode_fwd(
            mla_padded_q,
            kv_buffer,
            o,
            self.scale,
            decode.qo_indptr,
            decode.max_qo_len,
            decode.paged_kv_indptr,
            decode.paged_kv_indices,
            decode.paged_kv_last_page_len,
            **mla_kwargs,
        )

        if lse is None:
            return AiterMLAHelper.get_mla_unpadded_o(_eff_num_heads, o), None
        # dcp_manager.combine wants [B, H] matching the all-gathered head count.
        return (
            AiterMLAHelper.get_mla_unpadded_o(_eff_num_heads, o),
            AiterMLAHelper.get_mla_unpadded_o(_eff_num_heads, lse.unsqueeze(-1)).squeeze(-1),
        )""", "4h lse alloc + cp args + return"))

# Verify every anchor is present exactly once BEFORE touching either file.
srcs = {"ops": src_ops, "mla": src_mla}
for which, old, _new, tag in edits:
    n = srcs[which].count(old)
    if n != 1:
        sys.stderr.write(f"[dcp-lse] anchor '{tag}' found {n} times (want 1); aborting.\n")
        sys.exit(1)

for which, old, new, _tag in edits:
    srcs[which] = srcs[which].replace(old, new, 1)

# is_quantized_kv_cache is referenced by the guard; it is already imported in
# rocm_aiter_mla.py (used by use_persistent_metadata), but verify rather than assume.
if "is_quantized_kv_cache" not in srcs["mla"].split("class AiterMLAHelper")[0]:
    sys.stderr.write("[dcp-lse] is_quantized_kv_cache not imported; aborting.\n")
    sys.exit(1)

io.open(p_ops, "w", encoding="utf-8").write(srcs["ops"])
io.open(p_mla, "w", encoding="utf-8").write(srcs["mla"])
print("[dcp-lse] patched", p_ops)
print("[dcp-lse] patched", p_mla)
EOF
}

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
PR51705_URL="https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/51705.diff"
PR51705_SHA="78baf20117a2fdc128d1dea1d3b532e148317f448ed238683305aae91aff5126"  # re-pinned 2026-08-20: PR updated upstream; adds causal verify window, non_causal_multi_token_decode, fp8 decode routing, MTP split
patch_pr51705() {
    local label="pr51705"
    local root; root=$($PY -c 'import vllm,os;print(os.path.dirname(os.path.dirname(vllm.__file__)))' 2>/dev/null)
    if [ -z "$root" ] || [ ! -d "$root/vllm" ]; then
        echo "[$label] vllm root not found; skipping."; return 0
    fi
    if grep -q "cp_exempt_groups" "$root/vllm/v1/worker/gpu/block_table.py" 2>/dev/null; then
        echo "[$label] already patched."; return 0
    fi
    local d; d=$(mktemp)
    if ! curl -fsSL "$PR51705_URL" -o "$d"; then
        echo "[$label] download failed; leaving vllm unpatched." >&2; rm -f "$d"; return 0
    fi
    local got; got=$(sha256sum "$d" | cut -d" " -f1)
    if [ "$got" != "$PR51705_SHA" ]; then
        echo "[$label] sha256 mismatch -- PR updated since review; refusing to apply." >&2
        echo "[$label]   expected $PR51705_SHA" >&2
        echo "[$label]   got      $got" >&2
        rm -f "$d"; return 0
    fi
    # The PR also touches tests/, which do not exist under site-packages, so an
    # unfiltered apply always fails. Keep only vllm/ files.
    local dv; dv=$(mktemp)
    "$PY" - "$d" "$dv" <<'PYEOF'
import re, sys
src = open(sys.argv[1]).read()
blocks = re.split(r"(?m)^(?=diff --git )", src)
keep = [b for b in blocks
        if b.startswith("diff --git") and re.search(r"^\+\+\+ b/vllm/", b, re.M)]
open(sys.argv[2], "w").write("".join(keep))
print(f"[pr51705] filtered to {len(keep)} vllm/ files")
PYEOF
    # Hunk 7 of rocm_aiter_mla.py (the DCP gathered-head sizing) does not apply to
    # nightlies that refactored the line into AiterMLAHelper.get_actual_mla_num_heads.
    # Let the other 23 land and fix that one in patch [8] below.
    if ! patch -p1 -d "$root" --dry-run --forward < "$dv" >/dev/null 2>&1; then
        echo "[$label] dry-run has rejects; applying what fits" >&2
        patch -p1 -d "$root" --dry-run --forward < "$dv" 2>&1 | grep -iE "fail" | head -5 >&2
    fi
    patch -p1 -d "$root" --forward --backup --suffix=.pr51705.orig < "$dv" >/dev/null 2>&1
    if grep -q "cp_exempt_groups" "$root/vllm/v1/worker/gpu/block_table.py" 2>/dev/null; then
        echo "[$label] applied PR #51705 (sha ${PR51705_SHA:0:16})"
    else
        echo "[$label] apply did not take" >&2
    fi
    rm -f "$d" "$dv"
}

# -----------------------------------------------------------------------------
# [6] vLLM: size DCP block tables for the FULL sequence, not the local shard
# -----------------------------------------------------------------------------
# THE 0x1016 FIX. initialize_kv_cache sizes each group's block table to
#   cdiv(block_table_max_model_len, block_size * dcp_size)
# i.e. max_model_len/dcp tokens' worth of ROWS. block_table_max_model_len is the
# undivided max_model_len (model_runner.py:492), so at 1M/DCP8 the table can
# only index 131072 tokens. Chunked prefill then walks a longer request past the
# end of the table -> OOB read -> HSA_STATUS_ERROR_EXCEPTION 0x1016.
#
# Measured across three runs; the boundary tracks max_model_len/dcp exactly:
#   DCP8 bf16  budget 131072  last chunk fit 119040  faulted crossing 134400
#   DCP4 bf16  budget 262144  last chunk fit 254976  faulted crossing 262656
#   DCP8 fp8   budget 131072  last chunk fit 119808  faulted crossing 135168
# Halving DCP doubled both budget and fault point; fp8 vs bf16 changed nothing,
# so this is block COUNT, not bytes. All three were pure prefill
# (num_output_tokens=0), which is why swapping decode backends never helped.
#
# PR #51705 addresses the same narrowing but only exempts groups with
# spec.non_causal_multi_token_decode (the DSpark draft group). With
# speculative_config=None there is no such group, cp_exempt_groups is empty and
# nothing changes -- confirmed by run 32042030173, which failed at 135168 with
# the PR applied.
#
# Row count and slot mapping are deliberately decoupled here. Sharded groups
# still need CP_SIZE=dcp_size for correct position->slot math, so this touches
# ONLY the row count and leaves group_cp_size / cp_exempt_groups alone.
# Over-allocating rows costs int32 indices -- at 1M/64B blocks x 128 reqs that
# is ~8 MB per group -- while under-allocating is the bug above.
#
# Anchors on either the stock line or PR #51705's rewritten one, so it applies
# with or without patch [5].
patch_dcp_blocktable() {
    local label="dcp-blocktable"
    local target; target=$(_modfile vllm.v1.worker.gpu.model_runner)
    if [ -z "$target" ] || [ ! -f "$target" ]; then
        echo "[$label] target not found; skipping."; return 0
    fi
    if grep -q "KIMI-PATCH-DCP-BLOCKTABLE" "$target"; then
        echo "[$label] already patched."; return 0
    fi
    cp -n "$target" "$target.orig" 2>/dev/null || true
    $PY - "$target" <<'EOF' || echo "[dcp-blocktable] patch failed; unchanged." >&2
import sys, io
p = sys.argv[1]
src = io.open(p, encoding="utf-8").read()
NEW = """            # KIMI-PATCH-DCP-BLOCKTABLE: size the block table for the FULL
            # sequence. Dividing the ROW COUNT by dcp_size caps indexing at
            # max_model_len/dcp (131072 at 1M/DCP8); chunked prefill then reads
            # past the end of the table -> OOB -> HSA 0x1016. Slot mapping is
            # untouched: sharded groups still need CP_SIZE=dcp_size.
            max_num_blocks = cdiv(block_table_max_model_len, spec.block_size)"""
# PR #51705 form first, then stock -- they are mutually exclusive.
CANDS = [
"""            max_num_blocks = cdiv(
                block_table_max_model_len, spec.block_size * group_cp_size
            )""",
"""            max_num_blocks = cdiv(
                block_table_max_model_len, spec.block_size * self.dcp_size
            )""",
]
for old in CANDS:
    if src.count(old) == 1:
        io.open(p, "w", encoding="utf-8").write(src.replace(old, NEW, 1))
        print("[dcp-blocktable] patched", p)
        break
else:
    sys.stderr.write("[dcp-blocktable] no unique anchor found; aborting.\n")
    sys.exit(1)
EOF
}

# Per-patch switches, so a single patch can be isolated without disabling the
# others. Note patch [1] is load-bearing: without it ROCM_AITER_FA prefill dies
# at warmup with the fmha_fwd_bf16_opus TypeError, so skipping it does not give
# a clean baseline -- it gives a different crash.
#   SKIP_PATCH_AITER=1      skip [1] aiter pybind11
#   SKIP_PATCH_CUDAGRAPH=1  skip [2] TritonMLA UNIFORM_BATCH   <- the HIP-999 suspect
#   SKIP_PATCH_BLOCKPOOL=1  skip [3] KV block-pool clamp
#   SKIP_PATCH_DCPLSE=1     skip [4] DCP/LSE plumbing for ROCM_AITER_MLA
echo "[kimi-patches] applying in-container patches..."
if [ "${SKIP_PATCH_AITER:-0}" = "1" ]; then
    echo "[aiter-pybind11] SKIPPED via SKIP_PATCH_AITER=1"
else
    patch_aiter_pybind11 || true
fi
if [ "${SKIP_PATCH_CUDAGRAPH:-0}" = "1" ]; then
    echo "[triton-mla-cudagraph] SKIPPED via SKIP_PATCH_CUDAGRAPH=1"
else
    patch_triton_mla_cudagraph || true
fi
if [ "${SKIP_PATCH_BLOCKPOOL:-0}" = "1" ]; then
    echo "[kv-blockpool] SKIPPED via SKIP_PATCH_BLOCKPOOL=1"
else
    patch_kv_blockpool || true
fi
# [4] and [5] are mutually exclusive -- both rewrite AiterMLAImpl's decode call
# site. PR51705 wins by default while we evaluate upstream's design; set
# SKIP_PATCH_PR51705=1 SKIP_PATCH_DCPLSE=0 to go back to our own patch.
# DCP arm is parked: T5/T7 showed DCP costs ~6.5x throughput on this
# prefill-dominated trace and cudagraphs cannot recover it. For the non-DCP
# baseline every DCP patch is off, because [5] also rewrites speculative.py and
# kimi_gdn_linear_attn.py, which are live even at dcp_size=1 -- and T9 died on
# an RCCL collective timeout with spec decoding active.
# To go back to the DCP arm: SKIP_PATCH_PR51705=0 (or SKIP_PATCH_DCPLSE=0 for
# our own patch) plus SKIP_PATCH_BLOCKTABLE=0, and raise conc to >= 64.
SKIP_PATCH_PR51705="${SKIP_PATCH_PR51705:-0}"
SKIP_PATCH_DCPLSE="${SKIP_PATCH_DCPLSE:-1}"
if [ "${SKIP_PATCH_DCPLSE:-0}" = "1" ]; then
    echo "[dcp-lse] SKIPPED via SKIP_PATCH_DCPLSE=1"
else
    patch_dcp_lse || true
fi
if [ "$SKIP_PATCH_PR51705" = "1" ]; then
    echo "[pr51705] SKIPPED via SKIP_PATCH_PR51705=1"
else
    patch_pr51705 || true
fi
# [6] applies on top of [5] (or on stock) and is the actual 0x1016 fix.
if [ "${SKIP_PATCH_BLOCKTABLE:-0}" = "1" ]; then
    echo "[dcp-blocktable] SKIPPED via SKIP_PATCH_BLOCKTABLE=1"
else
    patch_dcp_blocktable || true
fi
# -----------------------------------------------------------------------------
# [7] direct DCP a2a combine -- load the out-of-tree ROCm kernel (T31)
#
# vLLM ships the Python call site at vllm/v1/attention/ops/dcp_utils.py:262 but
# compiles csrc/libtorch_stable/attention/dcp_utils/*.cu ONLY inside
# if(VLLM_GPU_LANG STREQUAL "CUDA"), so on ROCm torch.ops._C has no
# direct_dcp_* ops at all. That is exactly what killed T27:
#   AttributeError: '_OpNamespace' '_C' object has no attribute
#                   'direct_dcp_a2a_lse_reduce'
#
# Only the a2a combine was portable. The q_gather/kv_gather kernels use
# multimem.st.* PTX under #if __CUDA_ARCH__ >= 900 -- NVIDIA NVLink hardware
# multicast, with no AMD equivalent -- and are deliberately NOT loaded; their
# env stays 0 and they keep using the existing RCCL fallbacks. The a2a kernel
# contains zero multimem; its only PTX is st.global.release.sys.u32 /
# ld.global.acquire.sys.u32, which map exactly onto __hip_atomic_store /
# __hip_atomic_load at __ATOMIC_RELEASE/ACQUIRE + __HIP_MEMORY_SCOPE_SYSTEM,
# preserving the epoch-handshake ordering and scope.
#
# The .so is prebuilt on the host (it needs hipcc and ~1 min) and reaches the
# container through the harness's existing -v /data/hf_hub_cache:/mnt/hf_hub_cache
# mount. Build recipe and the full transform list live beside it in
# port_to_hip.py. Built against torch 2.12.0+git6bbd260 / ROCm 7.2.53211.
#
# It is dlopen'd rather than imported: there is no PyInit_, registration happens
# via STABLE_TORCH_LIBRARY_FRAGMENT(_C, ...) static initialisers. It must load
# AFTER torch (it links libtorch_cpu.so), so the loader is appended to
# dcp_utils.py, which imports torch at its top and is the sole consumer.
patch_dcp_direct_a2a() {
    local so="${DCP_DIRECT_SO:-/mnt/hf_hub_cache/dcp/vllm_dcp_direct_rocm.so}"
    local f
    # vLLM logs INFO banners to stdout on import, so _modfile's output can be
    # multi-line; the path is always the last line.
    f=$(_modfile vllm.v1.attention.ops.dcp_utils | tail -1) || true
    if [ -z "$f" ] || [ ! -f "$f" ]; then
        echo "[dcp-direct] dcp_utils.py not found; skipping"; return 0
    fi
    if [ ! -f "$so" ]; then
        echo "[dcp-direct] $so not present; skipping (DCP falls back to RCCL a2a)"
        return 0
    fi
    if grep -q "DCP-ROCM-DIRECT-A2A" "$f"; then
        echo "[dcp-direct] already applied"
    else
        [ -f "$f.orig" ] || cp "$f" "$f.orig"
        cat >> "$f" <<PYEOF

# DCP-ROCM-DIRECT-A2A (applied by apply_kimi_k3_patches.sh)
import ctypes as _dcp_ctypes, os as _dcp_os, sys as _dcp_sys
_dcp_so = _dcp_os.environ.get(
    "DCP_DIRECT_SO", "/mnt/hf_hub_cache/dcp/vllm_dcp_direct_rocm.so")
if _dcp_os.path.exists(_dcp_so):
    try:
        _dcp_ctypes.CDLL(_dcp_so, mode=_dcp_ctypes.RTLD_GLOBAL)
        print("[dcp-direct] loaded " + _dcp_so, file=_dcp_sys.stderr)
    except OSError as _dcp_err:
        print("[dcp-direct] load failed: %s" % _dcp_err, file=_dcp_sys.stderr)
PYEOF
        echo "[dcp-direct] appended loader to $f"
    fi
    # Prove the op actually resolves; a silent miss would look like a perf null.
    "$PY" - <<'PYEOF' || echo "[dcp-direct] WARNING: op did NOT resolve"
import torch, vllm.v1.attention.ops.dcp_utils  # noqa: F401
print("[dcp-direct] op resolves:", torch.ops._C.direct_dcp_a2a_lse_reduce)
PYEOF
}

if [ "${SKIP_PATCH_DCP_DIRECT:-0}" = "1" ]; then
    echo "[dcp-direct] SKIPPED via SKIP_PATCH_DCP_DIRECT=1"
else
    patch_dcp_direct_a2a || true
fi

# [8] DCP gathered-head sizing for AITER MLA decode.
# PR #51705 hunk 7 wants max(16, num_heads * dcp_world_size); newer nightlies
# refactored that line to AiterMLAHelper.get_actual_mla_num_heads(self.num_heads),
# which rounds up to a multiple of 16 but never multiplies by dcp_world_size.
# DCP gathers every rank's shard before decode (Kimi-K3 TP8/DCP8: 12 -> 96), so
# the metadata must be sized for the gathered count. No-op when DCP is off.
patch_dcp_gathered_heads() {
    local label="dcp-gathered-heads"
    local f; f=$(_modfile vllm.v1.attention.backends.mla.rocm_aiter_mla | tail -1)
    [ -n "$f" ] && [ -f "$f" ] || { echo "[$label] target not found; skipping"; return 0; }
    # Guard on the ASSIGNMENT, not the bare name: PR #51705's other hunks add 14
    # uses of self._decode_num_heads while hunk 7 -- the one that assigns it --
    # is exactly the hunk that fails. Matching the name skips this patch and
    # leaves 14 uses of an attribute that is never set.
    if grep -qE "self\._decode_num_heads *=" "$f"; then echo "[$label] already patched"; return 0; fi
    cp -n "$f" "$f.orig" 2>/dev/null || true
    "$PY" - "$f" <<'PYEOF' || echo "[dcp-gathered-heads] patch failed; unchanged" >&2
import io, sys
p = sys.argv[1]
s = io.open(p, encoding="utf-8").read()
old = ("        self._num_attention_heads = AiterMLAHelper.get_actual_mla_num_heads(\n"
       "            self.num_heads\n"
       "        )\n")
new = ("        self._decode_num_heads = self.num_heads * self.dcp_world_size\n"
       "        self._num_attention_heads = AiterMLAHelper.get_actual_mla_num_heads(\n"
       "            self._decode_num_heads\n"
       "        )\n")
if s.count(old) != 1:
    sys.stderr.write("[dcp-gathered-heads] anchor missing or not unique; aborting\n")
    sys.exit(1)
io.open(p, "w", encoding="utf-8").write(s.replace(old, new))
print("[dcp-gathered-heads] patched", p)
PYEOF
}

if [ "${SKIP_PATCH_GATHERED_HEADS:-0}" = "1" ]; then
    echo "[dcp-gathered-heads] SKIPPED via SKIP_PATCH_GATHERED_HEADS=1"
else
    patch_dcp_gathered_heads || true
fi

echo "[kimi-patches] done."
