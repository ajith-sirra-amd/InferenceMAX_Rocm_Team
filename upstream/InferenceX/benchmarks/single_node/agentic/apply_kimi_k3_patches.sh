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
# https://github.com/vllm-project/vllm/pull/51705 -- OPEN, not merged, so it is
# in no nightly. Applied verbatim (sha256[:16]=258ae29579c9ec26) rather than
# reimplemented, so this is a clean test of upstream's design.
#
# Mutually exclusive with patch [4]: both rewrite AiterMLAImpl's decode call
# site. SKIP_PATCH_DCPLSE defaults to 1 while this is enabled.
#
# PREDICTION (recorded before the run, so the result can falsify it):
# this will NOT fix our 0x1016. The PR exempts a KV cache group from the DCP
# block-table narrowing only when spec.non_causal_multi_token_decode is true --
# the DSpark DRAFT group (model_runner.py: group_cp_size = 1 if ... else dcp).
# Our failing runs had speculative_config=None, i.e. no draft group at all, so
# cp_exempt_groups is empty and the narrowing is unchanged for every group we
# actually have. Expect the same fault at max_model_len/dcp = 1048576/8 = 131072.
# Turning spec on to create that group does not help either: under DCP the PR's
# asm branch raises NotImplementedError for max_qo_len != 1, and its Gluon
# branch -- the one that handles qlen>1 -- self-gates off because aiter v0.1.19
# (pinned by BOTH nightlies in docker/Dockerfile.rocm_base) caps mla_gluon at
# "nhead <= 16 or nhead in (64,128)" while TP8 x DCP8 gathers 96.
# If it DOES survive past 131072, the prediction is wrong and the block-table
# story needs revisiting.
patch_pr51705() {
    local label="pr51705"
    local root; root=$($PY -c 'import vllm,os;print(os.path.dirname(os.path.dirname(vllm.__file__)))' 2>/dev/null)
    if [ -z "$root" ] || [ ! -d "$root/vllm" ]; then
        echo "[$label] vllm root not found; skipping."; return 0
    fi
    if grep -q "cp_exempt_groups" "$root/vllm/v1/worker/gpu/block_table.py" 2>/dev/null; then
        echo "[$label] already patched."; return 0
    fi
    local d="$root/.pr51705.diff"
    cat > "$d" <<'PR51705_DIFF_EOF'
diff --git a/vllm/config/speculative.py b/vllm/config/speculative.py
index 2b93113b7ed3..993a56c4ddb8 100644
--- a/vllm/config/speculative.py
+++ b/vllm/config/speculative.py
@@ -1046,16 +1046,6 @@ def __post_init__(self):
                 if self.method in ("dflash", "dspark"):
                     self.parallel_drafting = True
 
-                if (
-                    self.method == "dspark"
-                    and "K3DSparkModel" in self.draft_model_config.architectures
-                    and self.target_parallel_config.decode_context_parallel_size > 1
-                ):
-                    raise ValueError(
-                        "MLA DSpark does not currently support decode context "
-                        "parallelism; set decode_context_parallel_size=1."
-                    )
-
                 if self.num_speculative_tokens is not None and hasattr(
                     self.draft_model_config.hf_config, "num_lookahead_tokens"
                 ):
diff --git a/vllm/model_executor/layers/attention/mla_attention.py b/vllm/model_executor/layers/attention/mla_attention.py
index 839f4266cd97..0bf975f42b86 100644
--- a/vllm/model_executor/layers/attention/mla_attention.py
+++ b/vllm/model_executor/layers/attention/mla_attention.py
@@ -2111,6 +2111,10 @@ def __init__(
         except AssertionError:
             # DCP might not be initialized in testing
             self.dcp_world_size = 1
+        # Replicated draft groups (non_causal_multi_token_decode) use whole-sequence
+        # metadata and do not participate in the target's DCP merge.
+        if self.non_causal_multi_token_decode:
+            self.dcp_world_size = 1
         self.dcp_local_block_size = parallel_config.cp_kv_cache_interleave_size
         self.dcp_virtual_block_size = self.dcp_local_block_size * self.dcp_world_size
         self.cp_kv_cache_interleave_size = parallel_config.cp_kv_cache_interleave_size
diff --git a/vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py b/vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py
index cf0a17720f46..397bf0b5bf79 100644
--- a/vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py
+++ b/vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py
@@ -593,9 +593,11 @@ def _prefill_conv(
             else:
                 # pure-decode non-spec batch
                 assert non_spec_state_indices_tensor is not None
+                # Under spec decode this can be a strided 1-D view, which the
+                # packed KDA decode kernel rejects.
                 decode_conv_indices = non_spec_state_indices_tensor[
                     : mixed_qkv_ns.size(0)
-                ]
+                ].contiguous()
                 # Sibling beta and, for full-rank gates, output-gate views
                 # remain live, so write the conv output separately.
                 packed_conv_out = torch.empty(
diff --git a/vllm/models/kimi_k3/nvidia/mla.py b/vllm/models/kimi_k3/nvidia/mla.py
index 71e9fa79f04d..1da5ec7c7214 100644
--- a/vllm/models/kimi_k3/nvidia/mla.py
+++ b/vllm/models/kimi_k3/nvidia/mla.py
@@ -340,7 +340,22 @@ def __init__(
             "Kimi-K3 MultiHeadLatentAttention does not support prefill context "
             "parallelism."
         )
-        self.dcp_world_size = parallel_config.decode_context_parallel_size
+        if self.non_causal_multi_token_decode:
+            # The DSpark draft attends over the whole sequence and discards its
+            # decode LSE, so its KV cache group is replicated on every DCP rank
+            # rather than sharded. Such a layer sees the full sequence locally and
+            # takes no part in the target's gather/merge.
+            self.dcp_world_size = 1
+            self.impl.dcp_world_size = 1
+            self.impl.dcp_rank = 0
+            self.impl.need_to_return_lse_for_decode = False
+            if hasattr(self.impl, "_decode_num_heads"):
+                # Backends that size decode for the DCP-gathered query heads
+                # cached that count from the group's dcp_world_size at
+                # construction; this layer just left the DCP group.
+                self.impl._decode_num_heads = self.num_local_heads
+        else:
+            self.dcp_world_size = parallel_config.decode_context_parallel_size
         assert self.dcp_world_size <= 1 or self.rotary_emb is None, (
             "Kimi-K3 MultiHeadLatentAttention does not support RoPE with decode "
             "context parallelism because gathered queries require gathered "
diff --git a/vllm/v1/attention/backends/mla/rocm_aiter_mla.py b/vllm/v1/attention/backends/mla/rocm_aiter_mla.py
index ccbdf728903a..824fe96031b8 100644
--- a/vllm/v1/attention/backends/mla/rocm_aiter_mla.py
+++ b/vllm/v1/attention/backends/mla/rocm_aiter_mla.py
@@ -2,6 +2,8 @@
 # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 
 import functools
+import inspect
+import re
 from dataclasses import dataclass
 from typing import ClassVar, Final
 
@@ -26,6 +28,7 @@
     CommonAttentionMetadata,
     MultipleOf,
 )
+from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
 from vllm.v1.kv_cache_interface import AttentionSpec, is_quantized_kv_cache
 
 logger = init_logger(__name__)
@@ -94,6 +97,26 @@ def _gluon_mla_decode_supported() -> bool:
     return on_gfx950()
 
 
+@functools.lru_cache(maxsize=1)
+def _gluon_mla_max_bh16_heads() -> int:
+    """Largest head count the Gluon MLA bh16 regimes accept.
+
+    ROCm/aiter#4412 tiles the head range into ``cdiv(nhead, 16)`` blocks on the
+    grid, lifting the bound from 16 to 96, which is what TP8 x DCP8 gathers.
+    The bound is read off the wrapper's own assert so that an older aiter keeps
+    the pre-#4412 gating instead of asserting inside the kernel launch.
+    """
+    fallback = AiterMLAHelper._AITER_MIN_MLA_HEADS
+    try:
+        source = inspect.getsource(_get_mla_gluon())
+    except Exception:  # noqa: BLE001
+        return fallback
+    match = re.search(r"requires nhead <= (\d+)", source)
+    if match is None:
+        return fallback
+    return max(fallback, int(match.group(1)))
+
+
 def _aiter_mla_small_head_mode() -> str:
     """Small-head (<16) MLA decode kernel selection.
 
@@ -262,8 +285,29 @@ def __init__(
         vllm_config: VllmConfig,
         device: torch.device,
     ):
+        # Keeping a drafter's verify batch on the decode path under DCP needs a
+        # decode LSE for every uniform query length the runner can produce, and a
+        # causal window taken in global positions. The Gluon flatten gives both
+        # for any length; the asm decode has LSE kernels for only a few lengths
+        # and takes the window per rank. Interleaving the KV shard by more than
+        # one token is unsupported, as in FlashAttnMLA.
+        parallel_config = vllm_config.parallel_config
+        gathered_num_heads = (
+            vllm_config.model_config.get_num_attention_heads(parallel_config)
+            * parallel_config.decode_context_parallel_size
+        )
         super().__init__(
-            kv_cache_spec, layer_names, vllm_config, device, AiterMLAMetadata
+            kv_cache_spec,
+            layer_names,
+            vllm_config,
+            device,
+            AiterMLAMetadata,
+            supports_dcp_with_varlen=(
+                parallel_config.cp_kv_cache_interleave_size == 1
+                and AiterMLAHelper.use_gluon_verify(
+                    gathered_num_heads, 2, vllm_config.cache_config.cache_dtype
+                )
+            ),
         )
 
         self.compilation_config = vllm_config.compilation_config
@@ -308,10 +352,12 @@ def __init__(
 
         from aiter import dtypes, get_mla_metadata_info_v1
 
-        # For num_attention_heads < 16 (e.g. kimi-k2.5 head=8 with TP8),
-        # make sure get_mla_metadata_info_v1 / get_mla_metadata_v1 are consistent
-        # with the actual tensor shape passed to mla_decode_fwd.
-        self._num_attention_heads = max(16, self.num_heads)
+        # DCP gathers every rank's query-head shard before decode. Build the
+        # persistent schedule for the gathered tensor, not the local TP shard.
+        # This mirrors ATOM's persistent_num_heads sizing and is essential for
+        # Kimi-K3 TP8/DCP8 (12 local heads -> 96 gathered heads).
+        self._decode_num_heads = self.num_heads * self.dcp_world_size
+        self._num_attention_heads = max(16, self._decode_num_heads)
         kv_cache_dtype_str = getattr(vllm_config.cache_config, "cache_dtype", "auto")
         if kv_cache_dtype_str in ("fp8", "fp8_e4m3", "fp8_e5m2"):
             kv_cache_dtype_str = "fp8"
@@ -621,7 +667,7 @@ def _build_decode(
             ]
         )
         use_gluon_decode = AiterMLAHelper.use_gluon_decode(
-            self.num_heads, int(max_qo_len), self._kv_cache_dtype_str
+            self._decode_num_heads, int(max_qo_len), self._kv_cache_dtype_str
         )
 
         if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
@@ -697,16 +743,16 @@ def _build_decode(
         has_persistent_metadata = False
         use_persistent_metadata = (
             not AiterMLAHelper.use_gluon_decode(
-                self.num_heads, max_qo_len, self._kv_cache_dtype_str
+                self._decode_num_heads, max_qo_len, self._kv_cache_dtype_str
             )
             and not AiterMLAHelper.use_gluon_verify(
-                self.num_heads, max_qo_len, self._kv_cache_dtype_str
+                self._decode_num_heads, max_qo_len, self._kv_cache_dtype_str
             )
             # A padded rank has no bf16 persistent kernel past qlen 4 where the
             # gfx950 fold is absent; the non-persistent entry covers it. fp8
             # keeps the schedule -- its fold rejects non-persistent outright.
             and (
-                self.num_heads >= AiterMLAHelper._AITER_MIN_MLA_HEADS
+                self._decode_num_heads >= AiterMLAHelper._AITER_MIN_MLA_HEADS
                 or max_qo_len <= AiterMLAHelper._ASM_PADDED_MAX_PS_QLEN
                 or is_quantized_kv_cache(self._kv_cache_dtype_str)
             )
@@ -901,6 +947,15 @@ def get_mla_unpadded_o(num_heads: int, o: torch.Tensor) -> torch.Tensor:
         # first num_heads.
         return o[:, :num_heads, :]
 
+    @staticmethod
+    def get_mla_unpadded_lse(num_heads: int, lse: torch.Tensor) -> torch.Tensor:
+        m = AiterMLAHelper._AITER_MIN_MLA_HEADS
+        if num_heads >= m:
+            return lse
+        if m % num_heads == 0:
+            return lse[:, :: m // num_heads]
+        return lse[:, :num_heads]
+
     @staticmethod
     def use_gluon_decode(num_heads: int, max_qo_len: int, kv_cache_dtype: str) -> bool:
         # Small-head (<16) single-token decode takes either the Gluon kernel or
@@ -925,23 +980,60 @@ def use_gluon_decode(num_heads: int, max_qo_len: int, kv_cache_dtype: str) -> bo
 
     @staticmethod
     def use_gluon_verify(num_heads: int, max_qo_len: int, kv_cache_dtype: str) -> bool:
-        """Whether a small-head multi-token verify is flattened onto Gluon.
-
-        bf16 has no gqa<16, qseqlen>1 asm kernel, so the verify is flattened
-        into per-token Gluon decodes. fp8 has one via the q-row fold and must
-        not come here: the flatten hands Gluon the batch size its fp8 regime
-        asserts against. A predicate rather than inline in forward_mqa so the
-        builder sees the same answer the impl acts on.
+        """Whether a multi-token verify is flattened onto Gluon.
+
+        Each verify row becomes its own batch entry with its own KV range, so the
+        kernel applies no causal tail of its own. That is what makes the flatten
+        correct under DCP: the tail has to be taken in global positions, and the
+        dropped tokens land on different ranks, so no per-rank length can express
+        it (see `dcp_local_verify_row_lens`). It also returns a per-row LSE, which
+        is what DCP merges with.
+
+        Needs a head count Gluon takes: ROCm/aiter#4412 tiles up to 96, which is
+        what TP8 x DCP8 gathers. Gluon's fp8-KV regime reads a bf16 query, so an
+        fp8 KV cache keeps the asm decode.
         """
-        if num_heads >= AiterMLAHelper._AITER_MIN_MLA_HEADS or max_qo_len <= 1:
+        if max_qo_len <= 1:
+            return False
+        if num_heads > _gluon_mla_max_bh16_heads():
             return False
         if is_quantized_kv_cache(kv_cache_dtype):
             return False
         # Same arch and mode gating as use_gluon_decode.
         return _aiter_mla_small_head_mode() != "asm" and _gluon_mla_decode_supported()
 
+    @staticmethod
+    def dcp_local_verify_row_lens(
+        tot_seq_lens: torch.Tensor,
+        qlen: int,
+        dcp_world_size: int,
+        dcp_rank: int,
+        cp_interleave: int,
+    ) -> torch.Tensor:
+        """Local KV length of every row of a DCP verify block.
+
+        Row `i` of a qlen-token verify attends global positions
+        `[0, seq_len - qlen + i]`, so its local length is the round-robin count
+        evaluated at that bound. Taking the request's full local length and
+        subtracting the causal offset is wrong: those dropped tail tokens sit on
+        `qlen - 1 - i` different ranks, so only some ranks lose one, while the
+        subtraction takes one from every rank.
+
+        The mapping itself is the shared `get_dcp_local_seq_lens`; only the bound
+        it is evaluated at is per row rather than once per request.
+        """
+        offsets = torch.arange(
+            qlen, device=tot_seq_lens.device, dtype=tot_seq_lens.dtype
+        )
+        visible = (tot_seq_lens.unsqueeze(1) - (qlen - 1) + offsets).clamp_(min=0)
+        return get_dcp_local_seq_lens(
+            visible, dcp_world_size, dcp_rank, cp_interleave
+        ).flatten()
+
 
 class AiterMLAImpl(MLACommonImpl[AiterMLAMetadata]):
+    can_return_lse_for_decode: bool = True
+
     def __init__(
         self,
         num_heads: int,
@@ -971,6 +1063,21 @@ def __init__(
             **mla_args,
         )
         AiterMLAHelper.check_num_heads_validity(num_heads)
+        self._decode_num_heads = self.num_heads * self.dcp_world_size
+        AiterMLAHelper.check_num_heads_validity(self._decode_num_heads)
+
+        # Needed to place a verify row's causal window on this rank's KV shard.
+        self._cp_kv_cache_interleave_size = 1
+        if self.dcp_world_size > 1:
+            from vllm.config import get_current_vllm_config
+            from vllm.distributed.parallel_state import get_dcp_group
+
+            self._dcp_rank = get_dcp_group().rank_in_group
+            self._cp_kv_cache_interleave_size = (
+                get_current_vllm_config().parallel_config.cp_kv_cache_interleave_size
+            )
+        else:
+            self._dcp_rank = 0
 
         unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
         if any(unsupported_features):
@@ -1201,7 +1308,8 @@ def forward_mqa(
             )
             kv_buffer = kv_c_and_k_pe_cache.reshape(-1, kv_c_and_k_pe_cache.shape[-1])
             mla_gluon = _get_mla_gluon()
-            mla_gluon(
+            need_lse = self.dcp_world_size > 1
+            gluon_ret = mla_gluon(
                 q_nope=q_nope,
                 q_pe=q_pe,
                 kv_c=kv_buffer,
@@ -1214,8 +1322,18 @@ def forward_mqa(
                 use_2d_view=False,
                 kv_scale=1.0,
                 min_kv_seq_len=decode.min_kv_seq_len,
+                return_lse=need_lse,
             )
-            return o, None
+            lse = gluon_ret[1] if isinstance(gluon_ret, tuple) else None
+            if need_lse:
+                assert lse is not None, (
+                    "aiter mla_gluon(return_lse=True) returned no LSE; upgrade aiter "
+                    "to a build with gluon LSE support."
+                )
+                lse = lse.reshape(B, num_q_heads)
+            # Gluon is only reached with the query heads unpadded, so o and lse
+            # already have the gathered head count the merge expects.
+            return o, lse
 
         # 12-head (<16) multi-token verify (DSpark): the asm path has no
         # gqa<16, qseqlen>1 kernel. Flatten each verify token to its own
@@ -1228,7 +1346,7 @@ def forward_mqa(
         # builder -- which has to know whether the asm decode will run -- sees
         # the same answer as this branch.
         if AiterMLAHelper.use_gluon_verify(
-            self.num_heads, int(decode.max_qo_len), self.kv_cache_dtype
+            self._decode_num_heads, int(decode.max_qo_len), self.kv_cache_dtype
         ):
             qlen = int(decode.max_qo_len)
             if type(q) is tuple:
@@ -1262,14 +1380,21 @@ def forward_mqa(
             row_req = torch.arange(per_req_len.shape[0], device=dev).repeat_interleave(
                 qlen
             )
-            row_len = (
-                (
-                    per_req_len.unsqueeze(1)
-                    - (qlen - 1)
-                    + torch.arange(qlen, device=dev, dtype=per_req_len.dtype)
-                )
-                .clamp_(min=0)
-                .flatten()
+            # Under DCP a row's window has to be counted in global positions and
+            # then mapped onto this rank's shard, because the tail tokens a row
+            # drops sit on other ranks. Without DCP the shard is the whole
+            # sequence and this reduces to per_req_len - (qlen - 1) + t.
+            row_len_source = (
+                decode.dcp_tot_seq_lens
+                if self.dcp_world_size > 1 and decode.dcp_tot_seq_lens is not None
+                else per_req_len
+            )
+            row_len = AiterMLAHelper.dcp_local_verify_row_lens(
+                row_len_source,
+                qlen,
+                self.dcp_world_size,
+                self._dcp_rank,
+                self._cp_kv_cache_interleave_size,
             )
             new_indptr = torch.cat([old_indptr.new_zeros(1), row_len.cumsum(0)]).to(
                 torch.int32
@@ -1283,7 +1408,8 @@ def forward_mqa(
             )
             new_indices = decode.paged_kv_indices[src]
             mla_gluon = _get_mla_gluon()
-            mla_gluon(
+            need_lse = self.dcp_world_size > 1
+            gluon_ret = mla_gluon(
                 q_nope=q_nope,
                 q_pe=q_pe,
                 kv_c=kv_buffer,
@@ -1296,8 +1422,22 @@ def forward_mqa(
                 use_2d_view=False,
                 kv_scale=1.0,
                 min_kv_seq_len=int(row_len.min()),
+                return_lse=need_lse,
+            )
+            if not need_lse:
+                return o, None
+            lse = gluon_ret[1] if isinstance(gluon_ret, tuple) else None
+            assert lse is not None, (
+                "aiter mla_gluon(return_lse=True) returned no LSE; upgrade aiter "
+                "to a build with gluon LSE support."
             )
-            return o, None
+            # A row whose shard holds none of its window contributes nothing to
+            # the merge. Its output is never written either, so clear it: a -inf
+            # weight would still carry an uninitialized NaN into the sum.
+            empty_rows = (row_len == 0).unsqueeze(1)
+            lse = lse.reshape(B, num_q_heads).masked_fill(empty_rows, float("-inf"))
+            o = o.masked_fill(empty_rows.unsqueeze(2), 0)
+            return o, lse
 
         if type(q) is tuple:
             q = torch.cat(q, dim=-1)
@@ -1305,8 +1445,12 @@ def forward_mqa(
         assert isinstance(q, torch.Tensor)
         B = q.shape[0]
 
-        mla_padded_q = AiterMLAHelper.get_mla_padded_q(self.num_heads, q)
-        mla_num_heads = AiterMLAHelper.get_actual_mla_num_heads(self.num_heads)
+        assert q.shape[1] == self._decode_num_heads, (
+            "ROCM_AITER_MLA decode expected the DCP-gathered query head count "
+            f"{self._decode_num_heads}, got {q.shape[1]}"
+        )
+        mla_padded_q = AiterMLAHelper.get_mla_padded_q(self._decode_num_heads, q)
+        mla_num_heads = AiterMLAHelper.get_actual_mla_num_heads(self._decode_num_heads)
         o = torch.empty(
             B,
             mla_num_heads,
@@ -1338,17 +1482,49 @@ def forward_mqa(
                 reduce_partial_map=attn_metadata.reduce_partial_map,
             )
 
-        rocm_aiter_ops.mla_decode_fwd(
-            mla_padded_q,
-            kv_buffer,
-            o,
-            self.scale,
-            decode.qo_indptr,
-            decode.max_qo_len,
-            decode.paged_kv_indptr,
-            decode.paged_kv_indices,
-            decode.paged_kv_last_page_len,
-            **mla_kwargs,
-        )
+        final_lse = None
+        if self.dcp_world_size > 1:
+            if decode.max_qo_len != 1:
+                raise NotImplementedError(
+                    "ROCM_AITER_MLA DCP currently supports single-token decode "
+                    "only; multi-token verification requires global cprr metadata."
+                )
+            # The vLLM custom-op wrapper only exposes the in-place output and
+            # discards AITER's final LSE. DCP needs that LSE to combine partial
+            # softmax states across KV shards, so use AITER's native entry point.
+            from aiter.mla import mla_decode_fwd
+
+            _, final_lse = mla_decode_fwd(
+                mla_padded_q,
+                kv_buffer.view(-1, 1, 1, mla_padded_q.shape[-1]),
+                o,
+                decode.qo_indptr,
+                decode.paged_kv_indptr,
+                decode.paged_kv_indices,
+                decode.paged_kv_last_page_len,
+                decode.max_qo_len,
+                sm_scale=self.scale,
+                return_lse=True,
+                **mla_kwargs,
+            )
+            assert final_lse is not None
+        else:
+            rocm_aiter_ops.mla_decode_fwd(
+                mla_padded_q,
+                kv_buffer,
+                o,
+                self.scale,
+                decode.qo_indptr,
+                decode.max_qo_len,
+                decode.paged_kv_indptr,
+                decode.paged_kv_indices,
+                decode.paged_kv_last_page_len,
+                **mla_kwargs,
+            )
 
-        return AiterMLAHelper.get_mla_unpadded_o(self.num_heads, o), None
+        output = AiterMLAHelper.get_mla_unpadded_o(self._decode_num_heads, o)
+        if final_lse is not None:
+            final_lse = AiterMLAHelper.get_mla_unpadded_lse(
+                self._decode_num_heads, final_lse
+            )
+        return output, final_lse
diff --git a/vllm/v1/core/kv_cache_coordinator.py b/vllm/v1/core/kv_cache_coordinator.py
index f5cd79b285f6..0df5fd7490a3 100644
--- a/vllm/v1/core/kv_cache_coordinator.py
+++ b/vllm/v1/core/kv_cache_coordinator.py
@@ -140,7 +140,16 @@ def __init__(
                 block_pool=self.block_pool,
                 enable_caching=enable_caching,
                 kv_cache_group_id=i,
-                dcp_world_size=dcp_world_size,
+                # Replicated draft groups keep dcp_world_size=1 block geometry.
+                dcp_world_size=(
+                    1
+                    if getattr(
+                        kv_cache_group.kv_cache_spec,
+                        "non_causal_multi_token_decode",
+                        False,
+                    )
+                    else dcp_world_size
+                ),
                 pcp_world_size=pcp_world_size,
                 scheduler_block_size=self.scheduler_block_size,
                 needs_kv_cache_zeroing=self.kv_cache_config.needs_kv_cache_zeroing,
diff --git a/vllm/v1/core/kv_cache_utils.py b/vllm/v1/core/kv_cache_utils.py
index ad51e4b2b397..40fea23dc2df 100644
--- a/vllm/v1/core/kv_cache_utils.py
+++ b/vllm/v1/core/kv_cache_utils.py
@@ -1208,9 +1208,23 @@ def _get_kv_cache_groups_uniform_page_size(
     # is the minimum number of layers among all attention types. Need a better
     # strategy if we want to support more complex patterns (e.g., 20 full + 30
     # sw, where the group size should be 10).
-    min_num_layers = min([len(layers) for layers in layer_buckets])
+    bucket_sizes = [len(layers) for layers in layer_buckets]
+    max_num_layers = max(bucket_sizes)
+    # A drafter can add its own attention type with far fewer layers than the
+    # target (e.g. an MLA-only draft on a hybrid mamba/attention target). Letting
+    # that bucket set group_size over-splits the target's layers, and makes an
+    # engine that holds the drafter group its mamba layers differently from one
+    # that does not, breaking PD-disaggregated KV transfer. Draft layers are few,
+    # so padding them up to the target's group size is cheap.
+    primary_sizes = [
+        len(layers)
+        for layers, specs in zip(layer_buckets, spec_buckets)
+        if not all(
+            getattr(spec, "non_causal_multi_token_decode", False) for spec in specs
+        )
+    ]
+    min_num_layers = min(primary_sizes or bucket_sizes)
     group_size = min_num_layers
-    max_num_layers = max([len(layers) for layers in layer_buckets])
     if max_num_layers < min_num_layers * 1.5:
         # If the number of layers is not much larger than the minimum number of
         # layers, use the maximum number of layers as the group size to avoid
diff --git a/vllm/v1/kv_cache_interface.py b/vllm/v1/kv_cache_interface.py
index 90391f6ef0a6..766cdbe92663 100644
--- a/vllm/v1/kv_cache_interface.py
+++ b/vllm/v1/kv_cache_interface.py
@@ -395,6 +395,22 @@ def __post_init__(self):
         super().__post_init__()
         _apply_alignment_padding(self)
 
+    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
+        # Replicated draft groups store the full sequence on every rank.
+        if self.non_causal_multi_token_decode:
+            return (
+                cdiv(vllm_config.model_config.max_model_len, self.block_size)
+                * self.page_size_bytes
+            )
+        return super().max_memory_usage_bytes(vllm_config)
+
+    def max_num_blocks_per_req(self, vllm_config: VllmConfig, max_len: int) -> int:
+        # A replicated group's block table indexes the whole sequence, so it is
+        # not narrowed by the DCP shard count.
+        if self.non_causal_multi_token_decode:
+            return cdiv(max_len, self.block_size)
+        return super().max_num_blocks_per_req(vllm_config, max_len)
+
     @property
     def storage_block_size(self) -> int:
         return self.block_size // self.compress_ratio
diff --git a/vllm/v1/worker/cp_utils.py b/vllm/v1/worker/cp_utils.py
index 92d8383c1f12..1945fc6d5b12 100644
--- a/vllm/v1/worker/cp_utils.py
+++ b/vllm/v1/worker/cp_utils.py
@@ -37,6 +37,11 @@ def check_attention_cp_compatibility(vllm_config: VllmConfig) -> None:
             layer_impl = getattr(layer, "impl", None)
             if layer_impl is None:
                 continue
+            if dcp_size > 1 and getattr(layer_impl, "dcp_world_size", dcp_size) <= 1:
+                # A layer that pinned itself out of DCP keeps its KV replicated on
+                # every rank (the Kimi-K3 DSpark draft), so it never joins the
+                # cross-rank merge and needs neither a decode LSE nor interleaving.
+                continue
             if vllm_config.speculative_config is not None and interleave_size > 1:
                 assert layer_impl.supports_mtp_with_cp_non_trivial_interleave_size, (
                     "MTP with cp_kv_cache_interleave_size > 1 is not "
diff --git a/vllm/v1/worker/gpu/block_table.py b/vllm/v1/worker/gpu/block_table.py
index 90fc104dc29f..0027a9b91da9 100644
--- a/vllm/v1/worker/gpu/block_table.py
+++ b/vllm/v1/worker/gpu/block_table.py
@@ -26,7 +26,11 @@ def __init__(
         cp_size: int = 1,
         cp_rank: int = 0,
         cp_interleave: int = 1,
+        cp_exempt_groups: list[int] | None = None,
     ):
+        # Groups listed here use an unsharded slot mapping (CP_SIZE=1) so every rank
+        # writes the full sequence; used for DSpark draft MLA under DCP.
+        self.cp_exempt_groups = list(cp_exempt_groups or ())
         self.block_sizes = block_sizes
         self.kernel_block_sizes = kernel_block_sizes
         self.max_num_reqs = max_num_reqs
@@ -204,6 +208,25 @@ def compute_slot_mappings(
             PAD_ID=PAD_SLOT_ID,
             TRITON_BLOCK_SIZE=1024,  # type: ignore
         )
+        # Re-run the slot-mapping kernel with CP_SIZE=1 for replicated groups.
+        if self.cp_size > 1:
+            for gid in self.cp_exempt_groups:
+                _compute_slot_mappings_kernel[(1, num_reqs + 1)](
+                    slot_mappings.shape[1],
+                    idx_mapping,
+                    query_start_loc,
+                    positions,
+                    self.block_table_ptrs[gid:],
+                    self.block_table_strides[gid:],
+                    self.block_sizes_tensor[gid:],
+                    slot_mappings[gid:],
+                    slot_mappings.stride(0),
+                    0,
+                    CP_SIZE=1,
+                    CP_INTERLEAVE=self.cp_interleave,
+                    PAD_ID=PAD_SLOT_ID,
+                    TRITON_BLOCK_SIZE=1024,  # type: ignore
+                )
         return slot_mappings[:, :num_tokens_padded]
 
     def get_dummy_slot_mappings(self, num_tokens: int) -> torch.Tensor:
diff --git a/vllm/v1/worker/gpu/model_runner.py b/vllm/v1/worker/gpu/model_runner.py
index 93c3e250a2b2..8cec99875258 100644
--- a/vllm/v1/worker/gpu/model_runner.py
+++ b/vllm/v1/worker/gpu/model_runner.py
@@ -501,14 +501,24 @@ def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
 
         block_sizes = []
         max_num_blocks_per_group = []
+        # KV cache groups whose KV is replicated on every DCP rank (DSpark draft).
+        cp_exempt_groups: list[int] = []
         for kv_cache_group in kv_cache_config.kv_cache_groups:
             spec = kv_cache_group.kv_cache_spec
             block_sizes.append(spec.block_size)
             # When using DCP, each request's KV cache is sharded among different ranks.
             # As a result, one block on the current rank covers `block_size * cp_size`
-            # tokens in the full, global (unsharded) sequence.
+            # tokens in the full, global (unsharded) sequence. Replicated groups keep
+            # the undivided block count instead.
+            group_cp_size = (
+                1
+                if getattr(spec, "non_causal_multi_token_decode", False)
+                else self.dcp_size
+            )
+            if group_cp_size != self.dcp_size:
+                cp_exempt_groups.append(len(block_sizes) - 1)
             max_num_blocks = cdiv(
-                block_table_max_model_len, spec.block_size * self.dcp_size
+                block_table_max_model_len, spec.block_size * group_cp_size
             )
             # For Mamba/Hybrid Model, KVCaches need extra blocks for speculative tokens
             if isinstance(spec, MambaSpec):
@@ -552,6 +562,7 @@ def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
             cp_size=self.dcp_size,
             cp_rank=self.dcp_rank,
             cp_interleave=self.cp_interleave,
+            cp_exempt_groups=cp_exempt_groups,
         )
         self.pcp_manager = pcp.maybe_build_pcp_manager(
             self.vllm_config,
PR51705_DIFF_EOF
    # Verify every hunk applies BEFORE touching anything -- a half-applied
    # 11-file patch is far worse than an unpatched tree.
    if ! patch -p1 -d "$root" --dry-run --forward < "$d" >/dev/null 2>&1; then
        echo "[$label] dry-run failed; leaving vllm unpatched." >&2
        patch -p1 -d "$root" --dry-run --forward < "$d" 2>&1 | grep -iE "fail|hunk" | head -10 >&2
        rm -f "$d"; return 0
    fi
    patch -p1 -d "$root" --forward --backup --suffix=.pr51705.orig < "$d" >/dev/null 2>&1 \
        && echo "[$label] applied PR #51705 (258ae29579c9ec26)" \
        || echo "[$label] apply failed after a clean dry-run" >&2
    rm -f "$d"
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
SKIP_PATCH_PR51705="${SKIP_PATCH_PR51705:-0}"
if [ "$SKIP_PATCH_PR51705" = "1" ]; then
    SKIP_PATCH_DCPLSE="${SKIP_PATCH_DCPLSE:-0}"
else
    SKIP_PATCH_DCPLSE="${SKIP_PATCH_DCPLSE:-1}"
fi
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
echo "[kimi-patches] done."
