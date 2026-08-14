#!/usr/bin/env bash
# Stop the KV block-pool free-list corruption that kills the engine mid-run.
#
# Symptom (load-dependent; sooner at higher concurrency):
#   AssertionError at vllm/v1/core/kv_cache_utils.py:292  assert curr_block is not None
#   or            at vllm/v1/core/block_pool.py:667       assert block.ref_cnt == 0
#   -> EngineCore encountered a fatal error -> EngineDeadError
# Observed: c10 died at 3612 s, c12 at 487 s, c16 at 354 s.
#
# Root cause: single_type_kv_cache_manager.py:321, in
# allocate_external_computed_blocks(), is the ONLY unguarded get_new_blocks()
# call site in that file (siblings at :363 and :1608 clamp or early-return):
#
#     self.block_pool.get_new_blocks(
#         cdiv(num_total_computed_tokens, self.block_size) - len(req_blocks))
#
# When len(req_blocks) exceeds the block count implied by
# num_total_computed_tokens the argument is NEGATIVE, and a negative count is
# silently destructive rather than an error:
#   * block_pool.get_new_blocks only rejects  num_blocks > free
#   * FreeKVCacheBlockQueue.popleft_n passes  assert num_free_blocks >= n
#   * it then runs  num_free_blocks -= n   -> an INCREASE
#   * range(n) iterates zero times, so the linked list is untouched
#   * returns []
# num_free_blocks is now inflated relative to the real free list. A later
# legitimate popleft_n trusts the counter, walks past the tail and asserts.
#
# Note this is on the EXTERNAL (offloaded) block path, which is why it only
# appears with --kv-transfer-config / SimpleCPUOffloadConnector, and why
# --no-async-scheduling did NOT help (tested: c12 died at 490 s vs 487 s).
#
# Fix: clamp to >= 0. You cannot allocate a negative number of blocks.
# Idempotent; no-op if already patched or the anchor is absent.
set -euo pipefail
PY=${PYTHON:-python3}

TARGET=$($PY - <<'EOF'
import os
try:
    import vllm.v1.core.single_type_kv_cache_manager as m
    print(os.path.abspath(m.__file__))
except Exception:
    print("")
EOF
)
if [ -z "$TARGET" ] || [ ! -f "$TARGET" ]; then
    echo "[kv-blockpool-fix] target not found; nothing to do."
    exit 0
fi
if grep -q "num_new_blocks = max(" "$TARGET"; then
    echo "[kv-blockpool-fix] already patched: $TARGET"
    exit 0
fi
cp -n "$TARGET" "$TARGET.orig" || true
$PY - "$TARGET" <<'EOF'
import sys, io
path = sys.argv[1]
src = io.open(path, encoding="utf-8").read()
old = """        req_blocks = self.req_to_blocks[request_id]
        allocated_blocks = self.block_pool.get_new_blocks(
            cdiv(num_total_computed_tokens, self.block_size) - len(req_blocks)
        )"""
new = """        req_blocks = self.req_to_blocks[request_id]
        # PATCHED: clamp to >= 0; a negative count silently inflates
        # FreeKVCacheBlockQueue.num_free_blocks and corrupts the free list.
        num_new_blocks = max(
            0, cdiv(num_total_computed_tokens, self.block_size) - len(req_blocks)
        )
        allocated_blocks = self.block_pool.get_new_blocks(num_new_blocks)"""
if src.count(old) != 1:
    sys.stderr.write("[kv-blockpool-fix] anchor missing or not unique; aborting.\n")
    sys.exit(1)
io.open(path, "w", encoding="utf-8").write(src.replace(old, new))
print("[kv-blockpool-fix] patched", path)
EOF
echo "[kv-blockpool-fix] done."
