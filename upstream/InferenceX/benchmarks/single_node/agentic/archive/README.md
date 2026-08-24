# Archived configurable Kimi-K3 scripts

The live scripts one directory up are reduced to exactly the code path the
current configuration executes, so that a given commit reads as an unambiguous
record of what ran. These archived copies are the fully parameterised versions
they were reduced from, kept because they carry every knob and every alternative
arm that was explored.

| File | Lines | Contents |
|---|---:|---|
| `kimik3_fp4_mi355x_mtp.configurable.sh` | 1204 | All launcher arms and env knobs |
| `apply_kimi_k3_patches.configurable.sh` | 908 | All eight in-container patches |

## Knobs in the archived launcher

| Variable | Default | Effect |
|---|---|---|
| `DCP_SIZE` | 8 | Decode context parallel size. 1 disables DCP. Must divide `TP`. |
| `DP_SIZE` | 1 | Data-parallel attention groups. `TP * DP` must equal the GPU count; `DP > 1` requires `EP_SIZE > 1`. |
| `DISABLE_SPEC` | 1 | 0 enables DSpark MTP. |
| `SPEC_NUM_TOKENS` | 2 | Draft tokens per step. |
| `MAX_NUM_BATCHED_TOKENS` | 8192 | 0 omits the flag entirely, which is what the reference does. Also sizes the MLA chunked-prefill workspace, so it changes the KV pool. |
| `KV_CACHE_DTYPE` | fp8 | |
| `NONDCP_ATTN_BACKEND` | unset | Unset lets ROCm select `ROCM_AITER_MLA`. Setting it forces a backend. |
| `DCP_ATTN_BACKEND` | ROCM_AITER_MLA | Decode backend on the DCP arm. |
| `DCP_COMM_BACKEND` | a2a | `a2a` or `ag_rs`. |
| `CP_INTERLEAVE` | 1 | `cp-kv-cache-interleave-size`. |
| `DCP_CUDAGRAPH_MODE` | FULL_AND_PIECEWISE | `PIECEWISE` or `NONE` to fall back. |
| `DCP_CAPTURE_SIZES` | sparse ladder | Capture sizes on the DCP arm. |
| `ASYNC_SCHEDULING` | 0 | 1 removes `--no-async-scheduling`. |
| `DISABLE_CUSTOM_AR` | 0 | 1 adds `--disable-custom-all-reduce` (PYNCCL instead of AITER custom). |
| `PIN_RANKS` | 1 | Per-rank NUMA CPU pinning. |
| `PROFILE_DECODE` | 0 | 1 captures a decode-only torch trace and skips the benchmark arm. |
| `PROFILE_DECODE_CONC` / `_WARM_S` / `_WINDOW_S` / `_FLUSH_S` | CONC / 25 / 15 / 60 | Profiling window shape. |
| `EVAL_ONLY` | false | true runs the GSM8K harness instead of the benchmark. |
| `MAX_NUM_SEQS` | `CONC * 2` | |
| `SKIP_KIMI_PATCHES` | 0 | 1 runs stock. |
| `SKIP_PATCH_AITER` / `_CUDAGRAPH` / `_BLOCKPOOL` / `_DCPLSE` / `_PR51705` / `_BLOCKTABLE` / `_DCP_DIRECT` / `_GATHERED_HEADS` | see script | Per-patch disable. |

The archived launcher also contains the unified-image branch, which activates
when `/opt/aiter-local/aiter/configs/merged_bf16_tuned_gemm.csv` is present (the
aigmkt image) and overrides async scheduling, prefill args, spec config and the
capture ladder.

## Patches in the archived patch script

| # | Name | Status against the current image |
|---|---|---|
| 1 | aiter pybind11 internals | No-op where torch and standalone pybind11 already agree |
| 2 | TritonMLA cudagraph support | Live |
| 3 | KV block-pool negative-count clamp | Live |
| 4 | DCP-LSE plumbing | Disabled by default; superseded by PR #51705 |
| 5 | vLLM PR #51705 | Live, pinned by sha256 |
| 6 | DCP block-table sizing | Superseded by PR #51705's per-spec `max_num_blocks_per_req` |
| 7 | Direct DCP a2a ROCm port | Targets `ops/dcp_utils.py`, renamed upstream to `ops/dcp.py` |
| 8 | DCP gathered-head sizing | Superseded; PR #51705 carries `_decode_num_heads` |

Patch [7] additionally depends on a prebuilt `vllm_dcp_direct_rocm.so`; the build
recipe lives in `/home/asirra/dcp-build/` with `port_to_hip.py` documenting each
CUDA-to-HIP transform.
