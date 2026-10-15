# Experimental raw FP8 DCP context projection

[??](fp8ctx-raw-projection_zh.md)

This prototype gathers FP8 cache bytes into a uint8 workspace and preserves that
representation through the existing DCP collective and reorder. A Triton tiled
projection applies scalar cache scale inside the kernel, rounds latent inputs
to BF16, performs BF16 dot arithmetic with cached dequantized projection weights,
and writes FP8 expanded KV directly. Positional keys apply the same cache scale.
No BF16 latent or expanded-KV intermediate is materialized in steady state.
FP8 concatenation and the previous fused padding path remain. Installed AITER
attention and reduction kernels and the Docker image remain unchanged.

This is an experimental alternative projection, not an FP8 GEMM performance
claim. It uses additional cached BF16 weights and an FP8 workspace, and may lose
to the installed tuned projection. Weight dequantization is lazy and may fail
if the installed quantization method cannot export its effective weights.
Only scalar E4M3 cache format and bias-free projection are supported; source
installation requires the exact pinned DCP method and fails on drift.

First live context validates K/V against the installed quantized projection,
then compares complete ASM attention output and LSE against BF16 attention on
the original projected values. Provisional smoke gates: relative RMS <= 0.1
and max LSE error <= 0.5. These do not establish model-level correctness.
Repeated warmed projection timing is logged; it excludes gather/collective and
some outer allocations, so it is not full-path timing. Run C96, warmup=1 and
600-second benchmark with profiling disabled before a full comparison.

Runtime validation is pending. The previous preparation smoke is still running;
this branch is published for review without dispatching a competing GPU job.
