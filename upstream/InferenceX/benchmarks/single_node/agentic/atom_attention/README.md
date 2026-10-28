# vLLM with ATOM's FlyDSL FP8 attention

[??](README_zh.md)

- Framework: native vLLM, current pinned ROCm image.
- Model: Moonshot Kimi-K3.
- Kernel: `aiter.ops.flydsl.flydsl_flash_attn_fp8_func`, also used by ATOM.
- Registration: vLLM's general-plugin and custom attention/prefill registries.
- Source edits: this repository only; no installed vLLM/AITER source patch.

| Stage | First integration |
|---|---|
| Weight quantization, KV cache, scheduling | Existing native vLLM recipe |
| Latent KV expansion | Native vLLM; produces BF16 K/V |
| DCP communication | Native path; FP8 all-gather is not claimed |
| Attention preparation | ATOM fused dynamic per-tensor Q/K/V quantization |
| Causal and cached-context FMHA | AITER FlyDSL FP8, BF16 output, FP32 natural-log LSE |
| Decode | Native vLLM AITER implementation |

- ATOM quantizer source: ROCm/ATOM commit `ebd5f84f3a6af6a99d23f85e5c1107b238d1cdaf`; MIT license retained.
- Q is quantized per call. Cross-chunk Q reuse and direct FP8 up-projection are subsequent optimizations, not part of this first integration.
- Adapter fails on unsupported configurations instead of silently claiming FP8 engagement.
- Native AITER uncached ASM bypass is avoided by using vLLM's common prefill orchestration in the custom MLA implementation.
- This first version can be slower than BF16 because preparation and expansion remain. Compare the complete call, not kernel-only timing.
- GPU preflight covers ragged causal/noncausal inputs, empty cached rows, changed-input graph replay, and attention-plus-quantization timing.
- Model loading, TP8/DCP8 communication, runtime engagement and KV capacity require a separate model smoke; standalone kernel success does not prove them.

## Current validation status

- run40634449297: six numerical and changed-input graph checks passed.
- Ordinary-input relative RMS error versus unquantized FP64 reference: 3.7% to5.0%; this is not GSM8K accuracy evidence.
- Shared-host complete-attention timing including QKV quantization: cached context1.26x/1.62x; causal0.38x/0.94x. No end-to-end speedup claim.
- Model smoke: TP8/DCP8 C48, warmup1,600seconds. Full benchmark: warmup10,3600seconds.
- The read-only watcher reports current-attempt active steps and failure annotations every60seconds.

## Automatic continuation

- `utils/atom_attention/continue_benchmark.py` watches the smoke every60seconds and queues a full run only after success, eight-worker causal/cached FP8 engagement, and zero measured-phase request errors.
- Full run pins the tested smoke source SHA, uses warmup10 and3600seconds, and skips GSM8K.
- It does not cancel other runs, stop containers, or modify source. An unmet gate reports evidence and exits.
