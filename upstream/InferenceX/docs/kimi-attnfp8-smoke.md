# Kimi AttnFP8 checkpoint smoke

[中文](kimi-attnfp8-smoke_zh.md)

Source: user-selected chore/sa-agentx-v1.0, local HEAD
79cb30ed32023bc543655e1a13f99fe7c1104519. Model:
amd/Kimi-K3-Quark-MXFP4-AttnFP8. C48 TP8/DCP8 DSpark MTP k4,
existing offload/scheduling settings, image
vllm/vllm-openai-rocm:nightly-1b2a666391f2a24228a1436b64d547c34cc6b6ce.
One warmup request per lane, 600-second measured replay, no profiler.

Use checkpoint quantization without the source recipe's online quantization
override. Do not install the legacy BF16-to-FP8 context ASM cast. Record
checkpoint configuration and actual context Q/K/V dtypes without converting
tensors or changing attention dispatch. Fail preflight if the context probe
anchor is missing, and fail replay if context input evidence is absent.
BF16 input evidence is a finding; it does not prove native FP8 FMHA.
Runtime support, accuracy and kernel selection require runner verification.

Network access restored; fetched source matches the validated local base.
AMD model card confirms MXFP4 experts/shared experts and PTPC-FP8 self_attn;
this establishes projection quantization, not context FMHA operand dtype.
Use the source recipe image for the smoke; runtime support requires launch proof.
Do not apply the model card's MXFP4 emulation patch for a different checkpoint.
Keep existing queued/running workflows untouched; submit one additional run.
Check run status every 20 minutes while the monitoring session is active.
