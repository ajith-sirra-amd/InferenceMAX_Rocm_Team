# AMD fused FP8 KV reconstruction accuracy evidence

- Run:39726650098; measured commit:71c4e398be3358ea19f08b006b740e894898d0cb.
- Model:amd/Kimi-K3-Quark-MXFP4-AttnFP8; TP8/DCP8; GSM8K real DSpark verification.
- GSM8K:96.588324% strict;96.512509% flexible.
- All8 GPUs:FP8 kv_b weights/scales, FP8-byte all-gather, fused reconstruction, BF16 K/V and BF16 FMHA inputs verified.
- Real cached-prefix checks executed7680-row reconstruction on all24 MLA layers across all8 GPUs before GSM8K.
- Initial numerical checks compared fused output with independent FP32 decoded-checkpoint projection.
- Timing:cached-prefix HTTP checks completed05:33:35 UTC; GSM8K began05:33:35 UTC; final all8 gate passed05:37:30 UTC.
- Limitation:execution markers are one-time per backend and first appeared before GSM8K. They prove actual cached-context fusion in the same server, not which individual GSM8K requests exercised it or their share.
- New-token attention and decode retain their existing runtime paths. FP8 ASM FMHA is not enabled.
- Eval KV pool:3,574,832 tokens with32K server context; do not compare directly with1M-context throughput-run pool.
- Throughput improvement remains unproven. Subsequent experiments use Moonshot per user instruction.
