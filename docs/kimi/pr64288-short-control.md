# Short native C56 control

[??](./pr64288-short-control_zh.md)

Use the current PR source 8a55b8eacf9020a0b421a3c50a927c4a40e61604 for both arms.
Set native-comparison=native; native-fp8-enabled=0 selects the same-source control.
The default remains 1. Flag-off is restricted to replay without evaluation or profiling.
It does not load #59124 or force all attention to BF16; other existing vLLM paths retain their normal routing.

duration-override=300 selects 300 seconds of measurement and one extra warmup
request per lane after mandatory cache primers. Startup, compilation, corpus setup
and priming are additional time. The existing 600-second diagnostic also uses one;
3600-second runs retain ten. Keep all coverage and error gates unchanged.

Run the CPU harness checks first. Pin the resulting InferenceMAX commit in ref and
after-run-id to that successful check. Use the C56 MTP config, pinned e0d00e4b image,
TP8/DCP8, existing FP8 weight configuration, CCD binding, chunk size and BLAS settings.
Do not update the PR or image between the off/on measurements.

The observer checks source hashes, each worker's disabled routing and completed
noncapture prefill calls during replay. It rejects native direct FP8 projection or
FlyDSL FP8 attention in the control, including startup calls. The control artifact
reports flag_off_control=true and is not positive FP8 execution proof.
The enabled arm retains positive projection and cached-FMHA proof.

This is a short diagnostic, not canonical performance or long-run stability evidence.
The historical C56 stall appeared about 50 minutes into measurement, beyond this run.
