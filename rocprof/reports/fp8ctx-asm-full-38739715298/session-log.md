# FP8 ASM session log

Run: https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/38739715298
Commit: 74bc3baf2df75a932956c304c927f6904eadb748
Superseded run: 38737862968 (cancel requested).

C96 TP8/DCP8; 10 warmup requests per lane; 3600-second measured replay; profiler disabled. Installed Docker AITER retained; raw FP8 cache/projection and DCP changes on hold.

Log actual FP8 Q/K/V operands immediately before optimized noncausal ASM dispatch. Require every TP worker to record dispatch, successful context calls and optimized/ordinary ASM agreement. Wrong dtype and startup self-test failure are fatal. BF16 numerical accuracy failures are logged once; performance-only continuation authorized by user. No accuracy-qualified claim.

Retain asm_dispatch_evidence.log on cleanup, asm_dispatch_verification.json after replay, server.log and existing raw/metrics artifacts.

Local behavioral verifier cases passed: eight-worker success, missing worker, BF16 operand, runtime failure, optimized mismatch and nonoptimized dispatch. CLI success/failure outputs and exit codes checked. Embedded recipe Python compiles. Local GPU execution and Bash syntax execution unavailable; actual GPU proof pending runner logs.

Run 38739412160 failed before checkout/server launch: EBUSY on .nfs files left open by the cancelled benchmark container. Recovery removes bmk-server only if its mount source exactly matches GITHUB_WORKSPACE, on MI355X runners. Replacement run 38739715298 is active on pit2_p03_g07. Actual FP8 evidence pending.

Completed FlyDSL power-of-two quantizer smoke 38736515115: sampled total call 2.2452-2.3632 ms vs BF16 2.8734-3.2528 ms; quantizer 0.1592-0.1627 ms. All workers failed LSE gate (0.505182-0.641809 vs limit 0.5), output RMS 0.025846-0.041597. No completed throughput result; no further FlyDSL dispatch.
