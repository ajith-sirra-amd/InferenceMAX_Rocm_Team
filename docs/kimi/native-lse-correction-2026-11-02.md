# BF16 LSE correction investigation - 2026-11-02

PR revision: 492080779e9feea19308092b0aba0840e87c6115.
This is diagnostic work; the PR and serving path are unchanged.

Run 41433266352, attempt 1, workflow/execution
fa0e06998fb840ce2212a51c7212bb6373f69abf, GPU job 125236935427
failed its candidate numerical gate. All eight cases passed same-input
arithmetic checks. Per-head control LSE error: 0.208251953125.
Per-channel candidate LSE error: 0.27826881408691406.
Plain and preshuffled layouts agreed. Lazy rescaling did not fix LSE.
BF16 kernel versus FP64 LSE error was 8.1911e-6.
Failure was numerical, not setup, BLAS, or quota.
Evidence artifact: 13134739248; fixture SHA256:
a2b93754cfbea963182bb90f9b8dd10b923108c87aa2a810cdf16921bea53794.

Next: test one fixed normalized 128-dimensional Hadamard rotation of NoPE,
keeping existing RoPE residual columns and actual native FP8 projection.
Rotate Q and K projection weights consistently; prove unquantized attention
invariance before quantizing. Measure exact projection, weight quantization,
Q rounding, K rounding, and combined rounding separately.
Retain the unrotated per-head control; test both native weight layouts and
both FMHA rescaling settings.

Original gates remain output relative RMS <0.10 and absolute LSE <0.20.
All four candidate cases must pass, with projection and same-input checks.
No case is skipped or tolerance widened. One captured chunk is not serving,
model, or performance validation. Calibration uses captured K maxima;
transforms are unfused diagnostics. Integration requires broader numerical
coverage, graph replay, model accuracy and measured performance.

## Rotation result and binary-scale follow-up

Run 41436852207, attempt 1, job 125248773573, source
 ef9ebefa7fc46bc33966c3d4fae3802b88eb6b30 completed with numerical failure.
Artifact 13137092722; 8 arithmetic checks passed, 4/4 candidate BF16 gates failed.
Unrotated LSE 0.208251953125; fixed rotation 0.22992515563964844.
Unquantized rotation identity error 3.55e-15. FP8 weight conversion alone
produced LSE 0.05301 vs BF16. Control Q-only 0.15362; K-only 0.13156;
combined input rounding 0.20829. These maxima are not additive.
Rotation is not promoted. Current PR SHA and passing GSM8K are unchanged.

Next test: retain the original native weights; align Q descale and per-head
balancing to powers of two. This avoids basis rotation and tests whether
binary alignment reduces conversion error. Keep residual RoPE columns,
all four native cases and the original 0.20/0.10 gates. Calibration is still
diagnostic; no serving integration or performance claim.
