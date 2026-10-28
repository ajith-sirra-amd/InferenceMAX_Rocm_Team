[中文](projection-fp8-experiment_zh.md)

# BF16 projection with FP8 K/V output experiment

Status: GPU kernel experiment completed successfully; the stress diagnostic exceeded its numerical gate. The serving plugin does not select these kernels. See the measured results below.

ATOM source was checked at [773eb362](https://github.com/ROCm/ATOM/blob/773eb36254e43060bfd5fabcab7d3840bf591358/atom/model_ops/attention_mla.py#L1675).
Its cached FlyDSL projection accepts caller-supplied output scales. The
[new-token QKV quantizer](https://github.com/ROCm/ATOM/blob/773eb36254e43060bfd5fabcab7d3840bf591358/atom/model_ops/triton_fused_qkv_quant.py#L157)
provides cached gather descales as `2 * max(new_token_descale, 1e-6)`.
This avoids a cached-output maximum reduction, but cached values outside the
estimated range can clip.

The inspected [AITER FlyDSL projection](https://github.com/ROCm/aiter/blob/bbef5a6f2d3733b293c80d2cee6a8d464e88bc01/aiter/ops/flydsl/gather_kv_b_proj.py)
requires FP8 weights with weight scales. ATOM falls back to Triton for Kimi's
unquantized BF16 `kv_b_proj`; the direct FP8-output route is not available
for those weights. Current AITER main had the same wrapper when inspected.

The experiment adapts the AITER Triton flat gather/projection body, retaining
BF16 projection operands and FP32 accumulation. It compares:

| Mode | Projection passes | Expanded BF16 intermediate | Scale source |
| --- | --- | --- | --- |
| Existing native + dynamic quantization | 1 | Yes | Maximum of each cached K/V tensor |
| `fused_amax` | 1 | Yes; avoids separate maximum scan | Maximum collected inside projection |
| `recompute` | 2 | No | First projection collects maxima; second writes FP8 |
| `supplied_scales` | 1 | No | ATOM-style new-token scales with 2x headroom |

All experimental modes retain native BF16 output rounding **in registers**
before conversion to FP8. ATOM's FlyDSL direct epilogue skips that BF16
rounding; this experiment retains it to isolate materialization and scaling
effects. The supplied-scale mode changes cached quantization numerics even
with the same BF16 projection. It is not exact parity with current dynamic
cached scaling.

## Run

Use the existing pinned ROCm image
`vllm/vllm-openai-rocm:nightly-1b2a666391f2a24228a1436b64d547c34cc6b6ce`:

```bash
PYTHONPATH="$PWD/upstream/InferenceX/benchmarks/single_node/agentic/atom_attention" \
ATOM_USE_FLYDSL_FP8_PREFILL_ATTN=1 \
python3 utils/atom_attention/projection_experiment.py
```

The experiment is also wired into the existing `runner-diagnostics-only`
job in `.github/workflows/e2e-tests.yml`. This job allocates one node,
runs the isolated GPU test in a disposable container, and does not load Kimi
or launch a serving benchmark. Pin both the workflow ref and checkout ref;
its Actions job log contains the JSON evidence.

## Checks and evidence

- Exact FP8 byte/scale comparisons with native projection plus the existing
  quantizer for dynamic modes, including empty/tail/reordered rows and 12/24 heads.
- A hand-worked constant projection and independent Torch supplied-scale conversion.
- Changed-input and changed-scale graph replay.
- Q512/KV8192 actual plugin routing, BF16-versus-FP8 outputs, and independent
  FP64 attention references using both original and dequantized FP8 inputs.
- Relative RMS, maximum absolute error, LSE error, and nonfinite checks.
  The initial synthetic numerical limits reuse the preflight's RMS <0.10 and
  LSE absolute error <0.20; these are not model-quality acceptance criteria.
- ATOM-style scales are derived from separately generated new-token K/V using
  the real ATOM quantizer. A deliberate 4x cached-range stress case reports
  clipping and accuracy loss; it is diagnostic and is never presented as a pass.
- Graph timing at KV8192, 16384 and 65536 includes native projection plus
  current quantization, all candidates, and a separate full new-token
  quantization charge. Timings exclude model execution and DCP communication.

The existing GSM8K pass exercised BF16 attention under the fixed hybrid
thresholds. It does not validate these kernels. Performance adoption requires
kernel results, representative model-derived scale/clipping evidence,
serving integration, and a new controlled run.

## GPU result: 2026-10-28

[Run 40666434375](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40666434375)
completed successfully on source `8e7bb0eab2195ef2778994806040efbcd067196d`.
[Structured evidence](../../results/atom-attention/run40666434375/projection_summary.json)
and [numerical/timing log](../../results/atom-attention/run40666434375/numerical-and-timing.log)
are retained. Recipe/launcher validation passed.

The new BF16 projection epilogues passed empty/tail/reordered-row checks,
hand-worked constant and zero cases, and changed-input/scale graph replay.
Dynamic variants matched the current FP8 bytes and scales exactly. With
identical supplied scales, direct FP8 output matched native BF16 projection
followed by independent Torch quantization exactly on the covered cases.

At Q512/KV8192, the real plugin emitted `ATOM-FP8-ENGAGED`.
Current FP8 FMHA had **3.655% relative RMS output error versus BF16 FMHA**;
against the independent FP64 reference, BF16 error was 0.191% and FP8 error
was 3.650%. Using dequantized FP8 inputs for the FP64 reference isolated FMHA
arithmetic error to 1.157% RMS. FP8 LSE maximum absolute error versus the
original-input FP64 reference was 0.01509. These are tensor-error metrics,
not percentages of correctly answered model questions.

ATOM-style supplied scales had 3.700% RMS error versus BF16 with no clipping
in the ordinary synthetic case. In the deliberate 4x cached-range stress
case, K clipped 1.688% of values and V clipped 0.715%; output RMS error rose
to 19.875%, exceeding the 10% diagnostic gate. This is evidence about the
scale-reuse assumption, not a failure of same-scale projection equivalence.
The overall job success does not turn that stress diagnostic into an accuracy pass.

All times below are microseconds per call; 12 heads:

| Cached tokens | Existing projection + quant | Fused maximum | Recompute | Supplied scales | Supplied + new-token preparation |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8192 | 93.929 | 86.341 | 126.141 | 60.537 | 66.303 |
| 16384 | 146.367 | 137.705 | 190.534 | 92.875 | 98.765 |
| 65536 | 540.723 | 539.351 | 581.940 | 272.595 | 278.511 |

The supplied-scale projection stage was 1.55–1.98x faster before scale
preparation and 1.42–1.94x including the full measured ATOM new-token
quantization charge for one cached chunk. Exact-scale recomputation regressed.
Fusing the maximum showed a modest benefit at 8K/16K and essentially no
benefit at 64K.

These were graph microbenchmarks on a shared host with other containers present
and a queue-interception warning in the log. They establish a candidate,
not a serving-throughput gain. No model or DCP collective was measured.

Further validation is scoped to the introduced BF16 epilogue, scale handoff
and hybrid routing. The next useful check is scale/range behavior on
representative Kimi new-token and cached tensors before a serving performance
comparison. The existing ATOM FMHA algorithm is reused; SA's fixed GSM8K
remains the model evaluation. Production serving still uses the existing path.


The follow-up `utils/atom_attention/projection_numerics.py` reports K/V errors separately for cache widening, BF16 projection, and final FP8 quantization. It uses the same FP8 cache and BF16 weights for an independent FP64 reference, with cache scales 0.125 and 0.137. The diagnostics job currently runs this focused follow-up. Results pending.
