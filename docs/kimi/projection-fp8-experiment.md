[中文](projection-fp8-experiment_zh.md)

# BF16 projection with FP8 K/V output experiment

Status: prepared for GPU validation. The serving plugin does not select these
kernels. No new performance or model-accuracy result is claimed.

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
