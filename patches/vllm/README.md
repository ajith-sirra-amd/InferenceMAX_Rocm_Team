# Native FP8 FMHA parity draft

[中文](README_zh.md)

The [patch](native-fp8-fmha-parity.patch) ports the existing plugin into native
vLLM modules on **`1b2a666391f2a24228a1436b64d547c34cc6b6ce`**, the vLLM commit
used by the completed performance and GSM8K runs. It has not been applied to a
running server, and no native GPU or end-to-end pass is claimed.

The reference plugin revision is
`2a656fdc330761bb076b5c9115307dff4cbadc3a`. Its flag-on computation is the path
tested in the earlier `e4cf47a` experiment. The native backend retains that
runtime body with import, class-name and backend-name substitutions. The entire
quantizer is copied without changes, including launch choices, reciprocal
instruction, strided-input handling, and the currently unused KV-only helper.
MIT attribution is included. This preservation is for parity qualification;
deduplicating unused code is a later review step.

## Preserved behavior

- Only `ATOM_USE_FLYDSL_FP8_PREFILL_ATTN=1` enables cached FP8 FMHA.
- Causal and small cached chunks use BF16; FP8 requires max Q >= 512 and max KV >= 8192.
- Identical Q views are reused only within one forward, with cleanup in `finally`.
- Native DCP gather/reconstruction, BF16 expanded K/V and native decode remain in place.
- Main ASM prefill metadata is disabled for this selected backend.
- No model weight-quantization overrides are introduced.
- Native registration replaces the plugin's two CUSTOM registrations. Both native
  backend names must be selected; setting the flag alone does not change the default backend.

## Apply and qualify

Use a clean vLLM checkout at the pinned commit and its matching ROCm environment.
The patch paths are relative to the vLLM repository root, not this experiment repository.

```bash
git -C "$VLLM_CHECKOUT" rev-parse HEAD
git -C "$VLLM_CHECKOUT" apply --check \
    "$EXPERIMENT_REPO/patches/vllm/native-fp8-fmha-parity.patch"
git -C "$VLLM_CHECKOUT" apply \
    "$EXPERIMENT_REPO/patches/vllm/native-fp8-fmha-parity.patch"
```

After making that checkout available in the prepared vLLM ROCm virtual environment:

```bash
.venv/bin/python "$EXPERIMENT_REPO/utils/atom_attention/compare_native_plugin.py" \
    --plugin-root "$EXPERIMENT_REPO/upstream/InferenceX/benchmarks/single_node/agentic/atom_attention" \
    --output "$EXPERIMENT_REPO/results/native-fp8-fmha-parity.json"
```

The comparison uses real vLLM prefill metadata and both implementations on the
same tensors. It checks eight flag/shape combinations, strided Q/V, causal
output, two cached chunks, and changed Q values in reused storage across two
forwards. FP8 bytes, descales, BF16 outputs and FP32 LSE must match exactly.
It also verifies Q quantization/reuse counts. **The script has not been run.**

This kernel comparison does not exercise native backend selection, DCP
collectives, graph capture, complete model loading or the full server forward.
A native controlled-prefix smoke, native graph checks, model evaluation and a
flag-off/on performance comparison remain required before claiming equivalence.

For native serving, use:

```bash
export ATOM_USE_FLYDSL_FP8_PREFILL_ATTN=1
vllm serve "$MODEL" \
    --attention-backend ROCM_AITER_FLYDSL_MLA \
    --attention-config '{"mla_prefill_backend":"ROCM_AITER_FLYDSL_FP8"}'
```

Preserve the established model, TP8/DCP8, weight settings, speculative settings,
and other command arguments for the comparison. The two backend names are the
only replacement for the plugin's two CUSTOM selections.

## Upstream submission status

No vLLM PR was opened. This is a controlled native port, not an upstream-ready
submission. Existing [#59124](https://github.com/vllm-project/vllm/pull/59124)
and [#62447](https://github.com/vllm-project/vllm/pull/62447) already add native
FlyDSL FP8 prefill. Our causal/small-chunk BF16 policy and exact quantizer differ;
the appropriate upstream contribution should be narrowed around those differences.

The [vLLM agent rules](https://github.com/vllm-project/vllm/blob/1cb31be7028457c7eae7b6fabca1941d5d3c4730/AGENTS.md)
say, “If an open PR already addresses the same fix, do not open another.”
They also require the human submitter to review every changed line and run
relevant tests. The [PR checklist](https://github.com/vllm-project/vllm/blob/1cb31be7028457c7eae7b6fabca1941d5d3c4730/.agents/skills/pr-checklist/SKILL.md)
requires behavioral coverage, rendered docs, validation evidence and human review
before opening even a draft.

Review findings: runtime duplication needs consolidation for upstream; inherited
style and diagnostic prints need native conventions; a real server regression test
and ROCm CI selection are missing; native GPU tests, pre-commit and `git apply --check`
were not run here. Local Windows Bash and nested-file staging were denied.
Patch construction checks are not a substitute for those validations.

The existing [performance and GSM8K evidence](../../docs/kimi/fp8-fmha-results-2026-10-28.md)
belongs to the plugin. GSM8K passed 97.19%/97.12%, but its prompts triggered zero FP8 FMHA calls.
