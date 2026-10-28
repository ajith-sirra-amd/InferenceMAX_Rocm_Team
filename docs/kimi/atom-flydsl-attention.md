# vLLM Kimi hybrid BF16 / FlyDSL FP8 prefill

[中文](atom-flydsl-attention_zh.md)  [Overhead audit](kimi-prefill-overhead-audit.md)  [Baselines](atom-attention-BASELINES.md)

## Opt-in plugin flag

Set `ATOM_USE_FLYDSL_FP8_PREFILL_ATTN=1` in the vLLM server environment **before startup**. The flag is read when the plugin backend is constructed; changing it requires a server restart.

| Flag | Prefill behavior |
|---|---|
| Unset or `0` | BF16 for all prefill attention; no FlyDSL FP8 API import |
| `1` | FP8 for cached chunks with max query length >= 512 and max KV length >= 8192; BF16 for causal and smaller cached chunks |

Only the exact string `1` enables FP8. The benchmark recipe requires an explicit `0` or `1`; the reusable benchmark workflow sets `'1'` and the MI355X launcher forwards it into the container. For an attention-only ablation, set that workflow environment value to `'0'`.

The plugin must be available on `PYTHONPATH` and selected using `--attention-backend CUSTOM --attention-config '{"mla_prefill_backend":"CUSTOM"}'`. Its `vllm.general_plugins` entry point registers both custom backends. If `VLLM_PLUGINS` is an allowlist, include `inferencex_atom_attention`. The flag alone does not enable this extension in stock vLLM.

This change needs no installed vLLM/AITER source patch. Weight quantization remains independently controlled by `--quantization-config`; `kv_b_proj` stays BF16. Flag-off retains the plugin's native communication/expansion integration and metadata policy, so it is an FMHA ablation, not a complete stock-vLLM baseline.

CPU routing checks: `python -m unittest discover -s upstream/InferenceX/benchmarks/single_node/agentic/atom_attention/tests -p 'test_*.py'`. These exercise the real plugin with external GPU/vLLM doubles, covering off/unset, enabled threshold, causal/small-chunk fallback, missing API and unsupported shapes. GPU numerics and model accuracy require the separate GPU runs.

- Framework: native vLLM in the pinned ROCm image.
- Model: **Moonshot Kimi-K3**; native draft remains Inferact/Kimi-K3-DSpark.
- FP8 API: `aiter.ops.flydsl.flydsl_flash_attn_fp8_func`, also used by ATOM.
- Extension lives in this repository; installed vLLM/AITER source is not patched.
- Current source under test: `e4cf47a7650321e4bd40c83746088a2679ec0509`.
- Current smoke: [40643563020](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40643563020). Artifacts confirm all eight workers, 20/20 successful requests and 22,665,373 KV tokens. Full replay [40646139769](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40646139769) was dispatched automatically; results are pending.

| Stage | Current candidate |
|---|---|
| Loader, scheduler, weight quantization and KV storage | Native vLLM recipe, Moonshot checkpoint, online FP8 targets |
| DCP and latent KV expansion | BF16 kv_b weights make pinned native fused compact-FP8 gather/expansion eligible; the smoke verified the actual route on all eight workers |
| Expanded K/V | BF16; direct FP8 expansion is not implemented |
| Causal FMHA | Native BF16 |
| Cached FMHA with max Q < 512 or max KV < 8192 | Native BF16 |
| Larger cached FMHA | Fused dynamic Q/K/V quantization, then FlyDSL FP8 |
| Query reuse | Identical Q views within one forward only; always cleared in finally |
| Output and LSE | BF16 output and FP32 natural-log LSE; native chunk merge |
| Main ASM PS buffers | Disabled before allocation; draft native PS can remain |
| Decode and draft | Native vLLM AITER |

- FP8 kv_b online weight override was removed: approximately 36 MiB/GPU raw weight cost over 24 MLA layers.
- TP8 has **12 heads/rank**: Q/K head size192; V head size128.
- Complete cached-attention diagnostic: Q1024/KV8192 improved 1.656 including quantization; Q256/KV4096 regressed 0.827.
- QKV preparation costs about55µs in those diagnostics; KV-only preparation still costs about52µs.
- Full throughput and 23M KV pool are not established by standalone kernel timing.
- Numerical RMS drift is roughly45% versus unquantized FP64 reference; no model accuracy claim.

## Validation and continuation

- Kernel run40642162788 passed numerical, strided, empty-KV, same-forward reuse and changed-input graph tests.
- Current prefix smoke runs the latest numerical preflight before model loading.
- Actual `vllm bench serve --backend vllm` uses prefix_repetition at32K/128K prefixes, suffix1024 and short outputs.
- Gate:20 successful requests, zero failures, increased cache hits, KV-capacity line, and all-eight-worker BF16causal / FP8cached / disabledmainPS / Qreuse / nativeFP8gather markers.
- `utils/atom_attention/continue_benchmark.py` polls every60seconds, gates artifacts and dispatches the authorized full replay.
- Full uses **warmup10 per lane,3600seconds**, no GSM8K, and the smoke-tested source SHA.
- The controller avoids duplicate dispatches; an unmet gate requires diagnosis before another full run.

## Evidence boundaries

- Old all-FP8 smoke40634885348:22,695,620 KV tokens,11,789.99 total and61.13 output tokens/sec/GPU; measured requests476success/2empty-response errors.
- Old warmup also had85empty-response errors; job success alone does not establish acceptable request quality.
- Primary baseline39239112271 is Moonshot,15,476.47 total tokens/sec/GPU; historical AMD baseline labels are inaccurate.
- Communication, expansion GEMM and offload costs are not yet isolated. Roofline work remains on hold.
- See the [overhead audit](kimi-prefill-overhead-audit.md) for exact matrix flow, costs and remaining constraints.
