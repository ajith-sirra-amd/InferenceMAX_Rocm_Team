# ATOM Kimi 对比与 vLLM 性能候选项 — 2026-10-28

[English](atom-vllm-opportunities-2026-10-28.md) · [当前审计](kimi-prefill-overhead-audit_zh.md) · [基线](atom-attention-BASELINES.md)

## 范围与已有证据

已检查本仓库的 Kimi ATOM 启动脚本、ATOM 版本 `661ef7661eef0d804d2b52d81ed41d80aeaf9fdb` 和运行中的 vLLM 版本 `1b2a666391f2a24228a1436b64d547c34cc6b6ce`。源码支持某路径，不等于历史镜像实际启用了该路径。

完整性能 run40646139769 已成功：总吞吐 16,915.72154、输出吞吐 117.84290 tokens/s/GPU，相对 baseline39239112271 提升 9.30% 和 14.35%。这是完整候选配置的结果，不能归因于 FMHA 单项。详见[结果与证据](fp8-fmha-results-2026-10-28_zh.md)。GSM8K run40658068830 已通过：strict-match 为 1282/1319（97.1948%），flexible-extract 为 1281/1319（97.1190%），均超过 90%。没有 FP8 FMHA engagement marker；验证的是 GSM8K 使用 BF16 路径的服务配置，并非 cached FP8 kernel 的模型准确率。

**重要更正：** 检查的 ATOM 版本中，`ATOM_USE_FLYDSL_FP8_PREFILL_ATTN` 默认值为 **0**，Kimi 脚本没有设置它。FlyDSL gather 开关默认开启。FP8 FMHA 是可选功能；没有镜像环境和实际调用证据，不能把 ATOM 测试的吞吐归因于 FP8 FMHA。

## 值得验证的差异

| 候选项 | ATOM 证据 | 当前 vLLM 证据 | 下一步 |
|---|---|---|---|
| C48 关闭 speculative decoding | C48/56/72 明确不加载 draft | C48 使用 DSpark k4，主基线也是 k4 | 固定其他条件做 C48 no-draft 对照，验证当前 batch 下 draft/verification 是否仍划算；收益未知。 |
| 专用 cached-KV quantizer | Q 准备后使用专用 contiguous K/V quantizer | 仓库已有 `fused_kv_per_tensor_quant`，但 adapter 和计时仍调用 generic QKV 的 SKIP_Q 分支 | 先在实际 chunk 尺寸比较数值和完整调用耗时；仅对 contiguous K/V 考虑专用路径，strided 输入保留原路径。 |
| Producer 与量化融合 | SiTU→down_proj、sigmoid gate→o_proj、Q norm→q_b_proj、strided f_a→f_b_proj 可直接传递量化数据与 scale | 固定版本对应调用仍存在 activation 与 FP8 consumer 边界；native input AttnRes+quant 已有，不能重复计算收益 | 一次选择一个 producer，验证真实 consumer 接口和 producer+GEMM 总时间；旧 producer 实验不是当前 serving 的验证。 |
| 重建直接输出 expanded FP8 K/V | 支持的 FlyDSL gather epilogue 向 FMHA 返回 K8/V8 与 descales | Native fused gather 传输 compact FP8，但展开输出 BF16，之后再次量化 | 属于更大的内核集成；先验证镜像 API、权重布局与 scale、row mapping、cached-value scale 和数值。 |
| Mixed/prefill graph 覆盖 | ATOM recipe 请求 FULL | vLLM 因无 compiled submodule 且禁用 breakable graphs，降为 FULL_DECODE_ONLY | 检查受支持的 breakable/piecewise 路径；直接复制 FULL 不证明兼容或有收益。 |
| CPU offload NUMA locality | ATOM 在分配前绑定 worker 和首选 NUMA memory，启用 LMCache NUMA | Native offload 先分配、清零 CPU tensor，再 pin；脚本在 ready 后绑定 CCD，smoke 实际绑定1562线程 | 先测实际页分布；若不匹配，按本机 GPU 拓扑在分配前绑定，不复制其他机器的 rank/socket 列表。 |
| Recurrent state 存储与 checkpoint | 脚本选择 FP16 state，关闭 demand checkpoint，保留 prompt-end anchors | 固定版本 Kimi getter 选择 FP32 recurrent state；hybrid planner 使用1536-token block | 需要 state kernel 与 checkpoint 验证；getter 未传 SSM dtype override，单个 CLI flag 不能复现 ATOM。 |
| Graph memory 和 cache grouping | State/cache 组织不同 | 最终 graph capture6.67GiB/GPU；较早 sizing capture9.20–9.64GiB；Mamba page padding3.97%，另有 grouping 上界警告 | 测量真实分配与 graph dispatch；26.09% 是 planner 上界，不是全局浪费实测；不靠关闭记账制造容量提升。 |

最新大 chunk 诊断：QKV 准备约58.7µs，复用 Q 后的 KV 准备约52.9µs，FP8 FMHA 本体约51.6µs。这些数值说明减少准备开销值得验证，但没有测出它在完整模型中的占比。

直接 FP8 重建不能直接替换现有 BF16-weight native fused 路径：检查的 FlyDSL 实现要求受支持的 FP8 权重/scale 和输出 view。Cached K/V 的 scale 也要验证；new-token scale 不是全部历史 cached tensor 的通用上界。

两个 recipe 都请求 C48 下 TP8/DCP8、8192 batched tokens、96 sequence seats、FP8 KV cache、AITER SiTUv2/stage2 和广泛的 projection/shared-expert 在线 FP8 权重。两者都设置 INT4 quick reduction，但 vLLM 明确设置16MiB下限；名称相同不等于阈值和路由相同，修改前要检查实际 collective 消息大小。ATOM 的 communicator reuse 只在 C56及以上启用，因此不是遗漏的 C48 开关。

## 与真实基线比较

主基线39239112271 同样使用 DSpark k4、FULL_DECODE_ONLY、1536-token blocks，并出现相同 padding 警告。这些共同开销本身不能解释 hybrid 与基线之间的差异。

| Runtime 证据 | Baseline39239112271 | Hybrid prefix smoke40643563020 |
|---|---:|---:|
| 模型加载显存/GPU |187.43GiB|184.08GiB|
| GPU KV tokens |20,951,355|22,665,373|
| 最终 graph capture/GPU |6.96GiB|6.67GiB|
| 模型加载时间 |239.5–240.4s|439.4–440.3s|

Smoke 是实际调用测试，不是完整吞吐对照；加载时间不计入 measurement window。基线原始记录为 **warmup101成功/430错误**，**profiling3357成功/6错误**；不能把436条总错误都算成 benchmark phase 失败。

旧 aggregate JSON 即使实际为 DSpark k4，也会写 `spec_decoding: none`；speculation 来源以真实 server command 和 engine config 为准。

## 完整 run 后的决策

1. 确认 checkout、模型、真实命令、warmup10、3600秒 profile 和全部必要路径。
2. 同时比较 total/output tokens/sec/GPU 与15476.46881/103.05251，并检查 TTFT/ITL、完成请求数、分阶段错误、GPU/CPU hit rate、真实 KV pool 和 reporting duration。
3. 运行健康时选择一个独立实验：C48 no-draft 是较简单的系统对照；专用 K/V quantizer 是较小的代码候选，但先做数值与完整调用计时。
4. 若请求错误或 runtime failure 主导，先排查；若 workload mix/cache hits 差异明显，要明确说明。
5. 不在一次对照中同时更改 speculation、NUMA、graph 和 kernel。Full run40646139769 已启动，不重复 dispatch。

当前 recipe 拒绝 `SPEC_K=0`。No-draft 需要显式 caller/recipe 分支，省略 speculative config 并调整 graph sizing；目前不能靠设置0实现。

源码链接及固定版本见 [English source anchors](atom-vllm-opportunities-2026-10-28.md#source-anchors)。
