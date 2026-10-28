[English](projection-fp8-experiment.md)

# BF16 投影直接输出 FP8 K/V 的实验

状态：GPU 内核实验已成功完成，压力诊断超过数值门限。服务端尚未选择这些内核，测量结果见下文。

已检查 ATOM 提交 [773eb362](https://github.com/ROCm/ATOM/blob/773eb36254e43060bfd5fabcab7d3840bf591358/atom/model_ops/attention_mla.py#L1675)。
缓存 FlyDSL 投影使用调用方提供的输出 scale。新 token 的 QKV 量化器按
`2 * max(new_token_descale, 1e-6)` 生成缓存 gather descale，避免扫描缓存投影输出求最大值。
如果缓存值超出估计范围，就会发生截断。

已检查的 AITER FlyDSL 投影要求 FP8 权重及其 scale。
Kimi 的 `kv_b_proj` 是未量化的 BF16 权重，因此 ATOM 使用 Triton 回退路径，输出仍为 BF16。
检查时 AITER main 中的包装代码相同。

本实验改造 AITER Triton 的 flat gather/projection 内核，保持 BF16 投影操作数和 FP32 累加：

| 模式 | 投影次数 | 完整 BF16 中间张量 | Scale 来源 |
| --- | --- | --- | --- |
| 现有 native + dynamic quantization | 1 | 有 | 当前缓存 K/V 最大值 |
| `fused_amax` | 1 | 有，但减少一次最大值扫描 | 投影内融合计算最大值 |
| `recompute` | 2 | 无 | 第一次计算最大值，第二次直接输出 FP8 |
| `supplied_scales` | 1 | 无 | ATOM 新 token scale，留出 2 倍范围 |

所有实验模式都在寄存器内保留原有 BF16 舍入，再转为 FP8。
ATOM FlyDSL 的直接输出路径没有这次 BF16 舍入；本实验保留它，以隔离中间张量存储和 scale 的影响。
即便投影仍为 BF16，supplied_scales 的缓存量化数值也不同于当前动态 scale 路径。

## 运行

使用固定镜像 `vllm/vllm-openai-rocm:nightly-1b2a666391f2a24228a1436b64d547c34cc6b6ce`：

```bash
PYTHONPATH="$PWD/upstream/InferenceX/benchmarks/single_node/agentic/atom_attention" \
ATOM_USE_FLYDSL_FP8_PREFILL_ATTN=1 \
python3 utils/atom_attention/projection_experiment.py
```

现有 `.github/workflows/e2e-tests.yml` 的 `runner-diagnostics-only` job 也会运行本实验。
该 job 分配一个节点，在临时容器内执行 GPU 检查，不加载 Kimi，也不启动模型服务。
必须固定 workflow 和 checkout ref；Actions job 日志保留 JSON 证据。

## 验证和证据

- 动态模式与 native 投影和现有量化器逐字节比较 FP8 输出和 scale，覆盖空输入、尾部、重排和 12/24 heads。
- 手工可计算的常量投影，以及独立 Torch supplied-scale 量化参考。
- 修改输入和 scale 后的 graph replay。
- Q512/KV8192 调用真实插件并确认 FP8 路由；比较 BF16 和 FP8 输出，同时使用原始输入及 FP8 反量化输入的独立 FP64 attention 参考。
- 报告相对 RMS、最大绝对误差、LSE 误差和非有限值。初始合成数据门限沿用 preflight：RMS <0.10、LSE 绝对误差 <0.20，不代表模型质量标准。
- 使用真实 ATOM 量化器从独立的新 token K/V 生成 scale。缓存范围放大 4 倍的压力用例单独报告截断及误差，不作为通过的准确率证据。
- KV8192、16384、65536 的 graph timing 比较所有候选及现有路径，另计新 token 量化开销；不包含模型执行和 DCP 通信。

此前 GSM8K 通过时，固定混合阈值选择的是 BF16 attention，不能验证这些内核。
采用候选前仍需内核结果、真实模型 scale/截断证据、服务集成及新的受控运行。

## GPU 结果：2026-10-28

[Run 40666434375](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40666434375)
已成功完成，测试提交为 `8e7bb0eab2195ef2778994806040efbcd067196d`。
[结构化证据](../../results/atom-attention/run40666434375/projection_summary.json) 和
[数值/耗时日志](../../results/atom-attention/run40666434375/numerical-and-timing.log) 已保存。
Recipe 和 launcher 验证通过。

新增 BF16 投影输出逻辑通过空输入、尾部、重排、手工常量、零值及修改输入/scale 的 graph replay 检查。
动态模式的 FP8 字节和 scale 与现有路径完全一致。
使用相同 supplied scale 时，直接 FP8 输出与 native BF16 投影后独立 Torch 量化的结果完全一致。

Q512/KV8192 的真实插件输出了 `ATOM-FP8-ENGAGED`。
现有 FP8 FMHA 相对 BF16 FMHA 的输出 RMS 误差为 3.655%。
相对独立 FP64 参考，BF16 为 0.191%，FP8 为 3.650%。
使用 FP8 反量化输入的 FP64 参考时，FMHA 算术本身的 RMS 误差为 1.157%。
相对原始输入 FP64 参考的 FP8 LSE 最大绝对误差为 0.01509。
这些是张量误差，不是模型答题准确率。

普通合成输入中，ATOM supplied scale 路径相对 BF16 的 RMS 误差为 3.770%，没有截断。
缓存范围放大 4 倍的压力诊断中，K/V 截断比例分别为 1.688%/0.715%，输出 RMS 误差为 19.875%，超过 10% 诊断门限。
这说明 scale 复用存在范围假设，并不是同 scale 投影等价性失败；job 成功不代表压力诊断准确率通过。

下表为 12 heads 的单次耗时，单位微秒：

| 缓存 tokens | 现有投影 + 量化 | 融合最大值 | 重算 | Supplied scale | Supplied + 新 token 准备 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8192 | 93.929 | 86.341 | 126.141 | 60.537 | 66.303 |
| 16384 | 146.367 | 137.705 | 190.534 | 92.875 | 98.765 |
| 65536 | 540.723 | 539.351 | 581.940 | 272.595 | 278.511 |

Supplied-scale 投影阶段在不计准备时加速 1.55–1.98 倍，单个缓存 chunk 计入完整新 token 量化开销后为 1.42–1.94 倍。
精确 scale 重算出现回退；融合最大值在 8K/16K 有小幅收益，64K 基本持平。

这是共享主机上的 graph 微基准，日志中还存在其他容器和 queue-interception 警告。
这些结果不能代表服务吞吐提升，也不包含模型和 DCP collective。

后续只验证本次引入的 BF16 输出逻辑、scale 传递和混合路由。
下一步检查真实 Kimi 新 token 与缓存张量的范围，再进行服务性能比较。
复用的 ATOM FMHA 算法保持原样，SA 固定 GSM8K 仍作为模型评估。生产服务尚未切换到候选路径。
