# vLLM FP8 prefill PR 实现比较

[English](./vllm-fp8-prefill-pr-comparison.md) | **中文**

FMHA 部分与现有 PR 大量重叠。当前实验增加了原生 vLLM FP8 上投影和直接 FP8 K/V
输出。投影及 FMHA 内核均已存在于 AITER；不同之处是 vLLM 集成和验证。

检查版本：#59124 为 `132f6daa6b17cb4000ef58355943bb39339104f2`，#62447 为
`cefacabced64eda821bcb49493cc970056d15429`，已测试插件为
`14250d6c6c17cd54fb7cd480c74c5c4eb5da2e78`。这是源码比较，未在本次基准中重新运行这两个 PR。

| 项目 | PR #59124 | PR #62447 | 当前已测试实现 |
|---|---|---|---|
| FMHA 内核 | AITER FlyDSL FP8 | AITER FlyDSL FP8 | AITER FlyDSL FP8 |
| 启用方式 | 显式选择 `ROCM_AITER_FLYDSL_FP8` | 在 `ROCM_AITER_FA` 中设置 `VLLM_ROCM_MLA_PREFILL_FLYDSL_FP8=1` | InferenceMAX 自定义插件；原生移植尚未提交 |
| 缓存 K/V 输入 | 已展开的 BF16 张量再动态量化 | 已展开的 BF16 张量再动态量化 | FP8 缓存乘原生 FP8 权重，FP32 累加，直接存储 FP8 K/V |
| 原生权重量化 | PR 不修改 | PR 不修改 | 显式配置 `*.self_attn.kv_b_proj: fp8_per_channel`；已验证四种布局 |
| K/V 标度 | 每个缓存块实际 K/V 的最大绝对值 | 每个缓存块实际 K/V 的最大绝对值 | 使用本次 forward 新 token 的 ATOM gather descale：`2 * max(amax/448, 1e-6)` |
| Q 复用 | 量化完整新-token Q，并复用匹配切片 | 每次 attention 调用都重新量化 Q | 单次 forward 内按指针、形状、stride 精确匹配缓存；不同子视图可能再次量化 |
| 非连续量化输入 | 直接按 stride 读取 | 先将 Q/K/V 变为连续 | QKV 量化器支持 stride；缓存 K/V 直接以 FP8 输出，省去独立量化 |
| 新-token causal attention | FP8，并显式绕过竞争的 ASM 路径 | 进入该后端时使用 FP8；外层 ASM 路由不变 | BF16 |
| 缓存 attention 路由 | 支持正长度 Q/KV，空输入或 int32 超限时回退 | 按开关和支持的形状启用 | 正常 query >=512、context >=8192；accuracy stress 仅覆盖正长度限制 |
| GSM8K | 作者报告两组均为 96.21% | 作者说明尚未运行 GSM8K/lm-eval | 97.119%，1281/1319；八个 worker 均有原生 FP8 执行证据 |

不能说这两个 PR 强制 `kv_b_proj` 权重保持 BF16。模型或量化配置本来就可能选择 FP8
权重。它们缺少的是 FP8 缓存、FP8 权重、FP8 输出之间的直接融合及连接；仅打开权重
量化并不提供这项集成。

## 数值与性能证据

三者均采用 per-tensor E4M3，但路由、用于计算标度的张量范围和舍入细节不同。
这些 PR 的准确率不能证明当前集成正确，也不能证明输出逐字节相同。

- #59124 在随机样本上与 BF16 比较：cosine >0.998、最大 LSE 差 <0.25、平均差 <0.01；
  共享 SDPA 测试使用 FP8 专用 `rtol=2**-4, atol=0.3`。模型分数由作者报告，
  配置不同，包括不使用推测解码。
- #62447 对相同 FP8 输入反量化后的 FP32 参考进行验证：LSE `atol=1e-3, rtol=0`，
  输出相对范数误差 <=0.04。这将内核计算误差与输入量化误差区分开。
- 当前原生投影及相同 FP8 输入的算术测试通过，完整 GSM8K 与执行证据也通过。
  捕获的 Kimi 样本仍未通过原有 BF16 绝对 LSE <0.20 标准；没有放宽阈值。

性能数字不能直接横比：#59124 在无推测解码配置中报告 +4.5%；#62447 在两节点
分离式部署中报告 +1.5%。其引用的 #62444 只修改 CPU-offload LRU，不修改投影。
既有 16915.72154 tokens/s/GPU 结果使用 BF16 `kv_b_proj`；完整运行 40695099743
正在测量原生 FP8 投影的性能。

## 上游范围和后续机会

再提交一个动态 QKV 量化加 FP8 FMHA 的 PR 会与已有工作大量重复。合适的扩展是
在现有 FP8 后端上集成原生 FP8 `kv_b_proj` 和直接 FP8 输出。#59124 更接近这个
基础，因为它具备 Q 复用、stride 量化和显式 ASM 绕过；但需要解决合并冲突。
保留当前已测试的 BF16 causal 路由和 gather 标度策略需要明确设计，不能直接声称等价。

还发现两个具体机会：
1. 像 #59124 一样复用完整 Q 的切片。由子视图标度变为完整 Q 标度可能改变 FP8 数值，
   因此需要重新做数值和模型准确率验证。
2. 新-token causal attention 使用 BF16 时，当前标度准备仍会写入未被使用的临时
   FP8 K/V。如果能保持标度和 Q 输出完全相同，只生成 Q 和 K/V 标度可以省去这些写入。
当前完整运行没有启用这两项修改。

用户指定的新原生开关名称遵循 vLLM 风格：

```bash
export VLLM_ROCM_USE_AITER_FLYDSL_FP8_PREFILL_ATTN=1
```

默认值为 `0`。权重量化仍通过 `--quantization-config` 显式选择。这是原生移植的
拟用名称，已测试插件中尚未提供该名称。用户 fork 的 main 与上游 main 均为
`0beab6c251e8046c0c6cf2fef114cd23b58cda74`；当前工作尚无上游 PR。

源码链接、测试链接及版本信息见[英文页面](./vllm-fp8-prefill-pr-comparison.md#sources)。
另见[当前实现审计](./native-fp8-up-projection-audit_zh.md)和
[GSM8K 证据](../../results/atom-attention/run40688077118/gsm8k_evidence.json)。
