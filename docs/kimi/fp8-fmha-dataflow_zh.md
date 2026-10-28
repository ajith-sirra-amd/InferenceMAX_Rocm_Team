# FP8 FMHA：BF16 投影权重与 FP8 注意力张量

[English](fp8-fmha-dataflow.md) | [当前审计](kimi-prefill-overhead-audit_zh.md) | [ATOM 优化机会](atom-vllm-opportunities-2026-10-28_zh.md)

本文对应 serving checkout e4cf47a7650321e4bd40c83746088a2679ec0509 的大型缓存前缀路径。新 token 和小缓存块仍使用 BF16 注意力。完整运行结果尚未产生。

**关键区别：kv_b_proj 在 FMHA 之前重建 K/V。FMHA 不读取这些 BF16 投影权重。** 压缩 KV 缓存、投影权重和展开后的注意力 K/V 是三个不同对象。

| 对象 | 当前类型 | 每个 TP rank 的形状 |
|---|---|---|
| 压缩 KV 缓存 | FP8 E4M3 | 每 token：512 个 latent 值 + 64 个 RoPE 值 |
| kv_b_proj 权重 | BF16 | [3072,512] |
| 展开 K | BF16，再转换成 FP8 | [Tkv,12,192] |
| 展开 V | BF16，再转换成 FP8 | [Tkv,12,128] |
| 查询 Q | BF16，再转换成 FP8 | [Tq,12,192] |
| Q/K/V descale | FP32 | 当前量化调用中每个张量各一个标量 |
| 注意力输出 / LSE | BF16 / FP32 | [Tq,12,128] / [12,Tq] |

Tkv、Tq 是本次缓存块或查询调用打包后的 token 数。12 是此 TP8 配置每个 rank 的本地 prefill head 数。

**颜色：** 绿色 = FP8；蓝色 = BF16；黄色 = 类型转换；紫色 = FP32 scale/LSE。

```mermaid
flowchart TD
    C["Compressed KV cache<br/>FP8"]
    G["DCP all-gather<br/>FP8 payload"]
    W["kv_b_proj weights<br/>BF16"]
    R["Fused dequantization + up-projection<br/>BF16 operands"]
    B["Expanded K and V<br/>BF16"]
    Q["Query Q<br/>BF16"]
    KQ["Quantize expanded K/V"]
    QQ["Quantize Q; reuse within forward"]
    K8["K8 and V8<br/>FP8"]
    Q8["Q8<br/>FP8"]
    S["sQ, sK, sV<br/>FP32 descales"]
    F["FMHA with FP8 Q/K/V<br/>Higher-precision internal accumulation"]
    O["Attention output<br/>BF16"]
    L["LSE for chunk merge<br/>FP32"]
    C --> G --> R
    W --> R
    R --> B --> KQ --> K8 --> F
    Q --> QQ --> Q8 --> F
    KQ -.-> S
    QQ -.-> S
    S --> F
    F --> O
    F --> L
    classDef fp8 fill:#DCFCE7,stroke:#166534,color:#052E16,stroke-width:2px
    classDef bf16 fill:#DBEAFE,stroke:#1E40AF,color:#172554,stroke-width:2px
    classDef conversion fill:#FEF3C7,stroke:#92400E,color:#451A03,stroke-width:2px
    classDef fp32 fill:#EDE9FE,stroke:#5B21B6,color:#2E1065,stroke-width:2px
    class C,G,K8,Q8,F fp8
    class W,B,Q,O bf16
    class R,KQ,QQ conversion
    class S,L fp32
```

## 1. 用 BF16 权重重建 K/V

下标 16、8 分别表示 BF16 和 E4M3 FP8。C16 表示根据压缩缓存及其 descale 恢复的 latent。逻辑投影为：

$$
Z_{16}=\mathrm{BF16}(C_{16}W_{\mathrm{kv},16}^{T}).
$$

C16 为 [Tkv,512]，权重为 [3072,512]，其中 3072=12 x (128+128)。Z16 重排为 [Tkv,12,256]，拆成 128 维的非 RoPE K 和 128 维 V。再向每个 K head 拼接缓存中 64 维 RoPE：

$$
K_{16}=\mathrm{concat}(K_{\mathrm{NoPE},16},K_{\mathrm{RoPE},16}),\qquad V_{16}=V_{\mathrm{projected},16}.
$$

实际内核融合了缓存行读取、反量化、投影及拼接；这些逻辑公式不表示每一步都有独立中间缓冲。

## 2. 将展开张量量化成 FP8

对 X 属于 {Q,K,V} 分别计算：

$$
s_X =
\begin{cases}
\max|X_{16}|/448 & \max|X_{16}|>0 \\
10^{-6} & \text{otherwise}
\end{cases}
$$

$$
X_8=\mathrm{E4M3}\left(\mathrm{clip}(X_{16}/s_X,-448,448)\right),\qquad X_{16}\approx s_XX_8.
$$

448 是 E4M3FN 的最大有限值。实现使用 FP32 倒数及舍入。这里的 descale 属于展开张量，不是压缩 KV 缓存的 descale。Q8 及其 scale 只在同一不可变 forward 内复用，每个缓存 K/V 块单独准备。

## 3. FP8 FMHA 的逻辑数学

对一个 head：

$$
S\approx\alpha s_Qs_KQ_8K_8^T+M,\qquad P=\mathrm{softmax}(S),\qquad O_{16}\approx\mathrm{BF16}(s_VPV_8).
$$

Alpha 是模型配置的 softmax scale；M 是 mask；P 是注意力权重。以上是逻辑运算，不包含所有指令和中间舍入。FlashAttention 使用分块 online softmax，不物化完整 S/P 矩阵。审阅的 AITER FP8 实现也会将概率 tile 转换用于 FP8 矩阵乘法；FP8 输入不代表累加器和 softmax 中间结果全部采用 FP8。API 返回 BF16 输出和自然对数形式的 FP32 LSE，vLLM 使用它们合并分块结果。

## 要移除的额外开销

当前：重建 BF16 K/V -> 写入 -> 量化再次读取 -> 写 FP8 K/V -> FMHA。

目标：重建 -> 在输出阶段缩放并舍入 -> 直接写 FP8 K/V -> FMHA。

BF16 投影权重在数学上不强制要求 BF16 输出存储。保留 BF16 权重并添加 FP8 输出阶段是可行的内核设计方向，但当前候选没有实现。仍需获得正确的输出 scale。

上述形状中，展开 K/V 的逻辑存储为每 token、每 rank 7680 字节 BF16，或 3840 字节 FP8。这不是实测 HBM 流量，也不是 KV 缓存容量收益承诺。

## ATOM 如何直接输出 FP8

审阅 ATOM 661ef7661eef0d804d2b52d81ed41d80aeaf9fdb，以及 AITER bbef5a6f2d3733b293c80d2cee6a8d464e88bc01。后者只是源码审阅版本，不能证明当前镜像安装了这个 AITER 版本。

1. 仓库的 ATOM 启动脚本请求全局 ptpc_fp8 在线权重量化。layers.*.self_attn.kv_b_proj 的排除模式匹配没有前缀的 draft 树；主模型使用 model.layers.*，不被此模式排除。
2. 可选 FP8 prefill 路径先量化新 token 的 Q/K/V。缓存输出 descale 使用 max(新 token 的 descale, 1e-6) x 2。
3. DCP 传输压缩 FP8 缓存行；融合 gather/reconstruction 完成行重排、投影、拆分及 RoPE 拼接。
4. 当已安装 FlyDSL API 和布局支持时，ATOM 传入 k_out_scale、v_out_scale，在现有 BF16 workspace 的 FP8 view 中直接写 FP8，返回 K8,V8,sK,sV 给 FMHA，跳过单独的缓存 K/V 量化。
5. 此 FlyDSL 路径要求兼容的 FP8 缓存及 FP8 投影权重/scales。BF16 权重走 ATOM 的 Triton fallback。当前固定版本 vLLM 的原生融合路径则要求连续 BF16 投影权重，仅修改权重类型会使它选择 fallback。
6. 直接输出不计算缓存块的 amax。“新 token scale 乘二”是一项 scale 策略，不是所有缓存值都不会溢出的证明。投入 serving 前应检查缓存异常值、全零或小范围新 token、RoPE 列及完整注意力输出。
7. 审阅的 ATOM 脚本没有启用 ATOM_USE_FLYDSL_FP8_PREFILL_ATTN，其默认值为 0。不能在没有运行证据时，把该能力归因于历史 ATOM 性能结果。

AITER 的独立计时 helper 先从同一份重建测试输入计算 scale，再复用 scale 测融合路径。这证明已提供 scale 时的内核行为，不证明 ATOM 在真实缓存前缀上采用的新 token scale 乘二策略正确。

完整源码链接见 [English 文档的 Source anchors](fp8-fmha-dataflow.md#source-anchors)。
