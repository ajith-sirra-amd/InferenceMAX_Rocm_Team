# Kimi-K3：问题与拟议 KV 流程

[English](./kimi-kv-optimized-flow.md) | **中文**

## 问题

| 完整benchmark | Total tokens/s/GPU | KV-pool tokens |
|---|---:|---:|
| [Moonshot参考39239112271](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/39239112271) | 15,476.47 | 20,951,355 |
| [AMD + 单独FP8 gather适配39601846940](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/39601846940) | 15,258.78 | 23,098,919 |

- AMD容量增加**10.25%**，total throughput降低**1.41%**。
- 目标：KV pool接近**23M**，恢复原生融合context处理。
- 假设：单独cache decode/reorder及重建准备产生额外开销。
- 比较结果不能证明具体操作造成性能差距。

## 当前缺点

| 步骤 | AMD + 我们的单独gather适配 | 缺点 |
|---|---|---|
| Cache | FP8压缩KV | 保留 |
| All-gather | FP8字节，uint8 collective | BF16一半payload，保留 |
| Decode/reorder | 单独生成BF16 latent rows | 中间HBM流量和launch |
| 重建 | 单独FP8-weight kv_b GEMM及准备 | 无法使用完整原生融合 |
| Packing/FΜHA | BF16 K/V、BF16 FMHA | FP8 weights/cache不代表FP8 FMHA |

- **FP8 all-gather不要求BF16 kv_b**，通信不读取该权重。
- **固定版本融合重建内核要求连续BF16 kv_b且无独立weight scales**。
- AMD FP8 kv_b无法通过该eligibility检查。

## 要求的方向

| 问题 | 答案 |
|---|---|
| FP8 all-gather本身要求BF16 kv_b吗？ | **不要求**，collective只传输字节，不读取投影权重 |
| 固定版本原生融合context路径要求BF16 kv_b吗？ | **要求**，同时还有dtype/layout/metadata检查 |
| 本实验方向？ | **保留BF16 kv_b → 满足原生融合eligibility → 该路径执行FP8-byte gather及融合重建** |
| 选择FP8 gather会把kv_b转换为BF16吗？ | **不会** |
| FP8 gather可配合FP8 kv_b吗？ | **可以**，AMD适配已证明，但重建需单独处理 |

- BF16 kv_b是该原生融合实现的必要条件，单独满足并不充分。
- 需实际uint8 collective及fused-execution日志证明执行。

## 拟议解决方案

| 组件 | Moonshot实验 | 目的 |
|---|---|---|
| Model | moonshotai/Kimi-K3 | BF16 attention起点 |
| Expert weights | 保留checkpoint MXFP4 | 保留压缩 |
| 其他attention projections | Online fp8_per_channel | 节省权重内存，FP8 GEMM |
| kv_b_proj | **保留BF16，不加入targets** | 启用原生融合重建 |
| Convolutions/shared experts/residual projections | 保留checkpoint精度 | 避免意外量化 |
| KV cache/collective | FP8 cache，原生FP8-byte gather | 减少存储和通信 |
| Reconstruction/packing | **一个原生融合内核** | 避免单独中间处理 |
| FMHA/output | BF16 | 保留现有attention kernel |

```text
FP8 compressed KV cache
 → FP8-byte all-gather
 → ONE kernel: reorder + dequant + BF16 kv_b projection + K/V packing
 → BF16 K/V + BF16 Q
 → BF16 FMHA
 → BF16 output
```

- Targets：self_attn下的in_proj_qkvgfab, f_b_proj, g_a_proj, g_b_proj, o_proj, fused_qkv_a_proj, q_b_proj, q_proj, kv_a_proj_with_mqa, g_proj。
- 首次smoke选择**303 attention linears**；仅证明选择，未完成量化。
- 不安装AMD weight-conversion patch。
- 精度方案接近AMD，kv_b保留BF16；online量化值可能不同。

## 收益与代价

| 预期收益 | 剩余代价/不确定性 |
|---|---|
| Attention weights小于参考 | Dynamic activation quantization仍有开销 |
| 原生融合重建 | Projection及cache dequant仍必要 |
| 更少中间HBM/dispatch | 需实测throughput |
| KV pool可能接近23M | BF16 kv_b及runtime buffers占用内存，不保证容量 |
| 保留BF16 FMHA | 新增projection quantization需accuracy验证 |

## 为什么必须重建

| 张量/操作 | TP8本地维度 |
|---|---|
| Latent C | [T,512] |
| RoPE key | [T,64] |
| Cache row | [T,576] |
| kv_b weight | [3072,512] |
| Reconstruction | C × W_kv_b^T → [T,3072] |
| Packed K | [T,12,192] |
| Packed V | [T,12,128] |

- T为chunk token数；cache rows分布在DCP ranks。
- 重建是**up-projection**，不只是head rearrangement。
- Full-head FMHA不能直接使用压缩latent。
- Dequantization无法恢复quantization loss。

## 后续候选

| 方案 | 流程变化 | 要求 |
|---|---|---|
| Fused FP8-weight reconstruction | FP8 cache/gather → fused FP8-weight reconstruction → BF16 K/V → BF16 FMHA | 内核支持FP8 weights/scales |
| FP8 FMHA | 重建直接输出scaled FP8 K/V，准备scaled FP8 Q → FP8 FMHA → BF16输出 | Accuracy及完整调用性能验证 |

- 直接FP8输出仍执行量化；融合可避免BF16中间张量。
- 高精度accumulation不能恢复量化损失。
- Preparation/padding/launch可能抵消FP8 FMHA收益。

## 验证与状态

- Smoke：**C48 TP8/DCP8，warmup1/lane，600秒**。
- 全8 GPU证据：实际FP8 projection weights、24 BF16 kv_b、eligibility、uint8 collective、fused execution。
- 测量KV pool、total/output throughput、request errors。
- 单独real-verification accuracy；synthetic throughput不能证明accuracy。
- Fastsafetensors通过后：相同smoke移除loader flag。

| Run | 结果 |
|---|---|
| [39717243502](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/39717243502) | 全8 numerical preflight通过；GPU4 loading HSA resource exhaustion、free memory0 MB、nogds701；无pool/throughput结果 |
| [39718556543](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/39718556543) | Fastsafetensors queue限1重试；本修订时结果待定 |

- Loading失败不能证明GPU硬件故障。
- Queue限制是诊断修改，尚未证明解决问题。

## 来源

- [AMD model card](https://huggingface.co/amd/Kimi-K3-Quark-MXFP4-AttnFP8)
- [固定版本MLA backend](https://github.com/vllm-project/vllm/blob/1b2a666391f2a24228a1436b64d547c34cc6b6ce/vllm/v1/attention/backends/mla/rocm_aiter_mla.py)
- [固定版本gather/reconstruction](https://github.com/vllm-project/vllm/blob/1b2a666391f2a24228a1436b64d547c34cc6b6ce/vllm/v1/attention/ops/rocm_aiter_mla_prefill.py)
