# Kimi-K3：权重量化、KV容量与融合attention

[English](./kimi-kv-optimized-flow.md) | **中文**

## 1. 模型与问题

| Checkpoint | 量化内容 | kv_b_proj重建权重 |
|---|---|---|
| **Moonshot Kimi-K3**：moonshotai/Kimi-K3 | Expert weights MXFP4；online overrides前attention projections及shared experts BF16 | BF16 |
| **AMD Quark量化Kimi-K3**：amd/Kimi-K3-Quark-MXFP4-AttnFP8 | Experts MXFP4；attention weights/input arithmetic FP8；shared experts BF16 | FP8 + per-channel scales |

- 相同Kimi-K3架构；attention weight与KV cache quantization是不同配置。
- 两种benchmark都显式使用FP8 KV cache。
- FP8投影权重不代表FP8 Q/K/V输出；当前context FMHA输入BF16。

| 完整run | Total tokens/s/GPU | KV-pool tokens |
|---|---:|---:|
| [Moonshot参考39239112271](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/39239112271) | 15,476.47 | 20,951,355 |
| [AMD Quark + 我们的FP8-byte通信适配39601846940](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/39601846940) | 15,258.78 | 23,098,919 |
| 差值 | **−1.41% throughput** | **+2,147,564 tokens / +10.25%容量** |

- Moonshot参考仅online量化in_proj_qkvgfab，多数其他attention projections为BF16。
- 通信适配是我们修改runtime，先gather已有FP8 cache字节，再decode；不是AMD checkpoint的一部分。
- 问题：AMD容量增加但throughput略低。
- 目标：保留compact weights/cache并恢复融合context处理。
- 此比较不单独证明容量收益或throughput差距的原因。

## 2. 为什么量化权重重要

| 权重 | FP8收益 | 代价 |
|---|---|---|
| 大attention projections | 约BF16一半weight bytes加scales；可能加速GEMM | Dynamic activation quantization及accuracy验证 |
| kv_b_proj | 更少重建权重存储及流量 | 当前原生融合重建不支持FP8 weights/scales |
| Expert weights | 两checkpoint已有MXFP4 | Export/runtime allocations可能不同 |

- 权重占用减少可为KV cache释放内存。
- **+2.15M KV tokens是整个AMD配置的测量结果，不能归因于kv_b alone。**
- TP8 kv_b：每GPU24个[3072,512]矩阵。
- FP8替代BF16约节省**36MiB/GPU**，未扣scales/layout开销。
- 其他attention weights另有收益；workspaces、graphs、memory profiling决定最终pool。
- 恢复BF16 kv_b牺牲这部分小收益，以使用当前已有融合。

## 3. 当前处理及缺点

| 步骤 | Moonshot参考原生融合 | AMD Quark + 我们的适配 |
|---|---|---|
| Cache | FP8 | FP8 |
| Communication | 原生FP8-byte gather | 适配FP8-byte gather |
| Decode/reorder | 融合内核内完成 | 单独生成BF16 latent |
| kv_b | BF16 | FP8 + scales |
| Reconstruction/packing | 一个融合操作 | 单独FP8 GEMM及preparation/packing |
| FMHA | BF16 | BF16 |
| 主要代价 | 更大attention weights | 更多中间处理/dispatch |

- 单独处理可能增加HBM和launch，是候选瓶颈，尚未证明整个1.41%差距的原因。
- 原AMD fallback：FP8 cache → BF16 decode → BF16 gather → reconstruction。

## 4. FP8 gather与BF16 kv_b关系

| 问题 | 答案 |
|---|---|
| FP8 gather本身需要BF16 kv_b？ | **不需要**，只传cache字节 |
| 固定版本原生融合路径需要BF16 kv_b？ | **需要**，连续BF16、无weight scales及其他eligibility条件 |
| 本实验方向 | **BF16 kv_b → 原生融合eligibility → 该路径执行FP8 gather及融合重建** |
| Gather会转换kv_b吗？ | **不会** |
| FP8 gather可配合FP8 kv_b？ | **可以**，AMD适配已证明，重建单独处理 |

## 5. 当前方案：Moonshot选择性FP8

| 组件 | 设置 | 目的 |
|---|---|---|
| Model | Moonshot Kimi-K3 | BF16 attention起点 |
| Experts | 保留MXFP4 | 压缩 |
| 其他attention projections | Online fp8_per_channel | 接近AMD权重收益 |
| kv_b_proj | **保留BF16，省略target** | 使用原生融合 |
| Convolutions/shared experts/residual projections | 保留checkpoint精度 | 保留exclusions |
| Cache/communication | FP8/native FP8-byte collective | 节省内存/payload |
| Reconstruction | 融合reorder/dequant/BF16 projection/packing | 避免单独中间处理 |
| FMHA/output | **BF16** | 保留现有kernel |

```text
FP8 compressed KV cache
 → FP8-byte all-gather
 → ONE kernel: reorder + dequant + BF16 kv_b projection + K/V packing
 → BF16 K/V + BF16 Q
 → BF16 FMHA → BF16 output
```

- self_attn targets：in_proj_qkvgfab, f_b_proj, g_a_proj, g_b_proj, o_proj, fused_qkv_a_proj, q_b_proj, q_proj, kv_a_proj_with_mqa, g_proj。
- 首smoke选择303 linears，未证明量化完成；不安装AMD conversion patch。
- 精度方案接近AMD，但kv_b BF16；量化值可能不同。
- **23M pool及性能提升仍是目标，不是已验证结果。**

## 6. 优先改进：FP8 kv_b的融合重建

| 方案 | 收益 | 要求 |
|---|---|---|
| 现有BF16-weight融合重建 | 消除单独reorder/decode/projection/packing | BF16权重存储 |
| **支持FP8 weights/scales的融合重建** | 保留AMD compact kv_b，同时减少中间处理 | 实现并验证FP8-weight/scaled-input支持 |
| 后续FP8 FMHA | 可能降低attention compute/bandwidth | Scaled FP8 Q/K/V、kernel支持、accuracy及完整调用测试 |

```text
FP8 cache → FP8-byte gather
 → fused reorder + scaled FP8-weight reconstruction + packing
 → BF16 K/V → BF16 FMHA → BF16 output
```

- 兼顾收益需将支持FP8 kv_b的融合重建接入DCP路径。
- 当前AITER已有per-row FP8 scale支持；固定vLLM传入空weight scale并限制BF16 eligibility。
- Prototype `fp8_fused_reconstruction_probe.py`比较FP8/BF16融合及数值reference；尚未接入serving或GPU验证。
- Serving需处理preshuffled weights、cache scales及custom op；prototype使用unshuffled weights。
- **FP8 FMHA可选**，不是FP8 gather或消除单独重建准备的必要条件。
- 内核可内部decode；FP8-weight支持不代表所有算术FP8。
- Fully FP8候选：融合输出scaled FP8 K/V并准备scaled FP8 Q → FP8 FMHA → BF16输出。
- 直接FP8输出仍量化；融合可避免BF16中间张量。
- Conversion/padding/launch可能抵消收益；高精度accumulation不能恢复量化损失。

## 7. 维度与验证

| 张量/操作 | TP8本地维度 |
|---|---|
| Latent C / RoPE key | [T,512] / [T,64] |
| Cache row | [T,576] |
| kv_b | [3072,512] |
| Reconstruction | C × W_kv_b^T → [T,3072] |
| Packed K/V | [T,12,192] / [T,12,128] |

- 重建是up-projection，full-head FMHA不能直接使用512 latent。
- T为chunk tokens，cache rows分布于DCP ranks。
- Smoke C48 TP8/DCP8 warmup1/lane600秒。
- 全8 GPU：FP8投影权重、24 BF16 kv_b、eligibility、uint8 collective、实际fused execution。
- 测pool/throughput/errors；准确率另行real-verification。

| Smoke | 状态 |
|---|---|
| [39717243502](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/39717243502) | Fastsafetensors，preflight全8通过；GPU4 loading失败，0MB free,nogds701 |
| [39718556543](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/39718556543) | Queue1仍GPU0 loading失败，198MB free |
| [39719727532](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/39719727532) | 默认loader诊断，本修订时结果待定 |

- Loading失败无pool/throughput测量，不证明硬件故障。

## 来源

- [AMD model card](https://huggingface.co/amd/Kimi-K3-Quark-MXFP4-AttnFP8)
- [Moonshot checkpoint](https://huggingface.co/moonshotai/Kimi-K3)
- [固定版本MLA](https://github.com/vllm-project/vllm/blob/1b2a666391f2a24228a1436b64d547c34cc6b6ce/vllm/v1/attention/backends/mla/rocm_aiter_mla.py)
- [固定版本gather/reconstruction](https://github.com/vllm-project/vllm/blob/1b2a666391f2a24228a1436b64d547c34cc6b6ce/vllm/v1/attention/ops/rocm_aiter_mla_prefill.py)

## 2026-10-22 ???????

- **perf branch ????GPU ???????**
- AMD ?????? FP8 kv_b ??? channel scales???? BF16 ???
- ???FP8 cache ? uint8 all-gather ? fused ????? rank scale?FP8 ?????K/V packing ? BF16 K/V ? BF16 FMHA?
- ??????? AITER flat kernel???????? AITER??? AITER preshuffled ?????? unshuffle?
- ????8 GPU ???????? rank scales??????? mask??? checkpoint ???? FP32 reference?? cached-prefix HTTP ????? DSpark ????? GSM8K?
- ? lane replay warmup?**0**?fast override?**0**????? profiling ? graph capture ????
- FP8 fused execution ?????????? GSM8K ?????????????????????

| Moonshot ??39719727532 | ?? |
| --- | --- |
| Loader | ?? loader ?????? |
| KV pool | 22,534,301 tokens |
| Profiling throughput | 11,641.85674 total tokens/s/GPU?60.77441 output tokens/s/GPU??600? smoke |
| Warmup errors | 85/99 records?? streamed responses |
| Profiling errors | 4/478 records =0.84%?? streamed responses |
| Native fusion / transport | ?8 GPU ?? BF16 kv_b fused reconstruction ? FP8 ?? gather |
| ?? | ????? smoke pass???????? |
| ?? warning | 280? hipBLASLt internal-error fallback??????????????? |

- ??????? HTTP200???? server traceback??? accuracy gate ?? generation/streaming ???

## 2026-10-22 ????? Moonshot

- ?????????**??? moonshotai/Kimi-K3**?
- ?? baseline target?`*in_proj_qkvgfab: fp8_per_channel`?
- **???** `*.self_attn.kv_b_proj: fp8_per_channel`??? attention ???? BF16??? checkpoint expert ???
- kv_b ??????????cache latent ??? FP8?fused context reconstruction ??? activation ???
- Loader probe ??? GPU24? FP8 kv_b linears???? online PTPC ????? channel scales?
- FP8 cache ? byte all-gather ? fused FP8-weight reconstruction ? BF16 K/V ? BF16 FMHA?
- ???? accuracy?? lane replay warmup0?KV pool ??????????
- ???? AMD run39726650098 ???????????????

## o_proj 增量精度实验

- 在 Moonshot 两 target 实验上仅新增 `*.self_attn.o_proj: fp8_per_channel`。
- Target：`*in_proj_qkvgfab`、`*.self_attn.kv_b_proj`、`*.self_attn.o_proj`。
- 加载后要求全8 GPU FP8 o_proj 权重和 channel scales，保留 BF16 FMHA。
- 先运行数值、cached-context 和 GSM8K 检查，再测吞吐。与 run39729399329 比较；精度和内存收益尚未证实。

## Moonshot loader probe 修正

- run39729399329 加载后失败，原因是 probe 拒绝原生 FP32 KDA conv1d。未运行 serving/GSM8K。
- 允许未量化 BF16/FP32 conv1d，拒绝 FP8 卷积和 quantization scales。
- fused kernel 和 targets 不变。run39729912013 也包含旧 probe，修正后使用含 o_proj 的三 target 实验。
