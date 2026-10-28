# K/V 直接输出 FP8 的验证

[English](./direct-kv-fp8-validation.md) | **中文**

此可选服务实验在符合条件的缓存上下文分块中，消除展开后的 BF16 K/V 分配及其
独立量化步骤。`kv_b_proj` 的权重和操作数仍为 BF16，点积使用 FP32 累加；
最终写入 FP8 前，仍在寄存器中保留原生 BF16 舍入。Q 使用现有 ATOM 动态量化器。
这不是 FP8 投影计算。

同时设置 `ATOM_USE_FLYDSL_FP8_PREFILL_ATTN=1` 和
`ATOM_USE_FUSED_KV_FP8_OUTPUT=1`。直接输出开关未设置或为零时，保留现有投影路径。
缓存分块需满足 max Q >=512、max KV >=8192、原生融合 DCP 条件以及支持的 BF16
权重形状。因果注意力和较小缓存分块仍使用 BF16 FMHA。新 token K/V 提供 ATOM 的
gather descale：先应用现有最小值限制，再乘以二。scale 和量化 Q 仅在单次 forward 内有效。

## 精确参考与舍入诊断

[运行 40668922297](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40668922297)
在 cache scale 为 0.137 时发现，12,582,912 个 V 值中有六个与 Torch 倒数乘法参考
发生舍入差异。候选内核与原生 AITER HIP 静态量化的差异为零；动态版本也与现有量化器完全一致。
例如 BF16 V=0.0001678466796875、descale=0.02455357275903225 时，
FP64 缩放值为 0.0068359371295876326。候选内核和 AITER 得到 0.005859375，
Torch 倒数乘法得到 0.0078125。全局相对 RMS 差异为 3.0243e-08。

更新后的同 scale 精确参考使用 AITER HIP `per_tensor_quant`，仍单独报告 Torch
边界诊断。没有为适配 Torch 参考而修改投影内核，逐字节精确检查继续保留。

## 验证顺序

1. `runner-diagnostics-only=true` 在固定镜像内执行 `direct_kv_diagnostics.py`。
   对 cache scale 0.125 和 0.137，使用独立 FP64 投影测量
   FP8 cache -> BF16 投影 -> FP8 链路误差；随后执行实际 backend 的新 token scale
   准备、缓存 FP8 注意力、Q 复用、状态清理及输入变化后的 graph replay。
2. `attention-path-probe-only=true` 使用 32K 和 128K 共享前缀运行真实 Kimi 请求。
   workflow 设置 `ATOM_VALIDATE_FUSED_KV_FP8_OUTPUT=1`；由于验证使用主机断言，
   recipe 添加 `--enforce-eager`。每次优化缓存调用均与独立同 scale AITER 量化
   精确比较 K/V 字节及注意力输出/LSE，并从当前 BF16 输入重新计算 Q。
   同时相对之前动态 scale FP8 路径及独立 BF16 注意力路径检查相对 RMS <0.10、LSE 最大误差 <0.20，
   并报告 K/V 截断比例。
3. 探针要求全部八个 worker 均具有直接输出和验证成功标记。检查证据后才运行启用 graph 的性能测试。

这些张量门限沿用现有诊断标准，不代表模型准确率。之前的合成压力测试中，缓存范围为
新 token 范围四倍时，RMS 超过门限。因此，ATOM scale 复用仍需真实模型张量验证，
仅复用算法不足以证明数值质量。

workflow 显式提供两个开关，并经 launcher 转发。正常性能执行将验证设为零，不执行
主机比较；另以 graph replay 验证相同的 GPU 操作。

SA 固定 GSM8K 评估保持不变。此前 97.1948% strict 分数运行的是 BF16 注意力，
不能证明此路径或原 FP8 FMHA 路径的准确性。此集成尚无服务数值或性能结论。
现有五项 CPU 路由测试通过，Python 编译和 YAML 解析通过。
Shell 验证及实际 GPU 检查在诊断任务中执行。

初始内核计时和合成误差见[投影证据](./projection-fp8-experiment_zh.md)；
这些计时不包含模型服务与 DCP 通信。


## 已完成的 GPU 预检

[运行 40670162135](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40670162135)，源提交 7e24cd6a5c7eb7d324ea106c76f7a160a840e687，通过同 scale K/V 字节精确验证、实际 backend Q 复用与清理，以及输入变化后的 graph replay。两个 cache scale 均无截断；BF16 投影相对 FP64 的 RMS 误差为 0.136–0.175%，最终 FP8 K/V 相对 BF16 投影为 2.645–2.753%。backend 输出相对原动态 scale 路径的 RMS 为 3.580%（eager）和 4.148%（改变输入后 graph replay）。

[结构化证据](../../results/atom-attention/run40670162135/summary.json)。新增的独立 BF16 注意力比较将在模型 smoke 启动前的直接 K/V 预检中执行；真实模型验证仍待完成。首次运行 40669975181 在数值检查前因诊断导入作用域错误失败，已修复，未修改内核或门限。


完整性能运行可设置 after-run-id 为前置数值运行 ID，ref 必须与其源 SHA 完全一致。workflow 在分配 GPU 前等待前置运行成功；失败时不调度性能任务。空值保留直接执行行为。
