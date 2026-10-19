# Kimi AttnFP8 checkpoint 测试

[English](kimi-attnfp8-smoke.md)

基础：chore/sa-agentx-v1.0，commit 79cb30ed32023bc543655e1a13f99fe7c1104519。
模型：amd/Kimi-K3-Quark-MXFP4-AttnFP8。C48 TP8/DCP8 DSpark MTP k4。
保留 offload 与调度设置。镜像：
vllm/vllm-openai-rocm:nightly-1b2a666391f2a24228a1436b64d547c34cc6b6ce。
每 lane 预热 1 次，回放 600 秒，禁用分析器。

使用 checkpoint 自带量化，不传入原配方的在线量化覆盖。
不安装旧 BF16-to-FP8 context ASM 转换。记录配置与真实 Q/K/V 数据类型，
不转换张量或更改 attention 调度。缺失探针锨点或真实输入证据时终止。
BF16 输入不能证明 FP8 FMHA。兼容性、精度和内核选择需 runner 验证。

当前仅离线准备：网络禁止 GitHub/Hugging Face，工作区写入被拒绝，未提交运行。
发布前刷新远程源码，检查模型配置、镜像兼容性和 nodes:1 标签。
仅发布到 InferenceMAX_Rocm_Team，使用完整 SHA 提交，检查启动和精度后再考虑完整基准。

网络已恢复，远程源码与本地基础一致。模型卡确认 MXFP4 experts 和 PTPC-FP8 self_attn；FMHA 输入类型需运行证据。不取消现有运行，新增一个排队运行；会话活动时每 20 分钟检查状态。
