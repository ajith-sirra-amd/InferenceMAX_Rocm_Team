# 原生 FP8 运行时验证 — 2026-11-02

本报告更新此前文档中“运行时验证待完成”的状态。
PR [#64288](https://github.com/vllm-project/vllm/pull/64288) 仍为草稿，
版本为 `492080779e9feea19308092b0aba0840e87c6115`。

## 完整回放已完成

[41413372088](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41413372088)
工作流成功完成，GPU job 为 125171486390。
执行版本 `b56b28bb1fc7302f8f35095341165ddea89de4fa`；
工作流版本 `3307b42d65b9d848ff9eb7425b0f62f1429a160e`。

配置为 C48、TP8/DCP8、正常图与路由、3600 秒 agentic 回放、
MTP 合成接受率。FP8 上投影与缓存上下文 FMHA 已启用，
新 token 注意力仍为 BF16。这是当前镜像的优化路径参考结果，
不是 BF16 基线，也不能单独归因于相对历史原型的某项改动。

- **18,244.04135 总 tokens/s/GPU**；汇总测量区间为 **3627.88643 秒**。
- TTFT / token 间延迟覆盖率均为 **100.0%**，门槛保持 **98%**。
- **3834 条 profiling 记录：3823 成功、11 错误（0.29%）**。
  其中 10 条响应无实际内容，1 条请求写入失败，原因链为连接正在关闭。
  这些请求错误的底层原因尚未完全确认。
- 时限及宽限期结束时另外取消了 8 个请求；服务最终排空至 0 个运行请求。
- **BLAS 警告、服务运行错误、traceback、磁盘 quota 错误均为 0**。
  启动期间有共享内存等待提示，随后成功完成启动。
- 八个不同 worker 均在实际回放中执行了 FP8 上投影和缓存 FMHA，
  实际 BLAS 为 `_BlasBackend.Cublas`。
- 八个原生源文件哈希与已有静态/准确率证据完全一致；
  未生成周期性堆栈转储文件。

[完整机器可读证据](../../results/atom-attention/run41413372088/full-replay-evidence.json)
包含 artifact ID、版本、设置和限制。

## 运行时修正的范围

测试脚本设置 `TORCH_BLAS_PREFER_HIPBLASLT=0` 和
`TORCH_BLAS_PREFER_CUBLASLT=0`。此前不导入 FP8 内核的 BF16 独立测试，
在新旧镜像上都复现了 hipBLASLt 错误；显式切换后，两镜像各 112/112
测试通过，新镜像上的当前 PR 原生 GPU 测试 8/8 通过。

新增的周期性 faulthandler 转储已禁用。此前一个 worker 的首次定时
转储被截断，紧接着 EngineCore 关闭。禁用后启动成功，支持该修正，
但不能宣称已证明底层 C 崩溃原因。

最初的回放末段停滞本次未复现；一次成功不能证明此前 BLAS 警告导致了停滞。
原生注意力代码未加入全局 BLAS 切换。保留运行时 workaround 的复现说明，
所有性能对比配置必须使用相同的 BLAS 设置。

## 新镜像准确率验证已启动

[41426504427](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41426504427)
正在执行，GPU job 为 125214526544。
工作流版本 `c047b417fe471c7a3e913544ebd289343f82bd5c`。
执行版本、PR 版本、BLAS 设置不变，定时器保持关闭。

本次为完整 GSM8K，C48/TP8/DCP8、真实 MTP 验证、eager、
512-token 准确率压力路由；分数门槛保持 0.90。
使用固定 digest 的 11 月 1 日新镜像，详见英文报告。

完成准确率检查后继续新镜像完整性能回放。
历史 BF16 参考的绝对 LSE 差值 <0.20 门槛仍未通过；
reviewer 对比、上游 CI 和人工审查仍待完成。
