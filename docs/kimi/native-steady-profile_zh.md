# 原生 Kimi 稳态性能分析

[English](native-steady-profile.md)

使用仓库根目录的 `e2e-tests.yml`，设置 `native-steady-profile: true`、精确的 `native-vllm-sha` 和 `native-comparison: native`，运行完整的 3,600 秒 agentic replay。默认关闭性能分析；评估、路径探测和参考 PR 运行不允许启用此模式。

工作流明确传入 300 秒稳定等待时间、20 秒采集窗口、5 秒轮询间隔，以及进入测量阶段的 7,200 秒截止时间。控制器在服务就绪并完成 CCD 绑定后启动，等待 benchmark 的测量阶段标记，并确认生成 token 计数在稳定等待期间持续增长。

vLLM PyTorch profiler 记录 worker 的 CPU 和 GPU 活动，关闭 frontend、调用栈、shape 和内存分析。trace 使用 gzip 压缩，保存到仓库内的 `/outputs/results/native_steady_profile`。保留现有计算图、缓存 FP8 路径、MTP 配置和 CCD 绑定。

正常结束或收到中断信号时，控制器停止分析，将失败写入 `profile-control.json`，并恢复输出文件的所有权。recipe 等待控制器并传播其退出失败。即使 replay 失败，GitHub 仍将 trace 和控制器日志上传为 `native-steady-profile-<run_id>`。

选择下一项优化前，应检查所有八个 GPU worker 的实际事件、非零 kernel 执行时间，以及与持续生成阶段的时间重叠。仅有文件不能证明 kernel 覆盖或负载组成稳定。此运行用于诊断；profiler 开销使其吞吐量不适合作为普通性能基线。

CPU 测试覆盖阶段计时、不完整日志行、生成停滞、缺失阶段、服务退出、输出目录限制、SIGTERM 清理及启动响应丢失。GPU 验证需在运行结束并完成 trace 分析后记录。
