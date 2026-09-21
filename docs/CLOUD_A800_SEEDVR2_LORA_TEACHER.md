# SeedVR2 LoRA 0.50 teacher 重建与 router 重训

## 为什么要做

上一阶段已经确认：同一个 SeedVR2 LoRA 在推理时取 `0.50` 强度，比冻结 SeedVR2
更适合作为 Generate 恢复器，而且接回 ROI 和 33 帧连续视频后仍然改善画质与时序。
但是当前 router 仍由旧的冻结 SeedVR2 标签训练。换句话说，执行器已经变好，router
却还不知道“哪些区域现在更值得 Generate”。本阶段只更新这部分认知。

## 固定方案

- 仍只使用一张 A800 80GB；DCVC-UF 继续选择冻结官方 checkpoint，不使用过拟合的
  codec 微调权重。
- 训练样本不变：REDS train 500 个窗口，加 Beauty、Bosphorus、HoneyBee、Jockey、
  ShakeNDry 的 60 个 UVG 适配窗口，共 560 个 17 帧、512×512 窗口。
- 每个窗口仍划分为 4×4、共 16 个区域，因此共有 8960 个区域标签。
- SeedVR2 基础 DiT、VAE 与 LoRA adapter 都冻结，只在推理时使用 LoRA 强度 `0.50`。
- 重新计算 Generate 图像，以及它对应的局部 LPIPS、PSNR、时序误差和相对 Base 收益。
- Base、Enhance、编码端特征、字节标签和 ROI 耗时不受 LoRA 影响，直接逐字段复用并做
  相等性检查，不重复做没有变化的实验。
- router 继续采用 v5 的“v1 锚点 + 两个专家保守共识”，置信度、预算与训练日程保持
  不变，只用新标签重训专家权重。之后继续使用 v6 已选定的空间一致性系数 `λ=0.004`。
- 不设置新的硬性晋级门槛。先看新路由相对旧 v6 改了什么，再在固定 37 条 REDS + UVG
  视频上做真实码流和画质比较。

## 为什么仍要重做 QP8，而不直接读 LoRA 训练 cache

LoRA 训练 cache 来自真实、可 fresh decode 的全 Generate spatial-QP 流，但早期 teacher
使用的是普通标量 QP8 流。二者都合法，却不是同一份重建。2026-09-21 的两样本前置检查
首先尝试直接使用 cache；第一条 REDS 的冻结回放相对旧 teacher 出现最大约 `0.0527`
的区域 LPIPS 差异，因此该路径立即停止，没有写入正式结果。

正式协议改为：每条样本只重新编码旧协议的标量 QP8，真实写盘后读回并 fresh decode，
然后分别运行冻结强度检查和 LoRA 0.50。QP16、QP32、16 个 Enhance tile 与 ROI 计时均不
重复。这样多花很少的 codec 时间，却能保证新旧 teacher 的差异只来自 LoRA。

## 可恢复执行

正式根目录：

```text
/root/autodl-fs/DCVC/runs/a800_seedvr2_lora_teacher_20260921
```

入口：

```bash
tmux new-session -d -s a800_seedvr2_lora_teacher \
  'cd /root/autodl-tmp/adaptive-chunk-coding && \
   bash demo/run_stage_c_a800_seedvr2_lora_teacher.sh'
```

runner 会先冻结包含 560 条样本、checkpoint SHA-256、LoRA 强度与 Git commit 的计划。
每条新 teacher JSON 先写临时文件再原子改名；重启后跳过已经验证的样本。每分钟记录
完成数、显存、GPU 利用率和三块盘空间。正式开始前先完成 REDS、UVG 各一条小样本，
并把 LoRA 强度临时设为 0，要求它在 `1e-5` 容差内复现旧 teacher 的四项区域指标；
不通过就不会继续 560 条长任务。

## 完成后要看什么

1. 新 Generate 在 REDS、UVG 和合并口径下分别改善多少区域；
2. “Generate 比 Base 更好”的区域数量怎样变化；
3. 重训后 37 条视频的 G／B／E 数量、Generate 边界和连通块怎样变化；
4. 最后用新 route、LoRA 0.50 和不变的 codec 做真实落盘码流、fresh decode、消融与
   固定可视化。旧冻结 teacher、旧 router 与旧 v6 都保留为消融，不被覆盖。

## 当前状态

代码自检、REDS 一条和 UVG 一条的两级 smoke 已通过。两条标量 QP8 流分别为 5047 B
和 1088 B，与旧 teacher 记录完全一致；把 LoRA 临时设为 0 后，LPIPS、PSNR、RGB MSE
和时序误差的 16 区域最大差都为 0。两条 LoRA 0.50 输出及三联图也已生成并人工查看。
smoke 峰值 CUDA allocated 约 11.38 GiB，两条含冻结回放共用约 53 秒。正式 tmux 启动后
在本节继续补充进度、显存、总耗时、空间与最终结果。
