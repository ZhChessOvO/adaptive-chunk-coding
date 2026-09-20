# A800 spatial-QP codec 权重插值

1000-step DCVC-UF 微调不是简单的正结果或负结果：它在 REDS 上明显改善等画质码率，
但在 UVG 上损失细节。直接采用最终权重会把一种风格上的收益换成另一种风格上的退化。

本轮不重新训练，而是固定测试冻结权重与 1000-step 权重之间的三个位置：

```text
W(alpha) = (1 - alpha) * W(frozen) + alpha * W(1000-step)
alpha = 0.25, 0.50, 0.75
```

alpha=0 和 alpha=1 的 110 项结果直接复用，三个中间点各自重新做 37 条固定混合 route
和 6 条样本 × 3 个均匀 QP，共 165 个真实编解码任务。每个任务都要求实际落盘码流、
独立 fresh decode 和逐像素一致。

候选选择不设硬门槛。先在固定的 3 条 REDS + 3 条 UVG 均匀 QP 曲线上，选择平均 LPIPS
BD-rate 最低的 alpha；PSNR BD-rate 只作第二排序。37 条混合 route 用于确认该选择在真实
spatial-QP 用法中的字节、LPIPS、PSNR、时序和边界表现，不反过来临时改 alpha。

这批数据已经参与过开发和跨分布分析，因此只是 codec 候选选择，不包装成新的独立测试。
正式目录为：

```text
/root/autodl-fs/DCVC/runs/a800_spatial_qp_interpolation_20260920
```

运行入口：

```bash
tmux new-session -d -s a800_spatial_qp_interp \
  'cd /root/autodl-tmp/adaptive-chunk-coding && \
   bash demo/run_stage_c_a800_spatial_qp_interpolation.sh'
```

权重、码流、PNG、日志和正式汇总都放文件存储；代码和环境仍在数据盘。runner 每分钟记录
GPU、耗时和三块盘，任务级输出可续跑，最后在日志停止后保存精确普通文件字节快照。
