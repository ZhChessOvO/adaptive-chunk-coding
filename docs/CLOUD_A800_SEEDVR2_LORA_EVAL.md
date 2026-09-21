# A800 SeedVR2 冻结版与 LoRA 固定比较

## 目的

LoRA 的训练 loss 已经下降，但这不能说明显示视频一定更好。本轮只回答一个问题：在完全
相同的低码率 codec 输入和随机噪声下，LoRA 是否比冻结 SeedVR2 更接近原视频、更稳定。

## 固定范围

- 复用此前 REDS 30 条 + UVG 7 条联合评估中的 all-Generate spatial-QP QP8 真实码流；
- 码流、fresh decode PNG、帧数、裁剪和每条样本的 SeedVR2 seed 都不改变；
- 新跑一遍冻结 SeedVR2，再加载正式 1000-step LoRA 跑一遍；
- 新冻结输出必须与此前冻结输出逐像素一致，否则停止解释 LoRA 结果；
- 报告合并、REDS、UVG，以及 UVG 五条适配序列与 ReadySetGo／YachtRide 两条未参与
  LoRA 训练的序列；
- 主指标是 LPIPS，同时报告 PSNR、时序误差、逐样本方向和固定效果图；
- 这 37 条都已参与此前研究，只是开发／误差分析，不包装成新的独立证据；不设置硬性门槛。

这是全画面 Generate 后端的直接检查，先排除 router 与 ROI 几何变化。若 LoRA 的综合结果
合适，再把候选放回 v6 Generate ROI 和长视频重叠路径；若不合适，则保留为消融，并在固定
同一批输入的前提下考虑其他生成／恢复模型。

## 运行与恢复

正式目录：

```text
/root/autodl-fs/DCVC/runs/a800_seedvr2_lora_eval_20260921
```

入口：

```bash
tmux new-session -d -s a800_seedvr2_lora_eval \
  'cd /root/autodl-tmp/adaptive-chunk-coding && \
   bash demo/run_stage_c_a800_seedvr2_lora_eval.sh'
```

runner 先在看到新 LoRA 输出前冻结 37 条样本、输入摘要、checkpoint SHA 和随机种子。冻结版
与 LoRA 版均逐样本原子保存，中断后跳过已验证样本。每分钟记录 GPU、累计时间和三块盘；
正式结果、PNG、日志与资源快照只写文件存储，大文件不进入 Git。

## 正式结果（2026-09-21）

两版都完成 37/37 条。新冻结版与旧冻结版 37/37 逐像素一致，因此下面的变化确实来自
LoRA，而不是输入、随机噪声或推理代码变化。LPIPS 和时序误差越低越好，PSNR 越高越好。

| 分组 | N | 冻结 LPIPS | LoRA LPIPS | LPIPS 变化 | LoRA 胜出 | PSNR 变化 | 时序变化 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 合并 | 37 | 0.463127 | 0.444932 | -0.018194 | 24/37 | +0.646 dB | +0.027 |
| REDS | 30 | 0.490812 | 0.468006 | -0.022806 | 21/30 | +0.300 dB | +0.082 |
| UVG | 7 | 0.344476 | 0.346044 | +0.001568 | 3/7 | +2.128 dB | -0.209 |
| UVG 适配序列 | 5 | 0.387976 | 0.394763 | +0.006787 | 2/5 | +2.051 dB | +0.005 |
| UVG 未训练序列 | 2 | 0.235728 | 0.224249 | -0.011479 | 1/2 | +2.321 dB | -0.745 |

合并 LPIPS 相对下降约 3.93%，中位逐样本变化也是 -0.028906；33/37 条 PSNR 更高，21/37
条时序误差更低。LoRA 仍保持了 Generate 相对 QP8 codec 的感知收益：合并 LPIPS 为
0.444932，而 codec 输入为 0.555543；但 codec 输入的 PSNR 和时序误差仍更好，这正是低码率
生成恢复的感知质量与逐像素失真之间的取舍。

UVG 不是简单的整体退化。Beauty、HoneyBee 和未参加 LoRA 训练的 ReadySetGo 的 LPIPS
明显变好；Bosphorus 接近持平；Jockey、ShakeNDry 和 YachtRide 变差。其中 ShakeNDry
是主要失败样本（LPIPS +0.150916）。固定图显示满强度 LoRA 通常会压掉冻结模型的锐化与
幻觉纹理，这能改善许多 REDS、Beauty 和 ReadySetGo 画面，但在 Jockey、YachtRide 等
细纹理／快速运动场景会显得过平。这和训练中的 latent MSE／L1 目标相符。

因此不把结果解释为“LoRA 全面胜出”，也不丢弃它。后续相同 37 条输入和 seed 上的
0.25／0.50／0.75 扫描已经选择 0.50：它与 0.75 的均值几乎打平，但逐样本和时序更稳。
下一步把这个固定强度接回 Generate ROI 和 17 帧重叠长视频路径检查边界与播放稳定性；
详见 [`CLOUD_A800_SEEDVR2_LORA_STRENGTH.md`](CLOUD_A800_SEEDVR2_LORA_STRENGTH.md)。

正式运行墙钟 1,055 秒；`nvidia-smi` 峰值 16,875 MiB，只观察到 GPU 0。完成后目录真实
`du -sb` 为 474,001,928 B，三块盘约为 13%／17%／32%。`run.complete` 存在，
`run.failed` 不存在；完整表、逐样本 CSV 和固定图位于正式目录的 `formal/`。
