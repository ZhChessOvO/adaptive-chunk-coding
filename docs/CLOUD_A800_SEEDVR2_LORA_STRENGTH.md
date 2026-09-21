# A800 SeedVR2 LoRA 推理强度比较

## 为什么还要做这一步

满强度 LoRA 已经证明训练有用：37 条合并 LPIPS 和 PSNR 都比冻结 SeedVR2 更好。但固定图也
显示，它会把一部分细纹理压得过平，UVG 的平均 LPIPS 因此几乎持平并略差。这里不重新训练，
只调 LoRA 残差在推理时占多大比例，寻找“去掉 codec 伪影”和“保留生成纹理”的折中。

## 固定协议

- 继续复用上一轮全部 30 条 REDS + 7 条 UVG QP8 fresh decode、裁剪和随机 seed；
- 冻结 codec、SeedVR2 原权重和同一个 1000-step adapter；
- 复用已经验证的强度 0（冻结版）与强度 1 输出，只新跑 0.25／0.50／0.75；
- 每个强度都使用同一进程中的同一份模型和 adapter，只改变 LoRA residual multiplier；
- 报告合并、REDS、UVG、UVG 适配序列和两条未训练序列；
- LPIPS 为主，同时看 PSNR、时序、逐样本方向和固定图；ShakeNDry 与 HoneyBee 作为上一轮
  已知的失败／成功诊断样本提前加入视觉集合；
- REDS 与 UVG 等权的相对 LPIPS 只作一个便于阅读的排序，不是硬门槛，最终还要看固定图。

## 单卡运行

正式目录：

```text
/root/autodl-fs/DCVC/runs/a800_seedvr2_lora_strength_20260921
```

入口：

```bash
tmux new-session -d -s a800_seedvr2_lora_strength \
  'cd /root/autodl-tmp/adaptive-chunk-coding && \
   bash demo/run_stage_c_a800_seedvr2_lora_strength.sh'
```

runner 在新输出出现前冻结 strength 列表、基础结果摘要、checkpoint SHA 和 37 条输入。三个
强度均逐样本原子落盘，中断后先核对摘要再跳过。每分钟记录三个强度的进度、显存、墙钟和
三块盘；正式 PNG、表格、日志与资源快照只写文件存储。

## 结果与选择（2026-09-21）

三个新强度全部完成，共新增 111 组输出。0 和 1 直接复用上一轮已验证结果；新冻结版与旧
结果 37/37 逐像素一致，五档使用相同输入、seed 和 adapter。

| 强度 | 合并 LPIPS | 较冻结版 | REDS 变化 | UVG 变化 | LPIPS 改善数 | PSNR 变化 | 时序变化 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.445078 | -0.018049 | -0.021432 | -0.003546 | 34/37 | +0.484 dB | -0.402 |
| **0.50** | **0.429974** | **-0.033152** | **-0.038976** | **-0.008192** | **34/37** | **+0.783 dB** | **-0.608** |
| 0.75 | 0.429520 | -0.033606 | -0.039650 | -0.007703 | 31/37 | +0.827 dB | -0.473 |
| 1.00 | 0.444932 | -0.018194 | -0.022806 | +0.001568 | 24/37 | +0.646 dB | +0.027 |

LPIPS 和时序变化越低越好，PSNR 变化越高越好。REDS 与 UVG 等权的相对 LPIPS 诊断中，
0.50 为 -5.1596%，0.75 为 -5.1574%，差异只有 0.0022 个百分点，不能据此说 0.50 在均值
上明显胜出。真正让我们选择 **0.50** 的原因是它更稳：LPIPS 改善数是 34/37 而不是
31/37，时序平均最好，并且在 Jockey、ShakeNDry、YachtRide 三个已知过平滑序列上退化
更轻。0.75 稍高的 PSNR 不足以抵消这些风险。

固定图也显示从 0.25 到 1.00 是连续增强的平滑效果：Beauty、HoneyBee、ReadySetGo 随
强度增加继续受益；Jockey、ShakeNDry、YachtRide 则越来越平。0.50 位于两类趋势之间，
不是用硬门槛选出的“唯一最优”，而是后续系统实验最实用的单一默认值。

正式运行耗时 1,471 秒，只观察到 GPU 0；峰值 `nvidia-smi` 16,875 MiB，指标汇总峰值
CUDA allocated 550,817,792 B。结果目录有 2,023 个普通文件，最终 `du -sb` 为
683,888,444 B；完成时 `/root`、`/root/autodl-tmp`、`/root/autodl-fs` 分别约使用
13%／17%／32%。`run.complete` 存在，`run.failed` 不存在。

正式目录：

```text
/root/autodl-fs/DCVC/runs/a800_seedvr2_lora_strength_20260921
```

下一步不重训 router，也不改变 spatial-QP 码流。保持 codec、v6 route、输入和 seed 不变，
把 0.50 接回 Generate ROI 和 17 帧重叠长视频路径的检查也已完成：同一条 33 帧码流上，
LPIPS、PSNR、整体时序、空间边界和时间切换均改善，非 Generate 像素逐像素不变。因此
0.50 保留为后续 teacher 的恢复强度。完整结果见
[`CLOUD_A800_SEEDVR2_LORA_ROI_LONG.md`](CLOUD_A800_SEEDVR2_LORA_ROI_LONG.md)。
