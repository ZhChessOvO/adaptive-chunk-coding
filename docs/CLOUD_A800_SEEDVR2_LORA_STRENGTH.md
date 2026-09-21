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

完成后先选一个实用强度，再把它接回 Generate ROI 和 17 帧重叠长视频路径。这个小比较不会
重训 router，也不会改变 spatial-QP 码流或已完成的满强度 LoRA 消融。
