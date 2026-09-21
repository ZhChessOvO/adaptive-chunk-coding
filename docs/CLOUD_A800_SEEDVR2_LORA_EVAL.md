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
