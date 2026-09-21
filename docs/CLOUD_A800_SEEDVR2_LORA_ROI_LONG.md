# A800 SeedVR2 LoRA 0.50 ROI 与长视频接入检查

## 这一步回答什么

全画面固定比较已经选择 LoRA 强度 0.50，但实际系统不会总对整张画面运行 SeedVR2。这里把
它放回真实 Generate 连通 ROI 和重叠长视频路径，检查三件事：

1. adapter 是否只改变 Generate 区域，Base／Enhance 像素必须保持逐像素不变；
2. 16 像素空间羽化处是否出现更明显的接缝；
3. 17 帧窗口重叠、router 动作切换处是否出现新的播放抖动。

这不是新 benchmark，而是接入回归。使用的是此前反复分析过的 REDS validation 000 开发
片段，不据此宣称论文泛化优势。

## 固定比较

直接复用已经完成的 33 帧长视频机制实验：

- 同一条 22,707 B spatial-QP 码流，包含 1 个 I + 4 个 P8，codec 参考环不断；
- 同一组随时间变化的 v6 动作图；
- 同一个 fresh decode、三个长度 17／步长 8 的恢复窗口和三个 ROI 组件；
- 同样的 64 像素上下文、1.5 倍处理尺度、16 像素空间羽化与三角时间融合；
- 每个 ROI 组件使用与冻结版相同的确定性 seed；
- 唯一改变是把 SeedVR2 LoRA residual multiplier 从 0 改为 0.50。

正式 plan 在任何新恢复输出出现前冻结旧码流、ROI manifest、冻结版输出、强度扫描摘要和
adapter 的 SHA-256。恢复按 ROI 组件原子落盘；同一个进程只加载一次模型。汇总时再次核对
旧产物哈希和 adapter／强度，并强制验证所有非 Generate 像素逐像素一致。

## 报告内容

- 冻结 SeedVR2 ROI 与 LoRA 0.50 ROI 的 LPIPS、PSNR、时序误差；
- Generate 边界带的梯度误差、跳变和 RGB 误差；
- 17 帧硬窗口切换位置与动作图切换位置的单独时序误差；
- action map、GT、codec、冻结 ROI、LoRA ROI 和放大差异的固定图；
- 33 帧横向比较 MP4、真实码流字节、完整耗时、峰值显存和三块盘。

这里不设置“必须每项都赢”的硬门槛。若 0.50 的整体收益仍在、接缝和切换没有明显恶化，
就保留它进入 teacher 重建；若局部执行暴露出全画面评价看不到的问题，再决定降低强度、
对 ROI 边界单独处理或更换恢复后端。

## 单卡运行

正式目录：

```text
/root/autodl-fs/DCVC/runs/a800_seedvr2_lora_roi_long_20260921
```

入口：

```bash
tmux new-session -d -s a800_seedvr2_lora_roi_long \
  'cd /root/autodl-tmp/adaptive-chunk-coding && \
   bash demo/run_stage_c_a800_seedvr2_lora_roi_long.sh'
```

runner 每分钟记录 3 个 ROI 组件的完成数、GPU、墙钟和三块盘。中断后重新执行同一命令会
先核对 plan、组件 metadata、adapter SHA 和强度，再跳过完整组件。正式结果、比较视频和
资源快照都写入文件存储；codec 和 SeedVR2 原权重保持冻结，只做 LoRA 推理。

## 正式结果（2026-09-21）

正式检查完成，复用了同一条 22,707 B 码流、同一路由、同一 ROI、同一 seed；fresh decode
继续逐像素一致。LoRA 0.50 与冻结 SeedVR2 ROI 的结果如下：

| 输出 | LPIPS | PSNR | 时序误差 |
|---|---:|---:|---:|
| 冻结 SeedVR2 ROI | 0.556755 | 21.2150 dB | 19.9715 |
| LoRA 0.50 ROI | **0.547832** | **21.3565 dB** | **19.7161** |
| LoRA − 冻结 | **-0.008922** | **+0.1415 dB** | **-0.2554** |

这里的收益不是靠改变码流或扩大 Generate 区域得到的。所有非 Generate 像素的最大差值为
0；差异只发生在动作图声明的 Generate 区域内。空间边界带 RGB MAE 下降 0.3801，边界梯度
误差下降 0.0288，边界跳变量完全不变。两个 17 帧硬窗口切换位置的时序误差下降 0.3336，
动作图切换位置下降 0.2760。额外逐帧诊断中，32/32 个相邻帧对的时序误差都下降，没有
发现由 LoRA 引入的单点闪烁峰值。

固定图和 33 帧比较视频显示 LoRA 仍是较保守的去伪影修正；放大差异图在非 Generate 区域
全黑，16 像素羽化边界没有出现新的硬线。因此 **0.50 保留为性能增强版默认强度**。这是
一条开发视频上的接入回归，不把它夸大为长视频泛化结论；它足以说明工程接入没有抵消前一轮
全画面收益。

3 个 ROI 组件推理合计 33.43 秒，和冻结版 33.44 秒几乎相同；LoRA 不增加额外模型调用。
整条正式 runner 墙钟 146 秒，峰值 `nvidia-smi` 15,543 MiB，峰值 CUDA allocated
11,967,425,024 B，只观察到 GPU 0。最终目录 `du -sb` 为 72,267,921 B，三块盘约为
13%／17%／32%。`run.complete` 存在，`run.failed` 不存在。

正式产物：

```text
/root/autodl-fs/DCVC/runs/a800_seedvr2_lora_roi_long_20260921/formal/summary.json
/root/autodl-fs/DCVC/runs/a800_seedvr2_lora_roi_long_20260921/formal/visuals/lora050_long_video_boundary.png
/root/autodl-fs/DCVC/runs/a800_seedvr2_lora_roi_long_20260921/formal/lora050_long_video_comparison.mp4
```

下一步保持 DCVC-UF、动作语法和数据角色不变，用 SeedVR2 LoRA 0.50 重新计算 Generate
teacher，再重训轻量 router；冻结 SeedVR2 teacher 继续保留为迁移基线与消融。
