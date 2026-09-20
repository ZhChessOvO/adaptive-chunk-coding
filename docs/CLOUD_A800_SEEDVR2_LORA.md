# A800 SeedVR2 codec-artifact 适配

## 1. 这一步要解决什么

Generate 区域先用很低码率的 DCVC-UF QP8 编码，再交给 SeedVR2 恢复。现有 SeedVR2
是通用视频恢复模型，并没有专门见过我们选定 codec 产生的伪影。因此 codec checkpoint
确定后，下一步让 SeedVR2 适应真实 QP8 解码结果，而不是只在通用权重上反复调整 router。

这一步是性能增强，不会覆盖“冻结 SeedVR2 也能运行”的迁移结果。冻结版始终保留为基线。

## 2. 为什么使用 LoRA，而不是直接改 3B 全模型

公开的 SeedVR2 仓库提供了推理结构与权重，但没有发布可直接运行的训练入口。我们明确实现
自己的单卡适配协议：

- 冻结 3B DiT 和 VAE 的全部原权重；
- 只在最后 8 个 Transformer block 的注意力与 MLP，以及输出投影中加入 rank-8 LoRA；
- 一共 41 个低秩模块、2,822,656 个可训练参数；
- adapter 单独保存，推理时叠加在原 BF16 DiT 上，大 checkpoint 不复制进 Git。

最后 8 层足以调整输出风格，同时前 24 层不建立反向图，适合一张 A800。它也比全参数微调
更不容易把 REDS 风格强行写进整个生成模型。

## 3. 训练输入从哪里来

先等待 codec 权重插值选出 alpha。随后固定这个 codec，把已有训练／适配窗口重新做真实
QP8 编解码：

- REDS train 的 500 个 17 帧、512×512 窗口；
- UVG adaptation 的 60 个窗口；
- 每条码流先写到临时文件并重新读回，再解码；ZIP 或旧 teacher 图像不需要重新下载；
- 冻结 VAE 分别把原视频和 QP8 解码视频编码成确定性的 posterior-mode latent；
- 只长期保存 BF16 latent，预计不到 1 GiB，不保存 560 份重复 PNG 和码流。

训练时从完整 latent 随机裁 256×256 对应区域。UVG 被采到的概率固定为 25%，高于它的自然
样本占比，目的是吸取刚才 codec 微调在 UVG 上过拟合的教训，而不是把 REDS 数量优势原样
带进 SeedVR2。

## 4. 它具体学习什么

部署使用 SeedVR2 的单步恢复，所以训练也对齐这条路径：在最大噪声时刻输入噪声、真实
QP8 条件 latent 和固定正文本 embedding，让 DiT 预测从干净 latent 到噪声的 velocity。

损失保持简单：

```text
loss = velocity MSE
     + 0.1 × clean-latent L1
     + 0.2 × temporal-delta MSE
```

第一项是原 rectified-flow 目标；第二项约束单步还原；第三项要求相邻 latent 帧的变化也接近
原视频，减少逐帧看起来不错但播放时抖动。它不是新的感知模型或额外 teacher。

## 5. 运行与断点

先跑 2-step 烟测，确认真实 cache、梯度、adapter 保存和显存：

```bash
tmux new-session -d -s a800_seedvr2_lora_smoke \
  'cd /root/autodl-tmp/adaptive-chunk-coding && \
   bash demo/run_stage_c_a800_seedvr2_lora.sh smoke'
```

烟测通过后跑 1000 steps：

```bash
tmux new-session -d -s a800_seedvr2_lora_train \
  'cd /root/autodl-tmp/adaptive-chunk-coding && \
   bash demo/run_stage_c_a800_seedvr2_lora.sh train'
```

正式目录为 `/root/autodl-fs/DCVC/runs/a800_seedvr2_lora_v1_20260920`。latent cache 逐样本
原子保存；训练每 25 steps 原子覆盖恢复点，样本、裁剪、翻转、各项 loss、梯度、耗时和
显存逐 step 写 JSONL。runner 每分钟记录 GPU、墙钟和三块盘，重启后会跳过已完成 cache，
并从最近恢复点继续。

## 6. 完成后怎样比较

不设置硬性通过门槛。固定选择少量 REDS 与 UVG 窗口，在完全相同的 codec 输入和随机噪声
下比较：

- 冻结 SeedVR2；
- 冻结 SeedVR2 + 训练后的 LoRA；
- LPIPS、PSNR、时序误差和固定视觉图；
- 最后再放回 Generate ROI 与重叠长视频路径，检查边界和播放稳定性。

如果 adapter 只改善 REDS 而伤害 UVG，就像 codec 一样保留它作为域内适配消融，不强行设为
默认；如果两类数据的整体折中更好，就用它重新生成 teacher，再重训 router。
