# A800 spatial-QP-aware DCVC-UF 微调

这一步不是从头训练一个新 codec，也不是马上追求论文表格中的最好数字。它先回答一个更
直接的问题：原始 DCVC-UF 只见过“整张图使用同一个 QP”，而我们的实际码流会在同一张图
里混用 QP 8／16／32；让模型也见过这种输入后，区域交界和码率—画质关系能否进一步改善？

## 与上游训练有什么不同

上游 `train_image.py`／`train_video.py` 面向从头训练，样本中的整张图只抽一个 QP，并使用
多阶段、数十个 epoch 的日程。当前实验从已上传的官方 image 与 HT-S checkpoint 出发，
只做单卡适配：

1. 每步读取一个 17 帧裁剪，即 1 个 I 帧和 2 个 P8 coding unit；
2. 生成一张 4×4 的 B／G／E router 图，再按真实码流规则扩展到 64 像素 syntax cells；
3. I 帧及两个 P8 都通过与正式 codec 相同的 spatial-QP scale selection 前向；
4. 第二个 P8 偶尔改变 0–3 个 router 块，让连续参考状态见过动作图切换；
5. 分别更新 image model 和 HT-S video model，P8 之间保留参考状态，但在每个 P8 后截断
   梯度，控制单卡显存。

QP 档位仍固定为 Generate=8、Base=16、Enhance=32，不在这一步重新搜索。

## 动作图和损失

75% 的训练步使用连通的混合动作图：Generate 为 1–4 块，Enhance 为 3–8 块，其余为
Base。25% 使用全图统一的 Generate／Base／Enhance，作为“复习题”，减少微调破坏原有
均匀 QP 能力的风险。动作图由 `seed + global_step` 唯一决定，因此中断恢复不会改变后续
样本。

损失仍是 DCVC-UF 的 rate + distortion：

- rate 使用模型概率估计，并按区域为 z latent 选择对应 QP 的概率模型；
- distortion 在每个 QP 区域内按上游相同的 YUV／RGB 公式计算，再乘该 QP 对应的
  lambda，最后按面积合并；
- 当整图只有一个 QP 时，新公式与上游公式数值一致，轻量 self-test 会检查这一点；
- P8 的 8 帧继续使用上游 HT-S 的短／中／长时距失真权重。

这让 Enhance 区域承担更强的失真惩罚，Generate 区域更愿意节省码率，同时避免用一个
全局 lambda 把三种区域的目标混在一起。

## 数据角色

- REDS `train/000..239`：训练；
- 已准备的 60 个 UVG adaptation 17 帧窗口：训练，采样概率 10.7%；
- REDS validation、UVG 评估视频：不进入本轮训练，只在微调后做对照。

UVG adaptation 使用的是此前明确划入训练侧的五条序列／时段；用于跨分布评估的材料不因
这次微调改名为“独立测试”。运行时会把可见序列、帧数、尺寸和角色写入
`data_manifest.json`。

## 断点与正式输出

`demo/stage_c_spatial_qp_finetune.py` 保存模型、两个 AdamW optimizer、不可变配置和已经
完成的 step。每个 step 的数据、裁剪、动作计数、loss、估算 bpp、梯度范数、耗时和峰值
显存写入 JSONL。checkpoint 使用临时文件加原子改名；服务器重启后从最后一个已保存 step
继续。

训练参数保持 FP32，前向使用 BF16 autocast；512×512 的 HT-S 路径启用 activation
checkpointing。正式目录只放在 `/root/autodl-fs/DCVC/runs`，代码、环境和编译仍在
`/root/autodl-tmp`。

先在 tmux 中做两步 256×256 烟测：

```bash
tmux new-session -d -s a800_spatial_qp_smoke \
  'cd /root/autodl-tmp/adaptive-chunk-coding && \
   bash demo/run_stage_c_a800_spatial_qp_finetune.sh smoke'
```

烟测结束后会把导出的两个 checkpoint 真正写入 spatial-QP 码流，再用独立进程 fresh
decode，并逐像素核对编码器重建。它只验证梯度、续跑、导出和 codec 兼容性，不用于判断
画质是否提升。

长期版本仍只用一张 A800：

```bash
tmux new-session -d -s a800_spatial_qp_train \
  'cd /root/autodl-tmp/adaptive-chunk-coding && \
   bash demo/run_stage_c_a800_spatial_qp_finetune.sh train'
```

默认长期配置是 512×512、1000 steps、每 25 steps 保存一次，可用 `MAX_STEPS`、
`SAVE_EVERY` 和 `RUN_ROOT` 显式覆盖。先根据真实 step 时间和显存确定是否直接采用默认长度，
这只是资源安排，不设置“达不到某个数字就停止研究”的硬门槛。

## 微调后怎样判断

长期 checkpoint 先在固定开发样本上与原始 codec 做同 route、同 QP 的真实字节和画质
比较，同时检查均匀 QP 是否明显退化。若实现正常，再把新 codec 接回完整 B／G／E 管线，
重新生成 teacher 并重训 router。旧的冻结模型结果仍保留，可用于说明方法在不微调 codec
时也能迁移；微调结果是额外的性能增强，不覆盖那组证据。

训练后的第一轮比较已经预先写成
`demo/run_stage_c_a800_spatial_qp_finetune_eval.sh`，不会看过微调画质再挑样本或 route：

- 复用已有 30 条 REDS + 7 条 UVG 账本，对 37 条 v6 combined 动作图分别用冻结／微调
  checkpoint 真实编码；
- 固定 3 条 REDS + 3 条 UVG，在 QP 8／16／32 做均匀质量回归；
- 110 个任务全部要求真实码流、独立 fresh decode 和逐像素一致；
- 汇总 LPIPS、PSNR、时序误差、字节和 Generate 边界，并生成固定对照图；
- 用固定的均匀 QP 8／16／32 三点补充等 PSNR、等 LPIPS 的 BD-rate，避免把“同 QP
  更省字节”直接误写成“同画质更省字节”；
- 停止 heartbeat 后对正式目录内所有普通文件逐个读取 `st_size`，保存可复核的最终真实
  落盘字节快照；
- 这批数据都已在项目中使用过，因此角色是开发与跨分布候选比较，不重新包装成独立测试；
- 第一轮不运行 SeedVR2，以便把变化先单独归因给 codec。选定 checkpoint 后再评价完整
  Generate／Base／Enhance 管线。

## 2026-09-20 实测结果

1000-step 训练在一张 A800 上完成，共见过 17,000 帧，其中 118 步来自 UVG adaptation；
1000 条 loss 全部有限，峰值 CUDA allocated 为 3,574,242,816 B（约 3.33 GiB）。最终
checkpoint 写出的 17 帧验证流为 10,845 B，fresh decode 逐像素一致。训练正式目录共有
92 个普通文件、1,995,259,471 B。

随后完成 110 个冻结／微调真实编解码任务，全部 fresh decode 精确。固定混合 route 上，
微调版平均字节由 10,007.6 B 降到 7,217.1 B，但 LPIPS 从 0.471661 变差到 0.508935，
所以不能用“同 QP 少字节”直接宣称获胜。三档均匀 QP 的等画质比较进一步显示明显域差异：

| 数据 | 等 LPIPS BD-rate | 等 PSNR BD-rate | 解释 |
|---|---:|---:|---|
| REDS（3 条） | **−11.99%** | **−20.69%** | 微调有效 |
| UVG（3 条） | +29.71% | +6.39% | 过度平滑、跨风格退化 |
| 合并（6 条） | +8.86% | −7.15% | 两个指标给出不同取舍 |

因此 alpha=1 的最终权重保留为有价值的域内结果和消融，但暂不作为全局默认。下一步按
[`CLOUD_A800_SPATIAL_QP_INTERPOLATION.md`](CLOUD_A800_SPATIAL_QP_INTERPOLATION.md)
固定测试 0.25／0.50／0.75 权重插值；这不增加梯度更新，只寻找兼顾 REDS 与 UVG 的单一
checkpoint。
