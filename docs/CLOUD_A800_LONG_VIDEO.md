# A800 连续长视频与重叠恢复

这页记录从 17 帧原型扩展到长视频的设计、实现和单卡验证。它回答的是“现有方法能否
连续工作”，不是新的独立测试，也不拿一个开发样本宣称论文优势。

## 先说结论

17 帧现在只是 router 和 SeedVR2 的**局部观察窗口**，不再是 codec 必须重启的长度。
实际长视频采用三条不同时间节奏：

1. DCVC-UF 从一个 I 帧开始，随后连续编码任意多个 8 帧 P 单元，参考状态一直保留；
2. router 继续看 17 帧窗口，窗口从 0、16、32、…开始，每次为后续两个 P8 单元更新
   G／B／E 动作图；
3. SeedVR2 也看 17 帧，但窗口从 0、8、16、…开始。重叠帧的多个预测按固定三角权重
   融合，再只在该帧的 Generate 区域内用空间羽化贴回。

这样既复用现有 17 帧 controller／恢复模型，又避免把长视频切成彼此独立的小片段。

## 为什么 codec 可以连续

spatial-QP 码流语法原本就为每个 coding unit 保存一张动作图。一个 unit 是起始 I 帧或
一个 8 帧 P 组。此前编码器只是把同一张 route 重复写给所有 unit；现在新增的
`coding_unit_routes` 可以逐 unit 提供动作图：

```text
unit 0: I，帧 0，动作图 A
unit 1: P8，帧 1..8，动作图 A
unit 2: P8，帧 9..16，动作图 A
unit 3: P8，帧 17..24，动作图 B
unit 4: P8，帧 25..32，动作图 B
```

动作图从 A 变成 B 时，DCVC-UF 的 DPB／参考特征不清空，所以仍是一条连续预测链。
fresh decoder 从每个 unit 的码流头读出对应动作图，不依赖源图或隐藏命令行参数。旧的
单 route JSON 仍保持兼容，自动重复同一动作图。

## 为什么恢复窗口用步长 8

如果 SeedVR2 也每 16 帧才启动一次，17 帧窗口只共享一帧，边界附近几乎没有融合余地。
改成步长 8 后，33 帧会得到 0..16、8..24、16..32 三个窗口。每个窗口使用下面的
确定性权重：

```text
1/9, 2/9, ..., 8/9, 1, 8/9, ..., 2/9, 1/9
```

同一帧有多个 SeedVR2 预测时先按这些权重平均。ROI crop 可以包含上下文，但最终只有
该帧动作图中的 Generate 像素被采用；Base 和 Enhance 仍来自真实 codec 解码。

## 断点续跑

恢复任务按“17 帧窗口 × Generate 连通块”拆成组件。每个组件完成后原子写入 PNG 和
metadata；重启时逐个核对 manifest 哈希、seed、尺寸和 17 张输出，只计算缺失组件。
同一次进程内的待处理组件共享一个 SeedVR2 模型实例。主脚本每分钟记录 GPU、墙钟和
三块盘，并对 gate、route、codec encode、fresh decode、ROI prepare、restore、evaluate
逐级跳过完整产物。

## 代码入口

- `demo/stage_c_spatial_quality_codec.py`：连续参考环和 per-unit 动作图；
- `demo/stage_c_long_video_plan.py`：把 0、16、32、…的 17 帧 routes 组装为连续计划；
- `demo/stage_c_long_video_seedvr2.py`：重叠 ROI、时间融合、断点恢复和评估；
- `demo/stage_c_a800_spatial_consistency.py`：精确空间求解已支持声明的矩形网格；
- `demo/run_stage_c_a800_long_video_smoke.sh`：A800 单卡 33 帧完整复现实验。

轻量测试不需要运行 codec 或 SeedVR2：

```bash
python demo/stage_c_long_video_plan.py self-test
python demo/stage_c_long_video_seedvr2.py self-test
python demo/stage_c_a800_spatial_consistency.py self-test
python demo/stage_c_spatial_quality_format_test.py
```

正式 smoke 必须放在 tmux：

```bash
tmux new-session -d -s a800_long_video_v1 \
  'cd /root/autodl-tmp/adaptive-chunk-coding && \
   bash demo/run_stage_c_a800_long_video_smoke.sh'
```

正式目录是：

```text
/root/autodl-fs/DCVC/runs/a800_long_video_smoke_20260920
```

## 33 帧机制验证

数据使用 REDS validation `000` 的第 0..32 帧、`x=384,y=96,w=h=512` 裁剪，角色是
**反复使用过的开发／机制验证数据**。第 0..16 帧复用冻结 v6 route；第 16..32 帧用同一
冻结 controller、Base probe 和空间求解器生成新 route。它不属于独立测试。

2026-09-20 本轮已经完成：

- 33 帧写入一条包含 1 个 I + 4 个 P8 单元的实际 spatial-QP 码流；
- 第 3 个 coding unit 起动作图发生变化，codec 参考状态没有重置；
- 独立进程已 fresh decode 33 帧；
- SeedVR2 计划包含 3 个重叠窗口和 3 个可恢复 ROI 组件，实际处理像素帧为全画面窗口的
  54.6875%；
- fresh decode 与编码器重建的 33 张 PNG 逐像素一致，最大误差为 0；
- 正式码流为 22,707 B，SHA256 为
  `774572253d2654d8be057d163a2c7751658a8a7b6ba3144c594216f255bf8521`；
- 峰值 CUDA allocated 为 20,404,906,496 B（约 19.00 GiB）；正式目录共约 73.24 MB。

恢复对画质的影响如下。LPIPS 越低越好；这里的 PSNR 和 LPIPS 只用于检查机制，没有把
这条开发样本当成论文比较结果。

| 输出 | PSNR (dB) | LPIPS | 时序差分 MAE |
|---|---:|---:|---:|
| spatial-QP fresh decode | 21.4038 | 0.5700 | 19.7729 |
| SeedVR2 硬窗口切换 | 21.2119 | **0.5561** | 19.9791 |
| SeedVR2 重叠融合 | 21.2150 | 0.5568 | 19.9715 |

重叠融合相对硬切换的 PSNR 增加 0.0031 dB、时序差分 MAE 降低 0.0076，但 LPIPS
增加 0.0007，三者都很小。在动作切换附近的一对帧上，时序误差由 21.1483 降到
21.1084。因此选择重叠融合为默认，是因为它以几乎不变的画质消除了窗口硬切换，而不是
因为它在每个单帧指标上都获胜。

主要耗时为：codec 编码 1.78 秒、独立码流解码 1.65 秒、SeedVR2 模型加载 80.14 秒、
3 个 ROI 组件实际推理合计 33.44 秒。按真实顺序执行 fresh decode、恢复和融合共
138.06 秒。首次完整任务从启动到落盘为 344 秒，其中包含一次启动失败后的自动续跑；
成功的恢复进程为 119.18 秒。完成时 `/root`、`/root/autodl-tmp`、`/root/autodl-fs`
分别约使用 12%、17%、27%。

最终机器可读记录以
`/root/autodl-fs/DCVC/runs/a800_long_video_smoke_20260920/run_summary.json` 为准；固定图为
`evaluation/visuals/long_video_boundary.png`。

## 还没有解决的部分

- v5/v6 controller checkpoint 本身仍按 512×512、4×4 训练；底层支持矩形网格不等于
  router 已经在其他宽高比上验证；
- 相邻 router 窗口目前独立决策，尚未加入“收益接近时沿用上一动作”的时间惯性项；
- 任意帧数需要把末尾补齐到 codec 的 8 帧调度边界，解码后再裁回原长度；
- 33 帧只验证机制。分钟级视频、横竖屏和 9／17／33 敏感性留作后续轻量实验。

机制验证已经通过。主线顺序因此进入：spatial-QP-aware DCVC-UF 微调、SeedVR2 单卡
长期微调、重新生成 teacher、再训练带时间一致性的 router。所有阶段仍只使用一张
A800；本页结果不作为决定微调是否值得做的硬门槛。
