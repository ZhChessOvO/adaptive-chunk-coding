# A800 单卡 v6 真实画质评估

## 要回答的问题

当前方法有两项互补变化：少量 UVG 跨域适配负责判断“该不该 Generate”，空间一致性负责
判断“Generate 区域怎样连在一起”。本轮不再增加模型复杂度，只用真实输出回答三件事：

1. 两项变化单独是否产生可见、可测的影响；
2. 合在一起是否比旧 v5 更合适；
3. 这种变化在 REDS 与 UVG 上是否方向不同。

不设置硬性淘汰线。我们选择能形成清楚方法故事、且实际结果整体合理的版本，而不是要求
37 条视频全部改善。

## 四个版本

所有版本都使用相同的 DCVC-UF、SeedVR2、spatial-QP 格式、最多 4 个 Generate 块和
Enhance 字节预算；Generate 最终都用 16 像素羽化。

| 版本 | UVG 适配 | 空间项 |
|---|---|---|
| `baseline-v5-feather16` | 关 | 关 |
| `adaptation-only` | 开 | 关 |
| `spatial-only` | 关 | 开 |
| `combined` | 开 | 开 |

旧基线的真实码流、fresh decode、SeedVR2 ROI 和 16 像素重组已经完成并独立复核，不重复
运行。三个新逻辑版本共有 111 个“样本×版本”，但同一样本里动作图完全相同时，最终
spatial-QP 流、连通 ROI 和固定种子恢复也完全相同。冻结计划只在 16 个动作逐块相等时
复用，最终需要实际新跑 58 个动作图；其余 53 个逻辑结果引用完全相同的正式输出。

这不是按结果筛样本，而是运行前仅按动作图去重。计划由
`demo/stage_c_a800_v6_evaluation_plan.py` 生成并记录四套 route manifest 的 SHA-256。

## 正式检查

每个新的唯一动作图都执行：

1. 动作图写入真实 spatial-QP 文件，并用文件真实字节计费；
2. 新进程从文件 fresh decode，要求与编码端重建逐像素一致；
3. 从动作图形成 Generate 连通 ROI，单进程加载一次 SeedVR2 后顺序恢复各连通块；
4. 用 16 像素羽化贴回，计算 LPIPS、PSNR、时序差与边界带指标；
5. 保存第 9 帧和四套动作图的 3×3 固定对照图。

汇总同时报告 30 条 REDS、7 条 UVG、UVG 适配侧 5 条、未参与 v6 训练的 2 条以及合并
37 条。样本已经参与方法开发，因此只称开发／误差分析证据，不包装成全新盲测。

## 运行与断点

```bash
tmux new-session -d -s a800_v6_evaluation \
  'cd /root/autodl-tmp/adaptive-chunk-coding && \
   bash demo/run_stage_c_a800_v6_evaluation.sh'
```

每个唯一动作图都有 `variant.complete`，中断后从最后一个完整步骤继续。每分钟记录完成数、
GPU 和三块盘，每 5 秒采样一次显存。正式目录为：

```text
/root/autodl-fs/DCVC/runs/a800_v6_evaluation_20260920/
```

汇总脚本为 `demo/stage_c_a800_v6_evaluation_summary.py`。它会再次核对落盘字节、fresh
decode、连通块数量、17 帧 PNG、复用关系、固定图和 2×2 因子效果。所有大文件继续只放
文件存储，不进入 Git；整轮只使用一张 A800。
