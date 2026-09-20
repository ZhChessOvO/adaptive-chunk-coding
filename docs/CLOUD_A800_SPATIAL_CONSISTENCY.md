# A800 单卡 Generate 空间一致性

## 一句话说明

原来的控制器会分别判断 16 个块是否适合 Generate，因此可能选出几个互不相连的小岛。
每个小岛都要单独扩出上下文、运行恢复并贴回画面，既容易出现接缝，也增加执行开销。
本轮不改控制器预测，而是在最后选择动作时，让“收益高”和“形状连贯”一起参与决定。

## 方法

每个 512×512 窗口仍分成 4×4 个块，每块动作仍为：

- `Base`：普通质量真实编码；
- `Generate`：低码率真实上下文，解码后对连通 ROI 运行 SeedVR2；
- `Enhance`：发送更多真实编码信息。

控制器为每块给出 Generate 收益、Enhance 收益和 Enhance 额外字节预测。旧求解器只把
16 个区域的收益相加，再在最后删除极弱的孤立 Generate。新求解器直接优化：

```text
区域预测收益之和
− λ × Generate / 非 Generate 的上下左右相邻边数
```

同时继续满足两个原有预算：Enhance 额外字节不超预算，Generate 最多 4 块。这里没有
惩罚 Base 与 Enhance 的交界，因为二者都来自真实 codec 重建；重点处理的是生成内容贴回
真实重建时的边界。

求解不是模糊或贪心平滑。脚本沿 4×4 网格逐格推进，只保留仍可能成为最优解的状态，
因此能在很小的计算量下找到该目标的精确最优动作图。非正收益的 Generate 仍被禁止，
不会为了凑成漂亮形状而主动生成一个预测有害的块。

实现位于：

```text
demo/stage_c_a800_spatial_consistency.py
```

## 为什么取 λ=0.004

旧版本已经用 `0.004` 作为“极弱孤立 Generate”的收益尺度。本轮把同一尺度改写成每条
Generate／非 Generate 邻边的显式代价，不再另用画质结果搜索新超参数。

这不是硬门槛。脚本会同时回放 `0、0.001、0.002、0.003、0.004、0.006、0.008`，让我们
看到预测收益与边界数量怎样连续变化；`0.004` 只是一个容易解释、改动适中的默认折中。
最终是否保留，看真实 spatial-QP、fresh decode 和 SeedVR2 画质，而不是要求它通过一串
人为的淘汰线。

## 与 UVG 适配的关系

两项改动解决的是不同问题：

- UVG 小比例适配回答“这个内容到底应不应该 Generate”；
- 空间一致性回答“确定要 Generate 后，相邻块怎样组成更完整的 ROI”。

例如 UVG 适配已经让 HoneyBee 从 4 个 Generate 块变成 0 个，这是内容判断的变化；空间
项则会优先把零散小岛换成预测收益相近但相邻的块。论文方法可以因此写成一条完整链路：

```text
编码端特征与 Base 探针
  → REDS 主体 + 少量 UVG 的保守双专家校正
  → 带码率预算和空间项的三动作联合求解
  → spatial-QP 真实码流
  → Generate 连通 ROI 恢复
  → 16 像素羽化贴回
```

## 运行与断点

便宜的动作回放同时处理旧 v5 路由和 UVG 适配后路由：

```bash
tmux new-session -d -s a800_spatial_consistency \
  'cd /root/autodl-tmp/adaptive-chunk-coding && \
   bash demo/run_stage_c_a800_spatial_consistency.sh'
```

结果保存到：

```text
/root/autodl-fs/DCVC/runs/a800_spatial_consistency_20260920/
```

其中 `routes_v5_spatial` 是“只加空间项”，`routes_v6_combined` 是“UVG 适配 + 空间项”，
可与“都不加”和“只做 UVG 适配”组成简洁的 2×2 消融。`lambda=0` 必须逐块复现输入动作
图；求解器另用小网格穷举做独立精确性测试。

## 后续真实验证

动作回放只能说明路线怎样变化，不能当画质结果。后续在同一组 30 条 REDS + 7 条 UVG
上完成：

1. 真实 spatial-QP 落盘字节与 fresh decode；
2. `适配关/开 × 空间项关/开` 的关键组合；
3. Generate 使用 16 像素羽化，并保留旧 v5 结果作参照；
4. 分别汇总 REDS、UVG、UVG 适配侧和两条未参与 v6 训练的序列；
5. 固定保存 Beauty、Jockey、YachtRide 等动作图与第 9 帧效果图。

我们的目标是确认这条方法链是否产生清楚、可解释的改善，不追求把每个数据集上的数字都
调到最好。DCVC-UF、SeedVR2 和 spatial-QP codec 始终冻结，所有运行继续只用一张 A800。
