# v6 论文材料与主张边界

通俗图文版已同步到
[Notion：01 当前论文证据包（主图、主表与边界）](https://app.notion.com/p/3e18b22ebd8d81c281e7d6ae63d9a58e)。

## 现在已经有什么

当前主版本已经冻结为 `combined`，不再继续用同一批 37 条视频调参数。它由四部分组成：

1. v5 的 v1 锚点、同画面上下文专家和 Base-probe 专家；
2. 60 个 UVG 窗口带来的小比例跨域校正；
3. 在字节与 Generate 数量预算内精确求解的空间一致性目标；
4. 一条可 fresh decode 的 spatial-QP 码流，以及只对 Generate 连通区域运行的 SeedVR2。

DCVC-UF、SeedVR2 和 spatial-QP codec 始终冻结。现阶段论文要说明的重点不是“把大模型
训练得更强”，而是**如何在同一条合法码流中，按区域联合分配真实编码信息和生成恢复
计算**。

## 论文主线怎样讲

可以按下面的顺序组织 Method：

1. 编码器读取视频、字节预算和恢复计算预算；
2. 用源视频／上下文特征与临时全 Base probe，从两个角度描述每个区域；
3. 轻量控制器预测 Generate、Base、Enhance 的收益、风险和成本；
4. 精确空间求解器最大化区域总收益，并用 `λ=0.004` 轻度惩罚 Generate／非 Generate
   相邻边；
5. 自定义 spatial-QP 格式把 `4×4` 动作图和空间质量场写入一条码流；
6. 解码端先完整 fresh decode，再合并相邻 Generate 区域、运行一次常驻 SeedVR2，并用
   16 像素羽化贴回。

正式方法图已经生成：

```text
/root/autodl-fs/DCVC/runs/a800_paper_package_20260920/method_overview.png
```

## 可以作为贡献的内容

- **一次性、可独立解码的 spatial-QP 神经视频码流**：不是 16 个独立 tile 码流拼接；
  动作图、头部和全部熵载荷都按真实落盘字节计费。
- **恢复感知的 G/B/E 三路区域分配**：同时决定哪里少传后恢复、哪里普通编码、哪里发送
  更多真实细节。
- **两种编码端证据的保守融合**：v2 的同画面相对上下文和 v3 的真实 Base 压缩损伤，
  都只对 v1 做可信的小幅修正。
- **跨域内容校正与精确空间预算的互补组合**：前者主要修正“该不该 Generate”，后者
  主要修正“Generate 区域怎样连在一起”。
- **连通 ROI 解码执行**：减少碎片和重复调度，把恢复计算集中在真正需要的区域。

## 目前最适合放进正文的结果

| 版本 | 平均字节／17 帧 | LPIPS ↓ | PSNR ↑ | 完整时间 | G 边界 | G 连通块 |
|---|---:|---:|---:|---:|---:|---:|
| v5 + feather16 | 9956.5 B | 0.449352 | 24.499 | 20.00 s | 260 | 59 |
| 仅 UVG 适配 | 9983.4 B | 0.447349 | 24.635 | 19.34 s | 237 | 51 |
| 仅空间项 | 9947.5 B | 0.450299 | 24.497 | 19.32 s | 207 | 43 |
| **combined v6** | **10007.6 B** | **0.446839** | **24.682** | **18.69 s** | **186** | **36** |

combined 相对 v5 平均多 51.1 B（0.51%），LPIPS 降低 0.002513，Generate 边界从 260
降到 186，连通块从 59 降到 36，完整时间减少 6.58%。37 条中 18 条改善、6 条完全相同，
因此应写成“综合平衡最好”，不能写成“每条视频都更好”。

分数据集的结论尤其重要：

- REDS 30 条：LPIPS `+0.000101`，基本持平；
- UVG 7 条：LPIPS `−0.013719`，6/7 改善、1/7 相同；
- 空间项单独使整体 LPIPS `+0.000946`，但显著减少碎片；
- 在适配已经打开时再加空间项，LPIPS 继续 `−0.000510`，同时进一步减少连通块。

这正好支持“两部分分工不同但可以结合”的方法故事。

## 不应夸大的地方

- 这 37 条已经参与方法选择，只能称论文开发／误差分析证据，不能包装成全新盲测；
- UVG 留出的 ReadySetGo、YachtRide 没参加 v6 训练，但前 17 帧曾在 v5 中看过；
- 空间项证明的是边界数量、连通块和执行效率改善，不是所有剩余接缝都消失；
- REDS 平均画质没有提升；
- 当前 spatial-QP 是研究扩展，不宣称已兼容现成标准播放器；
- 现在没有证据说明微调 DCVC-UF 或 SeedVR2 会让论文主张更清楚。

## 证据包在哪里

正式目录为：

```text
/root/autodl-fs/DCVC/runs/a800_paper_package_20260920/
```

其中包括：

- `method_overview.png`／`.svg`：方法总图；
- `v6_main_results.tex`：正文主表；
- `v6_factorial_ablation.tex`：2×2 消融表；
- `v6_cross_dataset.tex`：REDS／UVG 分数据集表；
- `v6_tradeoff.png`：LPIPS 与 Generate 碎片的关系；
- `v6_dataset_effect.png`：收益主要来自哪个数据集；
- `claim_evidence_matrix.md`：每条论文主张对应的证据和不能越过的边界；
- `qualitative_manifest.md`：Beauty、Jockey、YachtRide 固定图及读图说明；
- `paper_snapshot.json`、`artifact_manifest.json`：来源、哈希和资源记录。

重新生成时只读取已经完成的汇总，不运行模型：

```bash
/root/autodl-tmp/DCVC/envs/dcvcuf/bin/python \
  demo/stage_c_a800_paper_package.py \
  --summary /root/autodl-fs/DCVC/runs/a800_v6_evaluation_20260920/formal/v6_evaluation_summary.json \
  --frozen-source /root/autodl-fs/DCVC/runs/a800_v6_evaluation_20260920/frozen_source.json \
  --resource-snapshot /root/autodl-fs/DCVC/runs/a800_v6_evaluation_20260920/final_resource_snapshot.json \
  --output-dir /root/autodl-fs/DCVC/runs/a800_paper_package_20260920
```

## 下一步

近期继续冻结 UF 和 diffusion，不再对同一批 37 条追小数点。下一项真正有新增信息的实验
应当是：拿一组风格明显不同、时间更长的视频，冻结 v6 后只检查外部分布与长时序行为。
数据较大时先列出准确下载清单，由用户本地下载并上传；服务器不自行慢速下载大数据。

只有出现下面两类证据之一，才值得另开“上限实验”讨论微调，而且它不替代当前主方法：

- 路由位置已经合理，但 Generate 区域仍反复出现同类恢复缺陷：再考虑小规模 SeedVR2
  LoRA；
- spatial-QP 在固定区域上出现系统性 codec 伪影：再考虑小规模 DCVC-UF 适配。
