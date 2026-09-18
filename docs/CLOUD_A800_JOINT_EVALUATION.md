# A800 单卡 REDS validation + UVG 联合评估

## 一句话结论

冻结的 v5 在 REDS 上成立，尤其新读取的 `val/024..029` 为 6/6 改善；但它没有完整迁移到
UVG。问题主要不在 Enhance，而在 Generate：面对不同画面分布时，Generate 有时改善感知
质量，有时反而改坏平滑纹理，并且不同动作相邻时仍会留下可见接缝。

因此，这一轮不是“全面成功”或“全面失败”。更准确的结论是：**spatial-QP 和 Enhance
已经得到稳定支持；v2/v3 共识对 REDS 有效；Generate 的跨域选择与边界融合是下一轮最值得
解决的问题。**

面向大同行的通俗记录同步在
[Notion：17 REDS + UVG 联合评估与 Generate 下一步](https://app.notion.com/p/3df8b22ebd8d81389267e7a9579102a6)。

## 这次实际做了什么

- 始终只使用一张 NVIDIA A800 80GB；DCVC-UF、SeedVR2、spatial-QP codec 全部冻结；
- 在读取联合质量结果前冻结代码 `03d1503` 和 v5 checkpoint；
- checkpoint SHA-256 为
  `0809fe755e3ae21d8f49c8b6b75dc26bbcefba02a80cf356ca72a8f91ccdbce6`；
- 固定 37 条样本：REDS validation 30 条、UVG 标准序列 7 条；每条取 17 帧、512×512；
- 继续保留每条样本的使用历史，不把旧样本包装成全新独立测试；
- QST 没有可登记的官方 clip ID，暂未纳入，也没有用旧 `sky` 别名替代；
- 对每条样本比较均匀 QP、BasicVSR++、全 Generate、Enhance-only、最终联合路由，以及
  同一路由关闭 Generate／Enhance 的两项消融；
- 码率使用真实落盘文件大小，所有正式 spatial-QP 流均从文件 fresh decode。

REDS 30 条的角色为：开发 6 条、历史评估 6 条、先前已消费评估 12 条、冻结后新读取
`024..029` 6 条。UVG 中 Jockey、ShakeNDry 是历史使用样本，其余 5 条记为跨分布评估，
但不声称独立。

## 主要结果

LPIPS 越低越好。“最近均匀 QP”是对每条样本分别寻找与最终路由字节数最接近的已测均匀
QP 点，因此比只拿一个固定 QP 作比较更公平。

| 数据 | 样本数 | 最终字节/17帧 | 最终 LPIPS | 相对最近均匀 QP | 改善条数 | 关闭 Generate | 关闭 Enhance |
|---|---:|---:|---:|---:|---:|---:|---:|
| REDS + UVG | 37 | 9956.5 | 0.447651 | -0.014411 | 21/37 | +0.012012 | +0.038419 |
| REDS | 30 | 11307.0 | 0.475218 | -0.020686 | 19/30 | +0.015137 | +0.043403 |
| UVG | 7 | 4168.7 | 0.329506 | +0.012478 | 2/7 | -0.001382 | +0.017060 |

这里的“关闭某分支”仍使用同一张冻结 action map，只把对应动作改成 Base。正数表示关闭后
变差，也就是原分支有贡献。

- 合并 37 条时，两条分支平均都有贡献；Enhance 为 37/37 方向一致，Generate 为
  29/37 方向一致；
- REDS 上 Generate 为 27/30 有益、Enhance 为 30/30 有益；
- UVG 上 Enhance 仍是 7/7 有益，但 Generate 平均略有害：只在 Beauty、ShakeNDry 上
  有益，在 Bosphorus、HoneyBee、Jockey、YachtRide 上有害，ReadySetGo 没有选择
  Generate；
- 最终路由相对全 Generate 平均快 29.74%，37/37 都更快；REDS 与 UVG 分别快
  29.14% 和 32.28%；
- 新读取的 REDS `024..029` 相对最近均匀 QP 为 6/6 改善，平均 LPIPS 降低
  `0.021238`；关闭 Generate／Enhance 也均为 6/6 变差。这部分支持“REDS 上有效”并非
  只来自旧开发样本。

37 条共路由 592 个区域：Base 233、Generate 132、Enhance 227。v2 上下文与 v3
Base-probe 共识共激活 955 个逐目标纠错。

完整逐方法表和逐样本值保存在：

```text
/root/autodl-fs/DCVC/runs/a800_joint_evaluation_20260918/formal/joint_evaluation_summary.md
/root/autodl-fs/DCVC/runs/a800_joint_evaluation_20260918/formal/joint_evaluation_summary.csv
/root/autodl-fs/DCVC/runs/a800_joint_evaluation_20260918/formal/joint_evaluation_summary.json
```

## 固定可视化说明了什么

37 条样本都保存了固定第 9 帧的 13 面板图。人工重点检查了 REDS 开发首条、冻结新 REDS
首尾两条，以及 UVG Beauty、Jockey、YachtRide：

- 没有错位、空块、越界或动作图与输出不对应等工程错误；
- REDS 三张抽查图没有发现明显拼接故障；
- Beauty 和 YachtRide 的最终联合输出能看到不同动作交界处的色调／纹理接缝；
- Jockey 的 Generate 会改坏原本较平滑的草地和背景纹理；
- 当前已经使用 64 像素 ROI 上下文和固定 8 像素 feather，因此问题不能简单归因于
  “完全没有上下文或融合”，而是固定窄融合与逐块独立决策仍不够。

固定图只是帮助定位问题，不代替 37 条的数值统计。Beauty 的 LPIPS 中 Generate 有益，
但局部接缝仍肉眼可见，也说明只优化单一平均指标不够。

## 完整性与资源

- 37/37 个 `sample.complete`、37/37 张固定图、37/37 条冻结路由均存在；
- 481 个方法-码流引用逐一与真实文件大小复核，涉及 333 个唯一流；
- 所有 spatial-QP fresh decode 与编码端重建逐像素一致；均匀流也从真实文件 fresh decode；
- 没有按结果剔除样本，没有使用多卡，没有微调三套冻结模型／codec；
- 正式运行墙钟 7598 秒，即 2 小时 6 分 38 秒；
- `nvidia-smi` 采样峰值 23275 MiB；实验结束后 GPU 为 0 MiB、0% 利用率；
- 正式目录普通文件真实字节为 2,798,871,238 B，`du` 为 2,805,813,734 B；
- 完成时 `/root`、`/root/autodl-tmp`、`/root/autodl-fs` 使用率约为 12%／17%／26%。

正式目录：

```text
/root/autodl-fs/DCVC/runs/a800_joint_evaluation_20260918
```

## 下一步建议：v6 只改 Generate 的稳定性

下一轮不需要多卡，也不应微调 DCVC-UF 或 SeedVR2。建议保留 v5 的上下文 + Base-probe
共识和已经稳定的 Enhance，先做三个成本较低、容易解释的改动：

1. **边界诊断**：直接复用现有 Generate ROI 输出，比较 8／16／32 像素融合宽度及边界
   色差，不重跑 SeedVR2，先确认接缝能靠合成修掉多少；
2. **空间一致路由**：在预算求解中加入相邻动作变化代价，尤其惩罚 Generate 与非 Generate
   在平滑区域频繁切换；4×4 动作图很小，可以在单卡甚至 CPU 上完成；
3. **Generate 跨域置信度**：把 v5 的 Generate 分数变成“收益足够明确才选”，但不设一条
   人为的实验成败硬门槛。Enhance 维持原逻辑。

这一轮 37 条一旦用于设计 v6，就应在账本里改记为开发／误差分析材料。v6 最终论文证据
需要再找未参与设计的视频，或者使用按序列留一法报告，不能把同一批 UVG 调完后继续称为
独立跨域验证。
