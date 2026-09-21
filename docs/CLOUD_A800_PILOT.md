# A800 80GB 云端试跑交接

## 目标与授权范围

本轮使用一张完整的 NVIDIA A800 80GB，完成下一阶段的单卡试跑：

1. 从全新服务器恢复并验证环境；
2. 用少量已允许的数据复现 DCVC-UF 空间质量码流与 SeedVR2 推理；
3. 冻结 DCVC-UF 和 SeedVR2，生成 500–1000 个 17 帧、512×512 样本的区域反事实标签；
4. 训练并评估一个轻量、预算条件化的 G／B／E 控制器；
5. 保存可恢复的代码、配置、日志、checkpoint、指标和固定可视化，然后停止并汇报。

本轮不授权多卡完整标签生产、SeedVR2 微调或 spatial-QP codec 微调。单卡试跑结束后，先根据吞吐、磁盘量、标签方差和开发集信号决定是否扩大。

## 当前方法状态

E20–E25 已经完成无训练验证：

- SeedVR2 在真实低码率 DCVC-UF 重建上稳定改善 LPIPS；
- 一次编码的空间质量格式可写盘并在新进程中 fresh decode；
- 码流包含 action map、质量档位、插值模式、头部和真实熵子流；
- Generate／Base／Enhance 共用一个 codec 参考环，不拼接不同 QP latent；
- 连通 Generate ROI 已经比全画面 SeedVR2 节省推理时间；
- 当前 action map 仍是开发集 oracle，尚无可部署控制器。

控制器不应只预测一个笼统的“置信度”。第一版至少学习或估计以下软量，再由预算求解器量化为三种动作：

- Generate 相对低质量重建的预期 LPIPS 收益；
- Enhance 相对 Base 的感知／保真收益；
- Enhance 的真实额外字节；
- 身份、文字、结构和时序幻觉风险；
- Generate ROI 面积、连通性与实际计算成本。

编码前预算和源视频都已知，因此控制器可以使用编码端可见的源图、编码特征和预算。解码端只能使用码流中实际发送的信息和 action map。

## 服务器与磁盘预检

开始安装前先保存以下输出：

```bash
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version \
  --format=csv,noheader
nvidia-smi -L
df -h / /root/autodl-fs /root/autodl-tmp .
free -h
nvcc --version
```

要求：

- `nvidia-smi` 应显示一张约 80GB 的完整 A800，而不是 10／20／40GB MIG 切片；
- 使用带编译工具链的 CUDA devel 镜像；
- 建议至少 64GB 主机内存；
- 当前系统盘是 `/root`（30GB），数据盘是 `/root/autodl-tmp`（50GB），文件存储是 `/root/autodl-fs`（200GB）；文件存储保存数据、模型和正式输出，数据盘保存仓库、环境、编译产物和不超过 8GB 的单样本临时缓存；
- 文件存储或 50GB 数据盘达到容量的 80% 时停止新样本、保存断点并汇报，不以删除已有正式结果掩盖空间问题。

上传资产和磁盘布局按 `docs/CLOUD_STORAGE_AND_UPLOAD.md` 执行，环境、扩展和 checkpoint 安装按 `README.md` 执行。五个文件最初直接放在 `/root/autodl-fs/`，但 ZIP 在验证解压后可以删除，权重也可能已经移入持久目录。每次恢复先盘点现状；已有内容不重复下载，真正缺少的数据或依赖从官方来源直接补齐。下载前检查并移除镜像覆盖，不编译当前 bridge 不需要的 Apex 或 FlashAttention。

## 执行阶段

### 阶段 0：环境与最小 smoke test

先完成：

- 两个 DCVC-UF 扩展可导入；
- `demo/stage_c_spatial_quality_format_test.py` 通过；
- DCVC-UF checkpoint 和 SeedVR2 四个文件存在且能载入；
- 1–3 帧 SeedVR2 推理完成；
- 一条小码流可在新进程中 fresh decode，且不读取源图或隐藏参数。

记录软件版本、GPU 型号、峰值显存和墙钟时间。若 smoke test 失败，先修复环境和复现问题，不直接开始长任务。

### 阶段 1：小规模吞吐校准

先对 8–16 个训练样本跑完整反事实流程，回答：

- 每个样本的 teacher 生成时间；
- 每个样本和每种保留策略的磁盘量；
- Generate／Base／Enhance 标签是否有足够方差；
- 哪些中间图像必须保存，哪些可以只保存指标和紧凑特征；
- 任务中断后能否根据 manifest 继续，而不重复已完成样本。

通过后可继续到 500–1000 个标签；不需要再次请求许可。若预计超过 12 小时，或文件存储／数据盘将达到 80% 使用率，先调整缓存策略和样本量，并在结果中保留原始估计。

### 阶段 2：teacher labels

只使用 REDS `train/000..239`。固定第一轮质量 profile 为 Generate=8、Base=16、Enhance=32，除非小校准发现实现错误；不要边看开发集边反复改档位。

标签应保存每个区域的候选效用和成本，而不只保存 oracle 的单一动作。至少包括：

- 三种动作下的局部与整段 LPIPS／诊断指标；
- Enhance 的实际增量字节；
- Generate 的实测 ROI 计算量、面积与碎片度；
- 时间一致性和重要内容风险的可计算代理；
- 样本、帧窗、裁剪、质量档位、随机种子和代码配置。

DCVC-UF 与 SeedVR2 在本阶段冻结。teacher 可以使用源图计算监督信号；部署时的动作必须由编码端控制器产生，并把 action map 写入码流。

### 阶段 3：轻量控制器

先实现容易解释的基线，再增加复杂度：

1. 固定阈值或线性模型；
2. 小型预算条件化网络；
3. 在相同预测量上使用预算求解器选择 G／B／E。

训练和推理必须分离。训练输入不得包含部署时编码端不可获得的信息；GT 只用于标签和损失。控制器输出需能在多个带宽／算力预算下改变同一区域的动作。

### 阶段 4：开发集评估

训练结束后可把 REDS `val/000..005` 作为开发集，并明确这样称呼。`006..023` 已影响历史实验，后续只能作为已使用评估或误差分析；`024..029` 可以在记录冻结版本、帧窗和裁剪后用于新评估。UVG、QST 等外部数据与 REDS validation 一起作为跨分布论文评估，并分别报告各数据集结果。

至少比较：

- 普通 DCVC-UF 均匀质量曲线；
- 普通低码率编码加同一 SeedVR2；
- Enhance-only；
- 全 Generate；
- 学习到的三路联合策略；
- BasicVSR++ 确定性恢复对照；
- 固定同一路由关闭 Generate／关闭 Enhance 的消融。

所有基础流、增强流、action map、头部和辅助信息按落盘字节计费。报告 LPIPS、固定视觉对照、时序指标、PSNR 诊断、完整 fresh decode 时间和峰值显存。不能把真实编码增强带来的提升算作 Generate 贡献。

## 长任务工程要求

- 每个阶段写 manifest，并使用确定性的样本 ID；
- 已完成项可跳过，部分文件必须先写临时名再原子改名；
- 至少每 1–2 分钟输出一次样本进度、GPU 显存、累计耗时和磁盘占用；
- 对单个恢复任务设置有界超时，避免重复 DiT CPU/GPU 搬运造成“看起来卡住”；
- 单卡 SeedVR2 使用单进程 `torchrun`，显式指定 GPU，并在退出前清理 process group；
- 每个正式 benchmark 使用独立进程，避免 CUDA warm-up 和共享 codec context 污染；
- 不批量保存所有视觉候选。固定保存少量 GT／Base／SeedVR2／三路拼接／action map 对照，供人工检查。
- 正式 manifest、码流、日志、checkpoint、指标和可视化直接写入 `/root/autodl-fs/DCVC/runs`；慢盘有明显随机 I/O 瓶颈时，只在 50GB 数据盘使用不超过 8GB 的当前样本缓存，样本结束后立即写回并清理。

## 试跑完成标准

试跑无论正负都要形成一个可复现结论。交付物至少包括：

- 环境与 preflight 记录；
- teacher-label manifest、吞吐、磁盘量和标签统计；
- 控制器配置与 checkpoint；
- 普通 QP、全 Generate、Enhance-only 和三路策略的开发集比较；
- Generate／Enhance 分开消融；
- 固定视觉对照；
- 下一阶段所需卡数、预计时长、磁盘量和继续／停止建议。

若轻量控制器在预先固定的训练日程后没有稳定开发信号，如实报告负结果并停止，不通过额外读取数据或反复挑选设置制造正结果。

## 当前数据角色（取代早期封存边界）

- 训练：REDS `train/000..239`；
- 开发：REDS `val/000..005`，已经多次使用，不能称独立测试；
- 历史已使用：`val/006..011`；可以复现、分析和参与论文汇总，但不能再称新的独立测试；
- 已消费测试：`val/012..023`；可以复现、分析和参与论文汇总，但不能再称新的独立测试；
- 新评估候选：`val/024..029`；读取前记录冻结版本、帧窗和裁剪，若结果反过来影响设计，立即改记为开发用途；
- UVG、QST 等外部视频：跨数据集论文评估，不单列为应用测试；已使用过的具体片段要明确标注。

数据集和输出不进入 Git。记录来源、真实字节和必要的完整性校验。不要使用 true-fill 冒充结果。

## 2026-09-17 单卡试跑实测结果

本轮已经在一张完整的 NVIDIA A800 80GB PCIe 上完成，正式结果位于
`/root/autodl-fs/DCVC/runs/a800_pilot_20260917`。DCVC-UF 和 SeedVR2 全程冻结，
没有使用多卡，也没有进行大模型或 codec 微调。训练只读取 REDS
`train/000..239`，开发评估只读取 `val/000..005`；`val/012..023` 和封存的
`val/024..029` 均未读取。

### 标签、控制器和预算响应

- 固定账本包含 500 个训练样本和 6 个开发样本，每个样本为 17 帧、512×512；
  训练集共得到 8000 个区域标签。质量 teacher 的 500+6 个样本累计用时
  6569.36 秒，ROI 实测 teacher 累计用时 3092.66 秒；标签树分别为
  25,062,284 和 6,802,041 字节。
- 质量标签的逐区域最低 LPIPS 动作为 Enhance 7560 次、Generate 425 次、Base
  15 次。这说明当前固定质量档位下 Enhance 占明显优势，控制器必须靠真实预算约束，
  不能把 teacher 的逐区域最优动作直接当作部署路由。
- 最终小型 MLP 有 5768 个参数，按预先固定的 240 epoch 日程训练，用时
  14.34 秒，没有用开发集挑 checkpoint。开发集上 Generate／Enhance 的 LPIPS
  收益预测 R² 分别为 0.238／0.753，ROI 时间和 Enhance 额外字节预测 R² 分别为
  0.691／0.970。
- 同一区域会随预算改变动作：6 个开发样本合计有 57 个区域发生变化。正式采用的
  中预算路由在 96 个区域中选择 Generate 34 次、Enhance 62 次、Base 0 次。
  因而这次结果证明了预算响应和感知收益，但也暴露出三路控制器暂时退化成两路的风险。

### 六条开发视频的正式比较

下面均为每个 17 帧样本的均值。字节数来自真实落盘文件，包含头部、action map 和
所有子流；LPIPS 越低越好。完整 14 个方法的逐样本记录和 PSNR／时序／显存数据见
`formal/formal_summary.json`、`formal/formal_summary.csv` 和
`formal/formal_summary.md`。

| 方法 | 平均落盘字节 | bpp | LPIPS | 完整 fresh decode 秒 |
|---|---:|---:|---:|---:|
| 均匀 QP8 | 4646.0 | 0.008340 | 0.592594 | 6.240 |
| 均匀 QP24 | 13571.8 | 0.024363 | 0.472186 | 6.269 |
| 均匀 QP32 | 22941.0 | 0.041183 | 0.416514 | 6.233 |
| QP32 + BasicVSR++ | 22941.0 | 0.041183 | 0.414429 | 7.989 |
| 普通 QP8 + SeedVR2 | 4646.0 | 0.008340 | 0.481470 | 28.319 |
| Enhance-only | 23052.2 | 0.041382 | 0.416475 | 6.226 |
| 全 Generate | 4786.8 | 0.008593 | 0.478122 | 28.248 |
| 学习到的联合路由 | 13771.5 | 0.024722 | 0.432362 | 37.923 |
| 同路由关闭 Generate | 15549.2 | 0.027913 | 0.458049 | 6.221 |
| 同路由关闭 Enhance | 6499.0 | 0.011667 | 0.506659 | 38.207 |

联合路由与字节最接近的均匀 QP24 相比，LPIPS 降低 0.039824；与全 Generate
相比降低 0.045760。保持同一路由关闭 Generate 后，LPIPS 变差 0.025687；关闭
Enhance 后变差 0.074297，因此两条分支都贡献了可测收益。另一方面，联合路由的
完整解码时间为 37.92 秒，比全画面 Generate 的 28.25 秒还慢，说明多个连通 ROI
的调度和模型调用开销仍需优化。

### 完整性、资源和结论

- 六个正式样本的所有记录字节都重新与文件大小核对；空间质量流均在独立新进程、
  不读取源 RGB 的条件下 fresh decode，并与编码进程输出逐像素一致。
- 每个样本固定保存第 9 帧的 12 宫格对照，共 6 张，已人工抽查首尾样本；正式
  输出树约 1.00 GB、5583 个文件。
- 整轮从 preflight 到正式汇总约 4 小时 23 分钟。最高 CUDA allocated memory 为
  20,664,085,504 字节，约 19.25 GiB，出现在质量 teacher；正式评估峰值约
  14.96 GiB。
- 完成时系统盘、50GB 数据盘和 200GB 文件存储使用率分别约为
  8.62%、16.67% 和 20.90%，均远低于 80% 停止线。
- 结论是“值得继续，但先修控制器和 ROI 执行，不进入多卡”。下一轮仍可在同一张
  A800 上完成：给 Base 加入明确的率失真占位或动作正则，降低 ROI 碎片和重复模型
  调用，然后用同一冻结开发协议复验。当前结果尚不足以读取新的验证分区、启动多卡
  标签生产或微调 SeedVR2／DCVC-UF。

服务器随后发生重启，但发生在 `formal_evaluation.complete` 和最终资源快照写入之后；
六个样本的原子完成标志、汇总、码流和可视化都已复核存在，重启没有造成结果丢失。

## 2026-09-18 单卡后续复验

第一轮暴露的两个问题已经用同一张 A800 做了有界复验，详细记录见
[`CLOUD_A800_FOLLOWUP.md`](CLOUD_A800_FOLLOWUP.md)。本轮没有新标签、没有重训，
也没有读取新的验证视频。

- 预先固定的低预算点在 96 个区域中自然得到 Base 24、Generate 24、Enhance 48；
  联合路由平均 10497.5 B／17 帧、LPIPS 0.463212，相对逐样本最近的已测均匀 QP
  点改善 0.058304，6/6 个样本方向一致。
- 同一路由关闭 Generate 或 Enhance 后，LPIPS 分别变差 0.022021 和 0.046784。
- 同一样本的多个 Generate ROI 改为只加载一次 SeedVR2 后，6/6 输出与旧实现逐像素
  一致；中预算完整时间从 37.923 秒降到 24.895 秒，比全画面 Generate 快 11.87%。
  逐样本为 5/6 更快，符合事先固定的六样本均值判据。
- 新正式峰值显存 17.65 GiB，运行约 18 分 14 秒，`du -sb` 约 368.8 MB；三块盘
  使用率约 11.01%／16.67%／21.08%。继续使用单张 A800 完全可行。

下一步应先确定并授权独立测试视频，再冻结本轮实现做一次不调参复验；不要继续用这六条
开发视频选方案，也不进入多卡或 SeedVR2／DCVC-UF 微调。

## 2026-09-18 一次性独立测试

用户随后授权读取此前未使用的 REDS `val/012..023`。完整预注册协议、逐方法指标和资源
记录见 [`CLOUD_A800_INDEPENDENT_TEST.md`](CLOUD_A800_INDEPENDENT_TEST.md)。本轮继续
冻结 DCVC-UF、SeedVR2 和 5768 参数控制器，没有新 teacher、重训或结果后调参。

- 中预算相对逐样本最近的已测均匀 QP，LPIPS 平均降低 0.024350，12/12 条改善；
  关闭 Generate／Enhance 分别有 10/12、12/12 条变差；完整时间比全 Generate
  平均快 26.55%，12/12 条更快。四项预注册判据全部通过。
- 低预算平均 LPIPS 降低 0.007339，分支贡献和速度判据都通过，但逐样本只有 5/12
  条改善，未达到至少 8/12，因此总判定失败。
- 所有真实流均通过逐像素 fresh decode；已人工检查首、中、末三张固定 17 面板。
- 总墙钟约 1 小时 47 分，正式峰值 18.33 GiB；结束时三块盘约
  11.10%／16.68%／23.28%。不需要多卡。

`val/012..023` 已消费，不能再用于调参或独立验证；`val/024..029` 继续封存。下一步应
只在既有训练／开发数据上分析和改进低预算跨视频稳定性，再为新的未读数据写预注册协议
并取得授权。中预算结论已经得到独立支持，没有理由为当前问题启动多卡或大模型／codec
微调。

## 2026-09-18 低预算稳定性后续

上述“先在既有训练／开发数据上改进”的步骤已经完成两轮。v2 同画面上下文方案未过
训练门槛；v3 编码端 Base 预分析过了训练门槛，但在六条开发视频上只通过 11 项固定判据
中的 10 项。v3 以 10349.0 B／17 帧得到 LPIPS 0.470660，6/6 优于最近均匀 QP，两项
消融也均 6/6 变差；但相近字节的 v1 为 0.463212，因此 v3 平均差 0.007448，不能替代
v1，也不应进入新独立测试。完整记录见
[`CLOUD_A800_LOW_BUDGET_V2.md`](CLOUD_A800_LOW_BUDGET_V2.md) 与
[`CLOUD_A800_LOW_BUDGET_V3.md`](CLOUD_A800_LOW_BUDGET_V3.md)。下一步需要先讨论新的
训练假设；`val/024..029` 仍保持封存。

## 2026-09-18 v4 锚定式融合复验

用户确认不再使用硬性门槛，并希望保留 v2 的合理部分。因此 v4 不让新模型重新推翻
v1，而是让 v2 同画面上下文与 v3 Base 预分析只预测对 v1 的小型残差纠错。训练选择
仍只用 REDS train 的 500 个既有 teacher 样本，按原视频做 5 折；开发集只运行训练阶段
自动选中的一个候选。详细协议与结果见
[`CLOUD_A800_LOW_BUDGET_V4.md`](CLOUD_A800_LOW_BUDGET_V4.md)。

- 117 个训练候选中，融合纠错的平均 oracle regret 为 0.093707，相对 v1 的 0.113505
  降低 17.44%；最差四分之一 utility 更高，有害 Generate 从 11.10% 降到 8.27%。
- 冻结后六条开发视频平均为 10594.2 B／17 帧、LPIPS 0.468980；v1 为 10497.5 B、
  0.463212。v4 平均多 96.7 B、LPIPS 差 0.005769，只有 2/6 条优于 v1。
- 关闭 Generate／Enhance 后分别平均变差 0.016557／0.046830，均为 6/6 方向一致；
  两条分支有效，但区域纠错的跨视频稳定性不足。
- 18 个新码流均核对真实落盘字节并通过逐像素 fresh decode；六张固定图全部生成，已
  人工检查首、中、末三张。峰值显存 22097 MiB，正式闭环 910 秒，结果约 300.3 MB。

本轮没有用门槛判生死，而是直接选择综合上更合适的版本：**低预算主版本继续使用 v1**。
v2/v3 的辅助信号保留为方法贡献和后续“保守纠错”研究方向，但 v4 不进入新的独立测试。
所有工作继续可由一张 A800 完成。后续 v5 先做保守纠错，再把冻结版本放到 REDS
validation 与 UVG／QST 的联合评估池中；不再设置人为的永久封存边界，但必须诚实记录
每条样本何时参与过设计选择。

## 2026-09-18 数据与评估口径更新

研究目标是形成论文证据，而不是模拟已经上线的产品验收。因此后续采用更实用的规则：

1. REDS train 继续用于训练；REDS validation 可用于开发、误差分析和论文评估，具体角色按使用历史标注；
2. UVG、QST 等外部视频与 REDS validation 放在同一个跨分布评估框架中，同时给出分数据集和合并统计；
3. 不再把 Jockey 等视频称作“应用测试”，也不因它们曾使用过就完全丢弃，只是不把它们包装成全新独立证据；
4. 每次运行前冻结代码、checkpoint、样本、帧窗和裁剪，运行后登记实际读取；如果结果用于改方法，样本角色随即更新；
5. DCVC-UF、SeedVR2 与 spatial-QP codec 继续冻结，仍只用一张 A800。

## 2026-09-18 v5 保守共识开发结果

v5 把 v2 上下文和 v3 Base 预分析保留为两个独立专家，只有二者对 v1 的纠错方向一致、
各自不确定性也足够小时才采用较小的可信纠错。训练集分组折外自动选择 `z=0.25`、纠错
比例 `1.0`，平均 oracle regret 从 v1 的 0.113505 降到 0.103148。

冻结后六条 REDS 开发视频平均为 10429.5 B／17 帧、LPIPS 0.461972；v1 为 10497.5 B、
0.463212。v5 平均少 68 B、LPIPS 低 0.001240，但只有 3/6 条逐视频更好，因此只能说
平均略优，不能说已经彻底解决跨视频稳定性。关闭 Generate／Enhance 后分别平均变差
0.023291／0.048774，均为 6/6；完整时间 22.566 秒，6/6 快于全 Generate。18 个新流均
通过真实字节和逐像素 fresh decode，固定图首、中、末已人工检查。正式闭环约 901 秒，
峰值 22097 MiB，输出约 294.3 MB。

在用户确认的“无硬门槛、选择综合最合适版本”口径下，v5 冻结进入 REDS validation 30 条
与 UVG 7 条联合论文评估。联合结果同时报告 REDS、UVG 和合并统计，并保留开发、历史已用
和新评估角色；不得再用联合结果反向调 controller。QST 在拿到可登记的官方 clip ID 或
官方整库前明确标为暂未纳入，不用旧 `sky` 别名冒充可复现样本。详细记录见
[`CLOUD_A800_LOW_BUDGET_V5.md`](CLOUD_A800_LOW_BUDGET_V5.md)。

## 2026-09-18 REDS + UVG 联合评估

冻结 v5 随后完成 30 条 REDS validation 与 7 条 UVG 的联合评估。合并 37 条时，最终
路由平均 9956.5 B／17 帧、LPIPS 0.447651，相对逐样本最近均匀 QP 平均降低
0.014411，21/37 条更好；相对全 Generate 平均快 29.74%，37/37 条更快。

分数据集结论并不相同。REDS 上平均改善 0.020686，19/30 条更好；特别是冻结后新读取的
`024..029` 为 6/6 改善。UVG 上却平均差 0.012478，只有 2/7 更好。关闭 Enhance 后
37/37 条都变差，说明 Enhance 很稳；关闭 Generate 在 REDS 上 27/30 变差，但在 UVG
平均反而略好。人工固定图也看到 Beauty、YachtRide 的跨动作接缝，以及 Jockey 的
Generate 平滑纹理失真。

全部 37 个正式样本、真实字节、fresh decode、固定图和消融均通过完整性复核。墙钟约
2 小时 7 分，峰值 23275 MiB，正式目录约 2.80 GB，三块盘约 12%／17%／26%。这说明
下一轮仍完全可以单卡完成。推荐保留 v5 和 Enhance，只针对 Generate 做更宽／自适应边界
融合、动作图空间一致性和跨域置信度；详细结果见
[`CLOUD_A800_JOINT_EVALUATION.md`](CLOUD_A800_JOINT_EVALUATION.md)。

## 2026-09-19 v6 跨域适配数据调整

REDS 继续作为训练主体，同时把 UVG 的 Beauty、Bosphorus、HoneyBee、Jockey、
ShakeNDry 纳入小比例适配训练；ReadySetGo、YachtRide 不进入本轮训练，作为 v6
留出测试侧序列。七条 UVG 的前 17 帧都已参加过 v5 联合评估，因此后两条只称
“v6 未参与训练”，不声称从未见过。

训练侧每条序列恢复 4 个相隔较远的时段，每个时段取左、中、右三个 512×512 裁剪，
合计 60 个窗口。下载与转换在 tmux 中断点执行，完成一条序列后删除其 7z 和原始 YUV；
DCVC-UF、SeedVR2 和 spatial-QP codec 继续冻结。完整协议见
[`CLOUD_A800_UVG_ADAPTATION.md`](CLOUD_A800_UVG_ADAPTATION.md)。

## 2026-09-19 Generate 羽化诊断完成

冻结 v5 的 37 条已有输出已完成 8／16／32 像素羽化诊断，没有重跑 DCVC-UF 或
SeedVR2，也没有改变动作图。8 像素 37/37 与正式结果逐像素一致；16 像素在 33/33 条
有 Generate 边界的样本上改善边界指标，并使 UVG 平均 LPIPS 降低 0.000866。它让合并
LPIPS 增加 0.001702，而 32 像素代价为 0.005303，因此选择 16 像素作为 v6 默认折中。

固定图显示 Beauty、YachtRide 的交界更柔和，但 Jockey 的平滑背景仍被误生成，说明
羽化只解决贴回边界，不能替代空间一致性和跨域路由校准。完整记录见
[`CLOUD_A800_FEATHER_DIAGNOSTIC.md`](CLOUD_A800_FEATHER_DIAGNOSTIC.md)。

## 2026-09-20 UVG 适配标签与轻量训练完成

五个 UVG 官方 7z 已全部完整展开并在制作 60 个窗口后删除；最终保留 1,020 张
512×512 PNG。60/60 个质量 teacher、60/60 个 ROI 成本 teacher 和两个轻量残差专家均
完成。合并训练集为 REDS 500 + UVG 60 个窗口，UVG 占 10.7%；v1 锚点、v5 选择、预算、
DCVC-UF、SeedVR2 和 spatial-QP codec 都没有改变。

37 条 route-only 回放中，UVG 的 Generate 块从 24 降到 19，HoneyBee 从 4 降到 0；
REDS 的 Generate 总数保持 108。这个结果支持“内容相关的跨域校正”，但还不是画质结论。
详细数据、资源和边界见
[`CLOUD_A800_UVG_ADAPTATION.md`](CLOUD_A800_UVG_ADAPTATION.md)。

下一步把旧的事后孤立块规则改成显式目标：在原有字节／Generate 预算内，最大化区域预测
收益，同时以 `λ=0.004` 轻度惩罚 Generate 与非 Generate 的相邻边。λ 沿用旧 fragment
收益尺度，不设置新的硬门槛；精确动态规划、敏感性回放和 2×2 消融见
[`CLOUD_A800_SPATIAL_CONSISTENCY.md`](CLOUD_A800_SPATIAL_CONSISTENCY.md)。

空间回放随后完成。旧 v5 单加空间项时，Generate 边从 260 降到 207；UVG 适配与空间项
组合时从 237 降到 186，连通块从 51 降到 36，预测收益只下降约 0.35%。两套
`lambda=0` 动作均逐块复现，说明变化来自显式空间目标而非实现漂移。

正式 2×2 画质评估复用了已经验证的 v5 + 16 像素基线，并只在同一样本、同一种子、16 个
动作完全相同时复用新逻辑版本。111 个逻辑目标中有 58 个唯一新动作图需要真实运行；
这些任务现已全部完成。

## 2026-09-20 v6 真实 2×2 完成

combined 平均为 10007.6 B／17 帧、LPIPS 0.446839、PSNR 24.682，相对 v5 + feather16
平均增加 51.1 B，LPIPS 降低 0.002513，平均完整时间减少 6.58%。Generate 边界从 260
降到 186，连通块从 59 降到 36。

结果的重点是跨域差异：UVG 平均 LPIPS 降低 0.013719、6/7 条改善；REDS 平均差异仅
+0.000101，基本持平。空间项单独不改善整体 LPIPS，但显著减少碎片；适配单独改善画质，
二者组合得到四个版本中最低的平均 LPIPS 和最短运行时间。因此在不设硬门槛的口径下，
选择 combined 作为 v6 主方法，其余三项作为消融。

58 个新码流全部通过真实字节复核和逐像素 fresh decode。单张 A800 墙钟 3231 秒，峰值
23275 MiB，正式目录真实普通文件约 1.114 GB；结束时三块盘约 12%／17%／27%。完整结果
见 [`CLOUD_A800_V6_EVALUATION.md`](CLOUD_A800_V6_EVALUATION.md)。

## 2026-09-20 v6 论文证据包完成

当前 `combined` 已冻结，不再用同一批 37 条继续调参。方法图、正文主表、2×2 消融、
REDS／UVG 分数据集表、定性图索引和主张边界已经从正式 JSON 自动生成，保存在
`/root/autodl-fs/DCVC/runs/a800_paper_package_20260920/`。这个步骤没有运行 codec、SeedVR2
或训练，只复核并整理既有证据。完整入口见
[`PAPER_V6_PACKAGE.md`](PAPER_V6_PACKAGE.md)。

下一项正式实验是冻结 v6 的连续长视频机制检查。用户已经确认：该链路跑通后，不需要再
等待一次单独授权，可以继续在同一张 A800 上依次做 spatial-QP-aware DCVC-UF、SeedVR2
和 router 重训；9／17／33 与网格变化只作为轻量消融，不作为微调前的硬门槛。多卡仍未
授权。长视频设计、命令和结果见
[`CLOUD_A800_LONG_VIDEO.md`](CLOUD_A800_LONG_VIDEO.md)。若需要新的大数据，先给用户明确
下载与上传清单，不在服务器上启动慢速大文件下载。

该 33 帧机制检查已于 2026-09-20 完成：一条连续码流包含 1 个 I 和 4 个 P8 单元，
动作图按 unit 更新，fresh decode 逐像素一致；SeedVR2 的 17 帧／步长 8 重叠恢复也已
完成。正式码流 22,707 B，峰值 CUDA allocated 约 19.00 GiB。它只证明接口和恢复调度
可行，不是论文质量结论。当前主线已经转入 spatial-QP-aware DCVC-UF 微调准备，完整
记录见 [`CLOUD_A800_LONG_VIDEO.md`](CLOUD_A800_LONG_VIDEO.md)。

spatial-QP-aware 微调现采用一个较窄的单卡适配方案：每步 17 帧，I + 2×P8，混合动作图
使用区域各自的 lambda，并保留 25% 均匀 QP rehearsal；REDS train 与已划为训练侧的 60
个 UVG adaptation 窗口共同参与。先做两步烟测并用真实码流验证 checkpoint，再安排
512×512 长期训练。协议与恢复命令见
[`CLOUD_A800_SPATIAL_QP_FINETUNE.md`](CLOUD_A800_SPATIAL_QP_FINETUNE.md)。

## 2026-09-20 codec 插值完成与 SeedVR2 LoRA 烟测

冻结／1000-step codec 之间的 0.25／0.50／0.75 插值已经完成。165 个新任务和 110 个复用
端点均有真实码流与 fresh decode。按固定的合并 LPIPS BD-rate 规则选择 alpha=0：alpha=0.25
虽然在 REDS 为 -5.74%，但 UVG 为 +6.36%，合并为 +0.31%；更大的 alpha 跨域退化更明显。
因此 SeedVR2 适配继续使用冻结 codec，微调 codec 保留为 REDS 域内消融。完整表见
[`CLOUD_A800_SPATIAL_QP_INTERPOLATION.md`](CLOUD_A800_SPATIAL_QP_INTERPOLATION.md)。

SeedVR2 LoRA 的 REDS／UVG 双样本真实 cache、2-step 反向传播、恢复点、adapter 保存与重新
加载推理均已通过。峰值训练 CUDA allocated 约 7.00 GiB；adapter 约 11.32 MB。正式方案仍
冻结 3B DiT／VAE 原权重，只训练最后 8 层与输出投影的 2,822,656 个 LoRA 参数。正式运行
使用 560 个 cache、1000 steps、每 25 steps 原子保存，并继续限定为一张 A800。若完成后
SeedVR2 不适合该任务，再保持 codec、route、预算和评价输入不变比较其他恢复后端；当前
实验不取消。详见 [`CLOUD_A800_SEEDVR2_LORA.md`](CLOUD_A800_SEEDVR2_LORA.md)。

## 2026-09-20 SeedVR2 LoRA 正式训练完成

正式单卡流水线已经完成 560/560 条真实 QP8／原视频 latent cache 和 1000/1000 个训练
step。实际抽样为 REDS 756 步、UVG 244 步，全部 loss 与梯度有限；前 25 步平均 loss
1.212808，后 25 步 0.751153。这个下降只说明优化正常，尚不能当作 LPIPS 或视觉质量收益。

平均训练耗时为 0.188 秒／step，峰值 CUDA allocated 约 8.38 GiB；包含 cache 的总墙钟为
4,520 秒。最终 11,319,997 B adapter、step-1000 adapter 和 `resume.pt` 权重逐张量一致，
`run.complete` 已落盘且没有 `run.failed`。正式目录共 578 个普通文件、828,218,240 B，完成
时三块盘约为 12%／17%／31%。下一步是固定输入比较冻结 SeedVR2 与 LoRA；只有评价后才决定
是否重建 teacher、重训 router，或在其余条件不变时更换恢复后端。完整记录见
[`CLOUD_A800_SEEDVR2_LORA.md`](CLOUD_A800_SEEDVR2_LORA.md)。

## 2026-09-21 SeedVR2 LoRA 固定输入评价完成

冻结版和满强度 LoRA 已在同一批 30 条 REDS + 7 条 UVG all-Generate QP8 fresh decode、同一
随机 seed 上完成成对比较。新冻结输出与旧冻结结果 37/37 逐像素一致。LoRA 将合并 LPIPS
从 0.463127 降到 0.444932，并提高 PSNR 0.646 dB；24/37 条 LPIPS 更低、33/37 条 PSNR
更高。REDS LPIPS 改善 0.022806；UVG 平均 LPIPS 微差 0.001568，但 PSNR 提高 2.128 dB、
时序误差降低 0.209。

固定图显示 LoRA 倾向于压掉锐化／幻觉纹理，多数 REDS、Beauty、HoneyBee 和 ReadySetGo
受益，Jockey、ShakeNDry、YachtRide 出现不同程度的过平滑。因此保留 adapter，但不直接把
强度 1.0 设为默认；下一步固定所有其他条件，只比较 0.25／0.50／0.75 推理强度，再把最合适
的折中接回 Generate ROI 与长视频重叠恢复。正式运行耗时 1,055 秒，单卡峰值
`nvidia-smi` 16,875 MiB，结果目录真实 `du -sb` 为 474,001,928 B。完整记录见
[`CLOUD_A800_SEEDVR2_LORA_EVAL.md`](CLOUD_A800_SEEDVR2_LORA_EVAL.md)。

## 背景文档

- [Notion：项目总览](https://app.notion.com/p/3d58b22ebd8d815483aad4e1471ee933)
- [Notion：01 当前论文证据包（主图、主表与边界）](https://app.notion.com/p/3e18b22ebd8d81c281e7d6ae63d9a58e)
- [Notion：02 下一步怎么走与何时转向](https://app.notion.com/p/3d58b22ebd8d8157bfa8ee431ac2d358)
- [Notion：07 数据怎样划分、哪些还不能看](https://app.notion.com/p/3d58b22ebd8d8195a3daebc308d4e354)
- [Notion：09 主线澄清](https://app.notion.com/p/3dc8b22ebd8d81a5b15bd0fe39236b3e)
- [Notion：11 E20–E25 结果与可视化](https://app.notion.com/p/3dc8b22ebd8d81cc84b8d2d658444c6f)
- [Notion：12 实现踩坑与排障](https://app.notion.com/p/3dc8b22ebd8d81d3ae37c7bfd17be89c)
- [Notion：15 v4 融合控制器](https://app.notion.com/p/3df8b22ebd8d81218518fd1c2a967c51)
- [Notion：17 REDS + UVG 联合评估与 Generate 下一步](https://app.notion.com/p/3df8b22ebd8d81389267e7a9579102a6)
- [Notion：18 当前整体方法（论文 Method 结构）](https://app.notion.com/p/3df8b22ebd8d819e8509d1ff2dc97095)
- [Notion：v6 跨域适配、空间一致性与 2×2 验证](https://app.notion.com/p/3e18b22ebd8d8167b0d0d476b3171869)
