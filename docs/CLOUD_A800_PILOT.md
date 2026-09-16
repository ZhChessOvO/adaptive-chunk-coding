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
- 当前 200GB 文件存储挂载在 `/root/autodl-fs`，保存数据、模型和正式输出；50GB 数据盘保存仓库、环境、编译产物和不超过 8GB 的单样本临时缓存；30GB 系统盘不放大文件；
- 文件存储或 50GB 数据盘达到容量的 80% 时停止新样本、保存断点并汇报，不以删除已有正式结果掩盖空间问题。

上传资产和磁盘布局严格按 `docs/CLOUD_STORAGE_AND_UPLOAD.md` 执行，环境、扩展和 checkpoint 安装按 `README.md` 执行。优先使用 `/root/autodl-fs/DCVC/assets` 中已经上传的文件；只下载缺少的小依赖。下载前检查并移除镜像覆盖，不编译当前 bridge 不需要的 Apex 或 FlashAttention。

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

训练结束后可把 REDS `val/000..005` 作为开发集，并明确这样称呼。不要读取 `val/012..023`，不要打开封存的 `val/024..029`。Jockey 等旧材料只用于回归和可视化。

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

## 数据边界

- 训练：REDS `train/000..239`；
- 开发：REDS `val/000..005`，已经多次使用，不能称独立测试；
- 历史已使用：`val/006..011`，本轮无需读取；
- 未读取：`val/012..023`，本轮禁止访问；
- 封存：`val/024..029`，禁止访问；
- Jockey、shake、sky、UVG：只作回归或展示。

数据集和输出不进入 Git。不要记录文件哈希。不要使用 true-fill 冒充结果。

## 背景文档

- [Notion：项目总览](https://app.notion.com/p/3d58b22ebd8d815483aad4e1471ee933)
- [Notion：02 下一步怎么走与何时转向](https://app.notion.com/p/3d58b22ebd8d8157bfa8ee431ac2d358)
- [Notion：07 数据怎样划分、哪些还不能看](https://app.notion.com/p/3d58b22ebd8d8195a3daebc308d4e354)
- [Notion：09 主线澄清](https://app.notion.com/p/3dc8b22ebd8d81a5b15bd0fe39236b3e)
- [Notion：11 E20–E25 结果与可视化](https://app.notion.com/p/3dc8b22ebd8d81cc84b8d2d658444c6f)
- [Notion：12 实现踩坑与排障](https://app.notion.com/p/3dc8b22ebd8d81d3ae37c7bfd17be89c)
