<div align="center">

# Budget-Adaptive Regional Coding and Restoration

**在 DCVC-UF 上进行 Generate / Base / Enhance 区域路由**

</div>

## 当前主线

编码前已经知道带宽和解码算力预算。编码器为每个空间区域选择三种动作之一：

- **Generate**：发送低码率的真实 DCVC-UF 上下文，解码后用视频生成／恢复模型修复最终画面；
- **Base**：发送并解码普通质量的 DCVC-UF 信息；
- **Enhance**：为重要或难恢复区域发送更多真实编码信息。

这不是“省略 latent，再让后训练模型猜回来”的任务。E01–E18 和 latent predictor 代码仅作为历史证据保留，不是当前主线的前置条件。当前实现包含：

- 合法的旧版 fallback：一个普通 Base 流，加若干可独立解码的 Enhance tile 流；
- 一次编码的空间质量格式：两 bit 动作图、明确的质量档位和自描述码流；
- 无源 RGB 的 fresh decode、真实落盘字节计费和完整运行时记录；
- SeedVR2 生成恢复、BasicVSR++ 确定性对照、LPIPS 优先评价和时序诊断；
- 只对 Generate 连通区域运行 SeedVR2 的 ROI 路径；
- 在同一预算下联合优化区域收益和 Generate 边界数量的精确空间一致性求解器；
- 连续长视频路径：每个 I／P8 单元可携带不同动作图，codec 参考环不中断，SeedVR2
  使用重叠 17 帧窗口和固定时间融合，并按 ROI 组件断点续跑。

所有 Base／Enhance 载荷、mask、头部和辅助语法都按真实文件大小计费。Generate 以 LPIPS 等感知指标为主，PSNR 仅作诊断；真实编码信息增加带来的收益不能记为生成收益。

## 云服务器快速恢复

下面按一台全新的 Linux NVIDIA 服务器编写。已经在 Python 3.12、CUDA 13.0、`torch 2.13.0+cu130`、`torchvision 0.28.0+cu130` 上验证；DCVC-UF 上游也说明 Python 3.12、CUDA 13.0 和 PyTorch 2.9.1 可用。CUDA 运行时和本机 `nvcc` 要匹配，建议直接选择带 CUDA 13.0 **devel** 工具链的云镜像。

如果由新的 Codex 会话接手，请先阅读根目录的 `AGENTS.md` 和
[`docs/CLOUD_STORAGE_AND_UPLOAD.md`](docs/CLOUD_STORAGE_AND_UPLOAD.md)、
[`docs/CLOUD_A800_PILOT.md`](docs/CLOUD_A800_PILOT.md) 与
[`docs/CLOUD_A800_FOLLOWUP.md`](docs/CLOUD_A800_FOLLOWUP.md)，再执行本节。当前工作始终限定为
**1×A800 80GB**。冻结 DCVC-UF／SeedVR2 的 controller、真实 spatial-QP 码流和 v6
评估已经完成；用户已授权在长视频链路验证后继续做单卡 spatial-QP-aware codec 与
SeedVR2 微调。多卡生产仍未授权。

1000-step codec 适配与 110 项真实码流对比现已完成：REDS 的等 LPIPS／等 PSNR
BD-rate 分别为 −11.99%／−20.69%，但 UVG 分别退化 +29.71%／+6.39%。因此最终权重
不直接替换冻结版。固定的 0.25／0.50／0.75 权重插值也已完成，按预先固定的跨数据集
LPIPS BD-rate 规则最终选择 alpha=0，即继续使用冻结 codec；中间权重只保留为率失真消融。
SeedVR2 LoRA 的真实 cache、两步反向传播和 adapter 重载烟测均已通过；正式运行随后完成
560/560 条 cache 和 1000/1000 个梯度 step。固定 37 条输入的正式比较也已完成：满强度
LoRA 将合并 LPIPS 从 0.463127 降到 0.444932、PSNR 提高 0.646 dB，24/37 条 LPIPS 更好；
但 UVG 平均 LPIPS 微差 0.001568，固定图显示部分运动纹理会被过度平滑。随后完成的固定强度
扫描表明 0.50 是更实用的折中：合并 LPIPS 为 0.429974（较冻结版降低 0.033152），34/37
条更好，PSNR 提高 0.783 dB、时序误差降低 0.608；它与 0.75 的平均 LPIPS 基本打平，但在
Jockey、ShakeNDry、YachtRide 上更保守。0.50 随后已接回同一条 33 帧连续码流和三个
Generate ROI：LPIPS 再降 0.008922、PSNR 提高 0.142 dB、时序误差降低 0.255；边界带误差
下降，所有非 Generate 像素逐像素不变，32/32 个相邻帧对的时序误差都改善。因此保留 0.50，
下一步用它重建 Generate teacher 并重训 router。如果后续仍不合适，再保持 codec、路由、
预算和评估输入不变，只替换 Generate 恢复后端。

租用 A800 时先用 `nvidia-smi -L` 和 `nvidia-smi --query-gpu=name,memory.total --format=csv` 核对实际可见的是完整 80GB 设备，而不是 MIG 切片。当前服务器的系统盘是 `/root`（30GB），数据盘是 `/root/autodl-tmp`（50GB），较慢的 200GB 文件存储是 `/root/autodl-fs`。环境和编译放数据盘；数据、模型和正式输出放文件存储。用户上传的五个文件直接平铺在文件存储根目录，具体清单、缺失下载、解压边界和链接方式见云端存储文档。

### 1. 克隆代码并检查下载源

```bash
git clone https://github.com/ZhChessOvO/adaptive-chunk-coding.git
cd adaptive-chunk-coding

conda config --show-sources
python3 -m pip config list -v
env | grep -E '^(PIP_INDEX_URL|PIP_EXTRA_INDEX_URL)='
```

先查看以上输出。如果 `.condarc`、`pip.conf` 或环境变量中有第三方镜像，先删除对应条目，再下载依赖和权重。下面的 PyTorch 命令显式使用官方 wheel 源。

Ubuntu 上建议先准备基础工具：

```bash
sudo apt-get update
sudo apt-get install -y build-essential git wget ffmpeg
nvidia-smi
nvcc --version
```

### 2. 创建 Python 环境

当前 AutoDL 服务器不要使用默认命名环境占用系统盘。先按
[`docs/CLOUD_STORAGE_AND_UPLOAD.md`](docs/CLOUD_STORAGE_AND_UPLOAD.md) 设置 50GB 数据盘路径，
再用 `conda create -p ...` 创建环境。其他服务器可以使用下面的普通命名环境：

```bash
conda create -n dcvcuf python=3.12 -y
conda activate dcvcuf
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.13.0 torchvision==0.28.0 \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-research.txt
```

`requirements.txt` 是 DCVC-UF 的基础依赖；`requirements-research.txt` 额外记录当前恢复、LPIPS 和 SeedVR2 bridge 实际验证过的版本。项目不需要 `torchaudio`。

### 3. 编译 DCVC-UF 的两个扩展

需要编译：

1. `MLCodec_extensions_cpp`：CPU rANS 熵编码器；
2. `inference_extensions_cuda`：基于 CUTLASS 的 CUDA 推理算子。

```bash
git clone --depth 1 --branch v4.4.1 \
  https://github.com/NVIDIA/cutlass.git third_party/cutlass

(cd src/cpp && bash install.sh)
(cd src/layers/extensions/inference && bash install.sh)
```

CUDA 扩展会针对编译时可见的 GPU 架构生成代码，所以应当在最终使用的云机器上编译，并保证至少一张 GPU 可见。如果内存较小，可在第二条命令前设置 `MAX_JOBS=4`。

编译后检查：

```bash
python -c "import torch, MLCodec_extensions_cpp, inference_extensions_cuda; print(torch.__version__, torch.cuda.get_device_name(0))"
python demo/stage_c_spatial_quality_format_test.py
```

第二条是轻量语法 round-trip 测试，不需要数据集或模型权重。

## 模型源码和 checkpoint

完整可运行目录应当接近：

```text
adaptive-chunk-coding/
├── checkpoints/
│   ├── cvpr2026_image.pth.tar
│   └── cvpr2026_video_hts.pth.tar
└── third_party/
    ├── cutlass/
    ├── SeedVR2/
    │   ├── ckpts/
    │   │   ├── seedvr2_ema_3b.pth
    │   │   ├── seedvr2_ema_3b_bf16.safetensors
    │   │   └── ema_vae.pth
    │   ├── pos_emb.pt
    │   └── neg_emb.pt
    └── mmagic/
        └── checkpoints/
            └── basicvsr_plusplus_c128n25_ntire_decompress_track1_20210223-7b2eba02.pth
```

### DCVC-UF（必需）

从 [DCVC-UF 官方 OneDrive](https://1drv.ms/f/c/2866592d5c55df8c/IgAalzb_985lQ79GkXyW2P5OASPpZHHcrcGWEVQxO-mQCVg?e=qyvMN6) 下载预训练模型，若主链接不可用可用[官方备用链接](https://1drv.ms/f/c/2866592d5c55df8c/EozfVVwtWWYggCitBAAAAAABbT4z2Z10fMXISnan72UtSA?e=BID7DA)。放入仓库根目录的 `checkpoints/`。

当前主线至少需要：

- `checkpoints/cvpr2026_image.pth.tar`
- `checkpoints/cvpr2026_video_hts.pth.tar`

上游压缩包还包含 HT-L 和 LD 视频模型；只有切换 `--model-structure` 时才需要对应权重。

### SeedVR2-3B（Generate 路径必需）

先克隆 [ByteDance SeedVR 官方源码](https://github.com/ByteDance-Seed/SeedVR)：

```bash
git clone https://github.com/ByteDance-Seed/SeedVR.git third_party/SeedVR2
mkdir -p third_party/SeedVR2/ckpts
```

再从 [ByteDance-Seed/SeedVR2-3B](https://huggingface.co/ByteDance-Seed/SeedVR2-3B) 下载官方 3B DiT、VAE 和文本 embedding：

```bash
hf download ByteDance-Seed/SeedVR2-3B \
  seedvr2_ema_3b.pth ema_vae.pth \
  --local-dir third_party/SeedVR2/ckpts

hf download ByteDance-Seed/SeedVR2-3B \
  pos_emb.pt neg_emb.pt \
  --local-dir third_party/SeedVR2
```

四个文件合计约 14.6 GB。`demo/stage_c_seedvr2_bridge.py` 保留官方结构和权重，但用 PyTorch SDPA 与参数兼容的 norm 替代 Apex／FlashAttention，因此当前路径**不需要编译 Apex 或 FlashAttention**。

当前 AutoDL 服务器已经在 `/root/autodl-fs/` 上传 BF16 DiT，不要再执行上面的整组下载。只从官方仓库补下 `ema_vae.pth`、`pos_emb.pt` 和 `neg_emb.pt`，保存位置按 `docs/CLOUD_STORAGE_AND_UPLOAD.md` 执行。实际推理必须用单进程 `torchrun` 启动：

```bash
torchrun --standalone --nproc-per-node=1 demo/stage_c_seedvr2_bridge.py \
  --input-dir /path/to/decoded_pngs \
  --output-dir output/seedvr2_smoke \
  --max-frames 9 \
  --sample-steps 1
```

E20–E25 的现有结果实际使用 [社区 BF16 转换](https://huggingface.co/szwagros/SeedVR2-3B-bf16)。当前 A800 试跑继续使用它，保证和本地基线一致；它不是字节跳动官方发布物：

```bash
hf download szwagros/SeedVR2-3B-bf16 \
  seedvr2_ema_3b_bf16.safetensors \
  --local-dir third_party/SeedVR2/ckpts
```

使用时增加：

```text
--dit-checkpoint third_party/SeedVR2/ckpts/seedvr2_ema_3b_bf16.safetensors
```

### BasicVSR++（确定性对照需要）

项目已经跟踪 `demo/standalone_basicvsrpp.py`，使用 `torchvision.ops.deform_conv2d`，不需要安装 MMCV，也不需要克隆整个 MMagic。只需下载 [MMagic 官方 compressed-video Track 1 权重](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c128n25_ntire_decompress_track1_20210223-7b2eba02.pth)：

```bash
mkdir -p third_party/mmagic/checkpoints
wget -c \
  https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c128n25_ntire_decompress_track1_20210223-7b2eba02.pth \
  -O third_party/mmagic/checkpoints/basicvsr_plusplus_c128n25_ntire_decompress_track1_20210223-7b2eba02.pth
```

适配器来源和修改说明记录在 `NOTICE.txt`。PnP-VCVE、SDXL 和历史 predictor checkpoint 都不是当前主线的必需项。

## 数据、输出与继续实验

- 用户上传的两个 ZIP 已在验证解压后删除；2026-09-18 已从 REDS 作者指向的官方仓库补齐 validation，当前 `val_sharp/000..029` 共 3000 张 PNG 完整。解压后的数据放在 `/root/autodl-fs/DCVC/`，正式结果放在 `/root/autodl-fs/DCVC/runs`。不要把全部数据或正式输出复制到 `/root/autodl-tmp`。详见 [`docs/CLOUD_STORAGE_AND_UPLOAD.md`](docs/CLOUD_STORAGE_AND_UPLOAD.md)。
- REDS `val/000..005` 是开发集，`006..023` 已影响历史实验；`024..029` 可以用于后续论文评估，但必须在读取前固定版本、帧窗和裁剪，并在用结果改方案后把它改记为“已用于开发”。
- UVG、QST 等外部视频与 REDS validation 一起构成跨分布论文评估池；同时报告各数据集和合并结果，不再把外部视频单列成“应用测试”。
- 数据集不进 Git。通过 `data/` 挂载或脚本的 `--data-root`／`--input-dir` 显式传入，并在 manifest 中记录每条样本的来源和角色。
- `output/`、真实码流、PNG／视频、checkpoint、第三方源码和编译产物均已在 `.gitignore` 排除。
- 历史 Stage B 脚本的默认数据目录已改为仓库相对路径 `data/REDS`；也可以用 `--data-root` 指向服务器上的合规数据挂载。当前 Stage C 主线参数使用仓库相对路径或显式输入路径。
- `training.md` 是上游 DCVC-UF 全量训练说明，不是当前控制器入口。单卡试跑、有界后续
  复验和既有评估均已完成。后续仍只在单张 A800 上推进；长视频机制确认后，已获授权
  依次尝试 spatial-QP-aware DCVC-UF、SeedVR2 和 router 重训。未经新决定不进入多卡。

### A800 单卡试跑状态

2026-09-17 的单卡试跑已经完成：冻结模型生成了 500 个训练样本和 6 个开发样本的
反事实标签，并训练了 5768 参数的预算条件化 MLP。六条 REDS 开发视频上，学习到的
联合路由平均使用 13771.5 个真实落盘字节／17 帧，LPIPS 为 0.432362；字节最接近的
均匀 QP24 为 13571.8 字节、LPIPS 0.472186。所有正式码流均通过独立进程 fresh
decode 和逐像素一致性检查。

这仍是开发集信号，不是独立测试。第一轮结束时，中预算路由没有选择 Base，且多个 Generate ROI
使完整解码比全画面 SeedVR2 更慢；下一轮应先在单卡上修正动作平衡和 ROI 调度，不直接
扩大到多卡。完整指标、资源记录和复现边界见
[`docs/CLOUD_A800_PILOT.md`](docs/CLOUD_A800_PILOT.md)。

2026-09-18 的单卡后续复验进一步拆开了这两个问题。没有重新训练或调参，只选择此前
已经固定的低预算点；它在 96 个区域中自然产生 Base 24、Generate 24、Enhance 48。
低预算联合路由平均为 10497.5 B／17 帧、LPIPS 0.463212，逐样本相对最近的已测均匀
QP 点平均改善 0.058304 LPIPS，6/6 个样本方向一致。关闭 Generate 或 Enhance 后分别变差 0.022021 和
0.046784，说明低预算点也保留了两条分支的可测贡献。

ROI 执行改为一个进程只加载一次 SeedVR2 后，中预算输出在 6/6 个样本上与旧执行器
逐像素一致，平均完整时间由 37.923 秒降到 24.895 秒；这比全画面 Generate 的
28.248 秒快 11.87%，逐样本为 5/6 更快。正式均值处于已预热文件缓存的共同基线下；服务器重启后的单个
冷启动回归另记为 51.52 秒，其中模型加载 42.70 秒。新正式评估峰值为 17.65 GiB，
仍只需要一张 A800。详细协议与结果见
[`docs/CLOUD_A800_FOLLOWUP.md`](docs/CLOUD_A800_FOLLOWUP.md)。

2026-09-18 随后在此前未读取的 REDS `val/012..023` 上完成了预注册的一次性独立测试。
中预算联合路线平均 17193.4 B／17 帧、LPIPS 0.451470，相对逐样本最近的已测均匀
QP 平均改善 0.024350，12/12 条更好；关闭 Generate／Enhance 分别有 10/12、12/12
条变差，平均完整时间 21.602 秒，比全 Generate 快 26.55%，因此四项预注册判据全部
通过。

低预算平均 LPIPS 也改善 0.007339，两项分支消融和速度判据通过，但只有 5/12 条优于
最近均匀 QP，未达到至少 8/12 的稳定性门槛，所以低预算总判定为失败。所有正式流均
通过真实字节和逐像素 fresh decode 检查；正式峰值显存 18.33 GiB，总墙钟约 1 小时
47 分，继续使用单张 A800 足够。`012..023` 已消费，`024..029` 仍封存。完整协议与
结果见 [`docs/CLOUD_A800_INDEPENDENT_TEST.md`](docs/CLOUD_A800_INDEPENDENT_TEST.md)。

同日又只用既有训练／开发数据做了两轮低预算稳定性改进。v2 的同画面上下文特征在训练
集分组交叉验证中只把 oracle regret 降低 0.62%，没有达到预先规定的 10%，因此未进入
开发集。v3 改用可部署的编码端两遍分析：先临时做一次统一 QP16 Base 编解码，再用局部
重建失真选 B/G/E。它在训练集把 regret 降低 18.83% 并过门槛，但六条开发视频上平均
10349.0 B／17 帧、LPIPS 0.470660，虽以 6/6 胜过最近均匀 QP，却比相近字节的 v1
差 0.007448 LPIPS，因此没有进入新的独立测试。两项协议与完整负结果见
[`docs/CLOUD_A800_LOW_BUDGET_V2.md`](docs/CLOUD_A800_LOW_BUDGET_V2.md) 和
[`docs/CLOUD_A800_LOW_BUDGET_V3.md`](docs/CLOUD_A800_LOW_BUDGET_V3.md)。

随后不再设置硬性门槛，完成了 v4 锚定式融合：让 v1 保持默认判断，v2 上下文和 v3
Base 预分析只学习纠错。训练集分组折外自动选中二者融合版本，oracle regret 从 v1 的
0.113505 降到 0.093707；但冻结后在六条开发视频上为 10594.2 B／17 帧、LPIPS
0.468980，比 v1 多 96.7 B 且差 0.005769，只有 2/6 条质量更好。关闭 Generate／Enhance
后均为 6/6 变差，说明信号和两条分支都有贡献，但纠错尚未稳定迁移到完整视频。因此当前
低预算主版本仍选择 v1，v4 不进入新独立测试；完整方法、负结果和资源记录见
[`docs/CLOUD_A800_LOW_BUDGET_V4.md`](docs/CLOUD_A800_LOW_BUDGET_V4.md)。

在此基础上又完成了 v5 保守共识：v2 上下文专家与 v3 Base-probe 专家分别预测对 v1 的
纠错，只有方向一致且各自不确定性足够小时才采用较小的可信幅度。训练集分组折外自动选择
`z=0.25`、纠错比例 `1.0`；六条开发视频上平均为 10429.5 B／17 帧、LPIPS 0.461972，
相对 v1 少 68 B、LPIPS 低 0.001240，但逐视频只有 3/6 更好。关闭 Generate／Enhance 后
分别平均变差 0.023291／0.048774，均为 6/6。按“不设硬门槛、综合选择”的口径冻结 v5
后，已经完成 REDS validation 30 条 + UVG 7 条联合论文评估。

联合 37 条平均为 9956.5 B／17 帧、LPIPS 0.447651，相对逐样本最近均匀 QP 改善
0.014411，21/37 条更好。分开看更有意义：REDS 平均改善 0.020686，新读取的
`024..029` 为 6/6 改善；UVG 平均反而差 0.012478，只有 2/7 改善。Enhance 在
37/37 条上都有贡献，而 Generate 在 REDS 上大多有益、在 UVG 上平均略有害；固定图还
显示部分 UVG 的跨动作接缝。完整协议见
[`docs/CLOUD_A800_LOW_BUDGET_V5.md`](docs/CLOUD_A800_LOW_BUDGET_V5.md)，联合结果、资源和
下一步见 [`docs/CLOUD_A800_JOINT_EVALUATION.md`](docs/CLOUD_A800_JOINT_EVALUATION.md)。

随后完成的冻结羽化诊断复用了上述 37 条已有输出，没有重跑 codec 或 SeedVR2。16 像素
在 33/33 条有 Generate 边界的样本上降低边界梯度误差，并改善 UVG 平均 LPIPS；合并
LPIPS 相比 8 像素增加 0.001702，明显小于 32 像素的 0.005303 代价。因此 v6 选择
16 像素作为默认折中，并继续单独修正 Generate 误选。详见
[`docs/CLOUD_A800_FEATHER_DIAGNOSTIC.md`](docs/CLOUD_A800_FEATHER_DIAGNOSTIC.md)。

五条 UVG 训练侧序列随后完整恢复为 60 个跨域适配窗口。60/60 个质量标签、60/60 个 ROI
成本标签和两个轻量残差专家已完成；37 条路由回放中，UVG 的 Generate 块从 24 降到 19，
其中 HoneyBee 从 4 降到 0，而 REDS 的 Generate 总数保持 108。详情见
[`docs/CLOUD_A800_UVG_ADAPTATION.md`](docs/CLOUD_A800_UVG_ADAPTATION.md)。该适配已与精确
Generate 边界正则组成正式 2×2；组合路由把 37 条中的 Generate 边从 237 降到 186，预测
收益只减少约 0.35%。方法与 λ 来源见
[`docs/CLOUD_A800_SPATIAL_CONSISTENCY.md`](docs/CLOUD_A800_SPATIAL_CONSISTENCY.md)，真实
spatial-QP、fresh decode 和 2×2 消融结果见
[`docs/CLOUD_A800_V6_EVALUATION.md`](docs/CLOUD_A800_V6_EVALUATION.md)。

v6 的 58 个新真实动作图已经在单张 A800 上完成，另有 53 个逻辑目标仅在 16 个动作完全
相同时严格复用。combined 相对 v5 平均只增加 51.1 B／17 帧，LPIPS 从 0.449352 降到
0.446839、PSNR 从 24.499 提到 24.682，Generate 边界从 260 降到 186、连通块从 59 降到
36，平均完整时间减少 6.58%。收益主要来自 UVG（LPIPS −0.013719、6/7 改善），REDS
基本持平（+0.000101）。因此当前选择 combined 作为论文主方法，其他三项作为 2×2 消融；
这证明了方法链有用，但不宣称每条视频都改善。

当前结果已经整理成可重复生成的论文证据包：方法总图、正文主表、2×2 消融、跨数据集表、
定性图索引和“主张—证据—不能夸大之处”均直接从冻结 JSON 生成，不手抄数字，也不重新
运行模型。入口见 [`docs/PAPER_V6_PACKAGE.md`](docs/PAPER_V6_PACKAGE.md)，正式小型材料保存在
`/root/autodl-fs/DCVC/runs/a800_paper_package_20260920/`。通俗图文入口为
[Notion：01 当前论文证据包](https://app.notion.com/p/3e18b22ebd8d81c281e7d6ae63d9a58e)。

连续长视频机制也已完成：33 帧写入同一条 1×I + 4×P8 码流，动作图在中途更新而
codec 参考状态不断开；独立 fresh decode 逐像素一致。SeedVR2 用 0／8／16 三个重叠
17 帧窗口恢复，边界融合相对硬切换保持近似画质并略降切换处时序误差。正式码流
22,707 B，峰值 CUDA allocated 约 19.00 GiB，结果目录约 73.24 MB。它是机制 smoke，
不是独立论文比较。完整记录见
[`docs/CLOUD_A800_LONG_VIDEO.md`](docs/CLOUD_A800_LONG_VIDEO.md)，当前下一步是
spatial-QP-aware DCVC-UF 单卡微调。微调的数据、区域 rate-distortion 损失、断点格式和
tmux 入口见
[`docs/CLOUD_A800_SPATIAL_QP_FINETUNE.md`](docs/CLOUD_A800_SPATIAL_QP_FINETUNE.md)。

## 主要脚本

- `demo/stage_c_a800_sample_manifest.py`：冻结 500 个训练裁剪和 6 个开发裁剪的确定性账本；
- `demo/stage_c_a800_teacher.py`：可断点续跑的冻结模型反事实质量／字节标签；
- `demo/stage_c_a800_roi_cost_teacher.py`：实测四类连通 ROI 几何的 SeedVR2 单卡耗时，不用全画面时间按面积估算；
- `demo/stage_c_a800_controller.py`：固定日程训练线性模型和小型 MLP，并在带宽／Generate tile 预算下求解动作；
- `demo/stage_c_a800_route_variants.py`：生成学习联合路由、全 Generate、Enhance-only 及关闭 G／E 的同路由消融；
- `demo/stage_c_a800_scalar_fresh_decode.py`：为每个均匀 QP／BasicVSR++ 对照启动独立进程，记录包含模型加载的完整 fresh-decode 时间；
- `demo/stage_c_a800_evaluate_variant.py`、`demo/stage_c_a800_formal_summary.py`：复核真实落盘字节、fresh decode、LPIPS／PSNR／时序指标并汇总六条开发视频；
- `demo/stage_c_a800_formal_visual.py`：在固定第 9 帧生成包含基线、联合策略、消融和动作图的完整对照；
- `demo/stage_c_a800_followup_summary.py`、`demo/stage_c_a800_followup_visual.py`：汇总低预算三路复验、常驻 ROI 执行和固定 12 宫格；
- `demo/stage_c_a800_compare_frames.py`：对新旧执行器做逐像素序列回归，不记录文件哈希；
- `demo/stage_c_a800_followup_finalize.py`：复核完成标志、单卡边界、显存、耗时和三个挂载点；
- `demo/stage_c_a800_independent_manifest.py`：在读取图片前固定 12 条独立测试样本；
- `demo/stage_c_a800_independent_summary.py`、`demo/stage_c_a800_independent_visual.py`：
  汇总预注册判据并生成固定 17 面板；
- `demo/stage_c_a800_independent_finalize.py`：复核 12 个完成标志、fresh decode、资源和
  数据封存边界；
- `demo/run_stage_c_a800_independent_test.sh`：tmux 中可断点续跑的一次性独立测试入口；
- `demo/stage_c_a800_low_budget_v2.py`、`demo/run_stage_c_a800_low_budget_v2.sh`：按原视频
  分组交叉验证上下文特征，并在训练门槛失败时自动停止；
- `demo/stage_c_a800_low_budget_v3.py`、`demo/run_stage_c_a800_low_budget_v3.sh`：训练并
  冻结编码端 Base 预分析控制器；
- `demo/run_stage_c_a800_low_budget_v3_formal.sh`、
  `demo/stage_c_a800_low_budget_v3_summary.py`：可断点续跑的六条开发复验、消融、固定
  可视化和预注册判据汇总；
- `demo/stage_c_a800_low_budget_v4.py`、`demo/run_stage_c_a800_low_budget_v4.sh`：以 v1
  为锚点，比较 v2 上下文、v3 Base 预分析及二者融合的训练折外残差纠错；
- `demo/run_stage_c_a800_low_budget_v4_formal.sh`、
  `demo/stage_c_a800_low_budget_v4_summary.py`：运行唯一冻结的 v4 候选，汇总真实字节、
  fresh decode、两项消融、固定可视化和资源边界；
- `demo/stage_c_a800_low_budget_v5.py`、`demo/run_stage_c_a800_low_budget_v5.sh`：以 v1 为
  默认答案，只在 v2 上下文与 v3 Base-probe 两个专家可信同向时作保守纠错；
- `demo/run_stage_c_a800_low_budget_v5_formal.sh`、
  `demo/stage_c_a800_low_budget_v5_summary.py`：运行冻结的 v5 开发闭环并汇总真实字节、
  两项消融、fresh decode、固定图和资源；
- `demo/restore_uvg_evaluation_samples.sh`：从 UVG 官方站断点恢复七条标准序列，验证原始
  YUV 大小后固定前 17 帧中央裁剪，并删除临时归档和 YUV；
- `demo/prepare_uvg_adaptation_samples.sh`：从 UVG 官方站断点恢复五条 v6 训练侧序列的
  更多时段，生成 60 个跨域适配窗口；ReadySetGo、YachtRide 不进入本轮训练。协议见
  [`docs/CLOUD_A800_UVG_ADAPTATION.md`](docs/CLOUD_A800_UVG_ADAPTATION.md)；
- `demo/run_stage_c_a800_uvg_adaptation.sh`、`demo/stage_c_a800_uvg_adaptation.py`：在单张
  A800 上生成 UVG 质量／ROI 标签，与 REDS 标签无复制合并，只重训 v5 残差专家，并先
  汇总 37 条样本的路由变化；
- `demo/stage_c_a800_spatial_consistency.py`：在不改变控制器预测和预算的前提下，用精确
  frontier 动态规划联合优化区域收益与 Generate／非 Generate 边界；
- `demo/run_stage_c_a800_spatial_consistency.sh`：对旧 v5 和 UVG 适配后路由执行 λ 敏感性
  回放、`λ=0` 动作回归和空间项 2×2 消融所需的两组路由；
- `demo/stage_c_a800_v6_evaluation_plan.py`、
  `demo/run_stage_c_a800_v6_evaluation.sh`：冻结四版本精确动作去重计划，并在 tmux 中可断点
  运行适配×空间项的真实 spatial-QP、fresh decode 和 SeedVR2 评估；
- `demo/stage_c_a800_v6_evaluation_summary.py`：核对真实字节和复用关系，汇总 2×2 因子
  效果、边界指标，并生成四版本输出与动作图的固定对照图；
- `demo/stage_c_a800_paper_package.py`：从已完成的 v6 JSON 生成方法图、LaTeX 主表／消融表、
  跨数据集图和主张证据矩阵；它只做验证与整理，不运行 codec、SeedVR2 或训练；
- `demo/stage_c_long_video_plan.py`、`demo/stage_c_long_video_seedvr2.py`：把连续 I／P8
  route 计划、重叠 SeedVR2 ROI 恢复、时间融合和断点续跑连成一条长视频路径；
- `demo/run_stage_c_a800_long_video_smoke.sh`：单张 A800 上可恢复的 33 帧连续机制验证；
- `demo/stage_c_spatial_qp_finetune.py`、
  `demo/run_stage_c_a800_spatial_qp_finetune.sh`：从官方 image／HT-S checkpoint 出发，
  使用区域 lambda、混合动作图和均匀 QP rehearsal 做可恢复的单卡 codec 适配，并以真实
  spatial-QP 码流和 fresh decode 验证导出物；
- `demo/stage_c_spatial_qp_finetune_eval.py`、
  `demo/run_stage_c_a800_spatial_qp_finetune_eval.sh`：冻结 37 条 combined routes 和 6 条
  三档均匀 QP 回归，在训练后自动比较原始／微调 codec 的真实字节、画质、时序与边界；
- `demo/stage_c_spatial_qp_interpolation.py`、
  `demo/run_stage_c_a800_spatial_qp_interpolation.sh`：在冻结与 1000-step codec 权重之间固定
  测试 0.25／0.50／0.75，使用 REDS + UVG 三点 BD-rate 选择跨风格折中，并对同一批
  mixed routes 做真实码流确认；协议见
  [`docs/CLOUD_A800_SPATIAL_QP_INTERPOLATION.md`](docs/CLOUD_A800_SPATIAL_QP_INTERPOLATION.md)；
- `demo/stage_c_seedvr2_lora_finetune.py`、
  `demo/run_stage_c_a800_seedvr2_lora.sh`：选定 codec 后缓存真实 QP8／原视频 VAE latent，
  在 SeedVR2-3B 最后 8 层训练 rank-8 LoRA；原 DiT／VAE 冻结，正式训练可断点续跑；协议见
  [`docs/CLOUD_A800_SEEDVR2_LORA.md`](docs/CLOUD_A800_SEEDVR2_LORA.md)；
- `demo/stage_c_seedvr2_lora_eval.py`、
  `demo/run_stage_c_a800_seedvr2_lora_eval.sh`：复用 37 条已验证 QP8 spatial-QP 输入和逐样本
  随机种子，成对比较冻结 SeedVR2 与正式 LoRA，并按 REDS／UVG／UVG 训练角色汇总；协议见
  [`docs/CLOUD_A800_SEEDVR2_LORA_EVAL.md`](docs/CLOUD_A800_SEEDVR2_LORA_EVAL.md)；
- `demo/stage_c_seedvr2_lora_strength_sweep.py`、
  `demo/run_stage_c_a800_seedvr2_lora_strength.sh`：复用强度 0／1 的已验证输出，只补跑同一
  adapter 的 0.25／0.50／0.75 推理强度，比较去伪影与保纹理的折中；协议见
  [`docs/CLOUD_A800_SEEDVR2_LORA_STRENGTH.md`](docs/CLOUD_A800_SEEDVR2_LORA_STRENGTH.md)；
- `demo/stage_c_seedvr2_lora_roi_long_eval.py`、
  `demo/run_stage_c_a800_seedvr2_lora_roi_long.sh`：固定既有 33 帧连续码流、v6 动作图、ROI、
  seed 和羽化，只把选定的 LoRA 0.50 接回局部恢复，检查非 Generate 像素回归、空间接缝与
  时间切换；协议见
  [`docs/CLOUD_A800_SEEDVR2_LORA_ROI_LONG.md`](docs/CLOUD_A800_SEEDVR2_LORA_ROI_LONG.md)；
- `demo/stage_c_a800_feather_verify.py`：独立复核 37 条羽化诊断、8 像素逐像素回归、保存
  帧与 SHA-256，并汇总不同羽化下 Generate 的真实贡献；
- `demo/run_stage_c_a800_joint_evaluation.sh`、
  `demo/stage_c_a800_joint_evaluation_summary.py`：在 tmux 中可断点续跑 REDS validation
  与 UVG 联合评估，按数据集和合并口径汇总，并只在路由完全相同时复用历史正式输出；
- `demo/stage_c_seedvr2_three_path_oracle.py`：三种动作的真实码流收益探针；
- `demo/stage_c_spatial_quality_format_test.py`：空间语法和旧格式兼容测试；
- `demo/stage_c_spatial_quality_forward_probe.py`：不训练的空间质量调制检查；
- `demo/stage_c_spatial_quality_codec.py`：一次编码的空间质量写盘／fresh decode；
- `demo/stage_c_evaluate_spatial_quality_codec.py`：LPIPS 优先的三路评价；
- `demo/stage_c_seedvr2_roi.py`：Generate 连通区域准备、单进程常驻恢复、逐像素拼接与算力对比；
- `demo/stage_c_make_visuals.py`：生成 GT、恢复结果和拼接结果的可视化材料。

本仓库基于 [Microsoft DCVC-UF](https://github.com/microsoft/DCVC)；研究代码仍在持续迭代。
