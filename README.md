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
- 只对 Generate 连通区域运行 SeedVR2 的 ROI 路径。

所有 Base／Enhance 载荷、mask、头部和辅助语法都按真实文件大小计费。Generate 以 LPIPS 等感知指标为主，PSNR 仅作诊断；真实编码信息增加带来的收益不能记为生成收益。

## 云服务器快速恢复

下面按一台全新的 Linux NVIDIA 服务器编写。已经在 Python 3.12、CUDA 13.0、`torch 2.13.0+cu130`、`torchvision 0.28.0+cu130` 上验证；DCVC-UF 上游也说明 Python 3.12、CUDA 13.0 和 PyTorch 2.9.1 可用。CUDA 运行时和本机 `nvcc` 要匹配，建议直接选择带 CUDA 13.0 **devel** 工具链的云镜像。

如果由新的 Codex 会话接手，请先阅读根目录的 `AGENTS.md` 和
[`docs/CLOUD_STORAGE_AND_UPLOAD.md`](docs/CLOUD_STORAGE_AND_UPLOAD.md)、
[`docs/CLOUD_A800_PILOT.md`](docs/CLOUD_A800_PILOT.md) 与
[`docs/CLOUD_A800_FOLLOWUP.md`](docs/CLOUD_A800_FOLLOWUP.md)，再执行本节。当前已完成的云端范围是
**1×A800 80GB、8–12 小时的单卡试跑**：冻结 DCVC-UF 和 SeedVR2，生成有界的反事实标签并训练轻量控制器；多卡完整阶段和大模型／codec 微调仍需下一次决定。
可直接粘贴到新会话的版本保存在
[`docs/CLOUD_CODEX_PROMPT.md`](docs/CLOUD_CODEX_PROMPT.md)。

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

- 当前已上传的权重和两个 ZIP 直接位于 `/root/autodl-fs/`；补充下载和解压后的数据放在 `/root/autodl-fs/DCVC/`，正式结果放在 `/root/autodl-fs/DCVC/runs`。不要把全部数据或正式输出复制到 `/root/autodl-tmp`。详见 [`docs/CLOUD_STORAGE_AND_UPLOAD.md`](docs/CLOUD_STORAGE_AND_UPLOAD.md)。
- `val_sharp.zip` 本轮只解压 `000..005`；其余边界以 `AGENTS.md` 和云端试跑文档为准。
- 数据集不进 Git。只把数据协议允许使用的部分挂载到 `data/`，或通过脚本的 `--data-root`／`--input-dir` 显式传入；不要复制封存数据。
- `output/`、真实码流、PNG／视频、checkpoint、第三方源码和编译产物均已在 `.gitignore` 排除。
- 历史 Stage B 脚本的默认数据目录已改为仓库相对路径 `data/REDS`；也可以用 `--data-root` 指向服务器上的合规数据挂载。当前 Stage C 主线参数使用仓库相对路径或显式输入路径。
- `training.md` 是上游 DCVC-UF 全量训练说明，不是当前控制器入口。单卡试跑和有界后续
  复验均已完成；下一阶段要先确定独立测试数据。不要自行进入多卡完整生产或
  SeedVR2／spatial-QP codec 微调。

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
- `demo/stage_c_seedvr2_three_path_oracle.py`：三种动作的真实码流收益探针；
- `demo/stage_c_spatial_quality_format_test.py`：空间语法和旧格式兼容测试；
- `demo/stage_c_spatial_quality_forward_probe.py`：不训练的空间质量调制检查；
- `demo/stage_c_spatial_quality_codec.py`：一次编码的空间质量写盘／fresh decode；
- `demo/stage_c_evaluate_spatial_quality_codec.py`：LPIPS 优先的三路评价；
- `demo/stage_c_seedvr2_roi.py`：Generate 连通区域准备、单进程常驻恢复、逐像素拼接与算力对比；
- `demo/stage_c_make_visuals.py`：生成 GT、恢复结果和拼接结果的可视化材料。

本仓库基于 [Microsoft DCVC-UF](https://github.com/microsoft/DCVC)；研究代码仍在持续迭代。
