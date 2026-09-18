# 云服务器上传与磁盘布局

这份说明记录当前这台 AutoDL 服务器的真实布局和已经上传的文件。

## 三块存储的固定用途

- 系统盘：`/root`，30GB。只保留系统和平台自带工具，不在这里创建大环境、下载权重或保存实验结果。
- 数据盘：`/root/autodl-tmp`，50GB。用于 Git 仓库、conda 环境、第三方源码、编译产物和少量临时缓存。
- 文件存储：`/root/autodl-fs`，200GB，读写较慢。用于数据集、模型权重、正式码流、日志、checkpoint、指标和可视化。

慢盘可以直接承载数据与正式输出。若它的随机 I/O 明显拖慢 GPU，只允许把当前样本缓存到数据盘，
缓存上限 8GB；样本结束后立即写回并清理。不要把整个数据集或整轮输出复制到数据盘。

## 最初上传的文件与当前状态

用户最初把以下五个文件直接放在 `/root/autodl-fs/`，没有再分子目录：

```text
/root/autodl-fs/
├── cvpr2026_image.pth.tar
├── cvpr2026_video_hts.pth.tar
├── seedvr2_ema_3b_bf16.safetensors
├── train_sharp.zip
└── val_sharp.zip
```

权重仍从这些路径以符号链接接入。两个 ZIP 已经按用户要求在验证解压后删除，当前根目录
不再有 `train_sharp.zip` 或 `val_sharp.zip`，不要重复下载已有的解压数据。

最初上传的 `val_sharp.zip` 只解压了开发集 `000..005`。一次性独立测试获得授权后，
又从 REDS 作者维护的
[数据集页面](https://seungjunnah.github.io/Datasets/reds.html)所指向的
[官方作者仓库](https://huggingface.co/datasets/snah/REDS)直接下载验证集归档，只选择性
解压 `012..023`，验证完成后删除归档。`024..029` 从未解压或读取。

## 仍需从服务器下载的内容

除上面五个文件外，当前任务需要的其他内容都在服务器上直接下载：

| 内容 | 保存位置 | 说明 |
| --- | --- | --- |
| SeedVR2 源码 | 仓库的 `third_party/SeedVR2` | 从官方 GitHub 克隆，源码和编译工作放数据盘 |
| CUTLASS 4.4.1 | 仓库的 `third_party/cutlass` | 从官方 GitHub 克隆，在 A800 上重新编译扩展 |
| `ema_vae.pth` | `/root/autodl-fs/DCVC/assets/seedvr2/` | 从官方 SeedVR2-3B 下载 |
| `pos_emb.pt`、`neg_emb.pt` | `/root/autodl-fs/DCVC/assets/seedvr2/` | 从官方 SeedVR2-3B 下载 |
| BasicVSR++ 权重 | `/root/autodl-fs/DCVC/assets/basicvsrpp/` | 确定性恢复对照需要 |
| Jockey | `/root/autodl-fs/DCVC/assets/regression/` | 只在需要旧回归对照时下载；不阻塞主试跑 |

当前基线使用已经上传的 BF16 DiT。不要再下载约 13.6GB 的
`seedvr2_ema_3b.pth`，也不要下载旧 SDXL、HT-L、LD、PnP-VCVE 或历史 latent predictor 权重。

下载前检查并移除 conda、pip、Hugging Face 和 Git 的第三方镜像覆盖。具体官方来源和命令见
`README.md`。

## 服务器初始化

先确认挂载和容量：

```bash
df -h /root /root/autodl-tmp /root/autodl-fs .
```

建议固定以下路径：

```bash
export DCVC_UPLOAD=/root/autodl-fs
export DCVC_PERSIST=/root/autodl-fs/DCVC
export DCVC_FAST=/root/autodl-tmp/DCVC

mkdir -p "$DCVC_FAST"/{envs,cache,tmp,torch_extensions}
mkdir -p "$DCVC_PERSIST"/{assets/seedvr2,assets/basicvsrpp,assets/regression,data/REDS,runs}

export TMPDIR="$DCVC_FAST/tmp"
export CONDA_PKGS_DIRS="$DCVC_FAST/cache/conda-pkgs"
export PIP_CACHE_DIR="$DCVC_FAST/cache/pip"
export TORCH_EXTENSIONS_DIR="$DCVC_FAST/torch_extensions"
export HF_HOME="$DCVC_PERSIST/cache/huggingface"
export TORCH_HOME="$DCVC_PERSIST/cache/torch"
```

仓库应位于 `/root/autodl-tmp`。conda 环境使用显式路径，避免占用系统盘：

```bash
conda create -p "$DCVC_FAST/envs/dcvcuf" python=3.12 -y
conda activate "$DCVC_FAST/envs/dcvcuf"
```

## 解压数据

`train_sharp.zip` 可以完整解压，因为 `train/000..239` 都在本轮训练许可内。解压目标是：

```text
/root/autodl-fs/DCVC/data/REDS/train_sharp/000..239
```

先用少量路径检查 ZIP 内的顶层结构，再选择正确的 `unzip -d` 目标，避免得到重复的
`train_sharp/train_sharp`。解压后核对恰好有 `000..239`，不要读取图像内容做额外筛选。

验证集归档必须使用带路径过滤的选择性解压。当前已经合法得到：

```text
/root/autodl-fs/DCVC/data/REDS/val_sharp/000..005
/root/autodl-fs/DCVC/data/REDS/val_sharp/012..023
```

其中 `012..023` 是后来单独授权的一次性测试集，不是开发集。当前目录必须缺少
`006..011` 和封存的 `024..029`。两个 ZIP 均已在完整性检查后删除；只有在得到新的
数据授权且本地确实缺少所需内容时，才从官方来源重新下载相应归档。

## 接入仓库

第三方源码直接克隆到数据盘上的仓库中。模型、数据和正式输出留在文件存储，通过符号链接接入：

```bash
mkdir -p checkpoints third_party/SeedVR2/ckpts third_party/mmagic/checkpoints data

ln -s "$DCVC_UPLOAD/cvpr2026_image.pth.tar" \
  checkpoints/cvpr2026_image.pth.tar
ln -s "$DCVC_UPLOAD/cvpr2026_video_hts.pth.tar" \
  checkpoints/cvpr2026_video_hts.pth.tar
ln -s "$DCVC_UPLOAD/seedvr2_ema_3b_bf16.safetensors" \
  third_party/SeedVR2/ckpts/seedvr2_ema_3b_bf16.safetensors
ln -s "$DCVC_PERSIST/assets/seedvr2/ema_vae.pth" \
  third_party/SeedVR2/ckpts/ema_vae.pth
ln -s "$DCVC_PERSIST/assets/seedvr2/pos_emb.pt" third_party/SeedVR2/pos_emb.pt
ln -s "$DCVC_PERSIST/assets/seedvr2/neg_emb.pt" third_party/SeedVR2/neg_emb.pt
ln -s "$DCVC_PERSIST/assets/basicvsrpp/basicvsr_plusplus_c128n25_ntire_decompress_track1_20210223-7b2eba02.pth" \
  third_party/mmagic/checkpoints/basicvsr_plusplus_c128n25_ntire_decompress_track1_20210223-7b2eba02.pth
ln -s "$DCVC_PERSIST/data/REDS" data/REDS
ln -s "$DCVC_PERSIST/runs" output
```

创建链接前先确认同名路径不存在，不要使用强制覆盖。SeedVR2 推理必须显式传入：

```text
--dit-checkpoint third_party/SeedVR2/ckpts/seedvr2_ema_3b_bf16.safetensors
```

所有正式 manifest、码流、日志、controller checkpoint、指标和可视化写入 `output/`，即
`/root/autodl-fs/DCVC/runs`。两个 ZIP 已删除，但仍要在每个阶段记录 `df -h` 与输出目录大小。
文件存储或数据盘达到 80% 时停止新样本、保存断点并汇报，不删除已有正式结果来掩盖空间问题。
