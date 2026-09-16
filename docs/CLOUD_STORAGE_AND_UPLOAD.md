# 云服务器上传与磁盘布局

这份清单针对当前这台服务器：系统盘 30GB、数据盘 50GB、文件存储
200GB，文件存储挂载在 `/root/autodl-fs`。本轮的原则是：

- 数据、模型权重和正式实验输出放在 200GB 文件存储；
- 仓库、Python 环境、源码编译和少量临时文件放在 50GB 数据盘；
- 不把大文件放进 30GB 系统盘；
- 文件存储较慢是可接受的。若 I/O 明显拖慢 GPU，只把当前样本临时缓存到
  数据盘，完成后立即写回文件存储并清理缓存。

## 建议的云端目录

上传时尽量整理成下面的结构。`assets` 是只读输入，`runs` 保存新的正式结果：

```text
/root/autodl-fs/DCVC/
├── assets/
│   ├── checkpoints/
│   │   ├── cvpr2026_image.pth.tar
│   │   └── cvpr2026_video_hts.pth.tar
│   ├── seedvr2/
│   │   ├── seedvr2_ema_3b_bf16.safetensors
│   │   ├── ema_vae.pth
│   │   ├── pos_emb.pt
│   │   └── neg_emb.pt
│   ├── basicvsrpp/
│   │   └── basicvsr_plusplus_c128n25_ntire_decompress_track1_20210223-7b2eba02.pth
│   ├── sources/
│   │   ├── SeedVR2/
│   │   └── cutlass/
│   ├── data/
│   │   └── REDS/
│   │       ├── train_sharp/000..239
│   │       └── val_sharp/000..005
│   ├── regression/
│   │   └── jockey/
│   └── reference/
└── runs/
```

## 从本机上传什么

以下大小是 2026-09-16 在本机实测的近似值。前两组总计约 41GiB。

### 必须上传

| 内容 | 本机路径 | 云端目标 | 约占空间 |
| --- | --- | --- | ---: |
| REDS 训练集 | `/home/czh/data/adaptive_chunk_coding/REDS/train_sharp` | `assets/data/REDS/train_sharp` | 33GB |
| REDS 开发集 000–005 | `/home/czh/data/adaptive_chunk_coding/REDS/val_sharp/000` 到 `005` | `assets/data/REDS/val_sharp/000` 到 `005` | 751MB |
| DCVC-UF 图像模型 | `/home/czh/code/DCVC/checkpoints/cvpr2026_image.pth.tar` | `assets/checkpoints/` | 162MB |
| DCVC-UF HT-S 视频模型 | `/home/czh/code/DCVC/checkpoints/cvpr2026_video_hts.pth.tar` | `assets/checkpoints/` | 310MB |
| 当前实验实际使用的 SeedVR2 BF16 DiT | `/home/czh/code/DCVC/third_party/SeedVR2/ckpts/seedvr2_ema_3b_bf16.safetensors` | `assets/seedvr2/` | 6.4GB |
| SeedVR2 VAE | `/home/czh/code/DCVC/third_party/SeedVR2/ckpts/ema_vae.pth` | `assets/seedvr2/` | 957MB |
| SeedVR2 正文本 embedding | `/home/czh/code/DCVC/third_party/SeedVR2/pos_emb.pt` | `assets/seedvr2/` | 584KB |
| SeedVR2 负文本 embedding | `/home/czh/code/DCVC/third_party/SeedVR2/neg_emb.pt` | `assets/seedvr2/` | 644KB |
| BasicVSR++ 确定性对照权重 | `/home/czh/code/DCVC/third_party/mmagic/checkpoints/basicvsr_plusplus_c128n25_ntire_decompress_track1_20210223-7b2eba02.pth` | `assets/basicvsrpp/` | 169MB |

开发集只上传 `000..005`。不要把 `val_sharp` 整体上传；这样可以从物理上避免云端任务误读
`012..023` 和封存的 `024..029`。

### 建议上传

| 内容 | 本机路径 | 云端目标 | 约占空间 |
| --- | --- | --- | ---: |
| 当前验证过的 SeedVR2 源码 | `/home/czh/code/DCVC/third_party/SeedVR2`，排除 `.git/`、`ckpts/`、`pos_emb.pt` 和 `neg_emb.pt` | `assets/sources/SeedVR2` | 3.8MB |
| CUTLASS 4.4.1 源码 | `/home/czh/code/DCVC/third_party/cutlass`，排除 `.git/` | `assets/sources/cutlass` | 164MB |
| Jockey 回归帧 | `/home/czh/code/DCVC/data/test_sequences/PNG/jockey` | `assets/regression/jockey` | 66MB |

源码包含很多小文件，上传工具若处理得很慢，可以先在本机打成普通 tar 包；到云端后只把源码解包到
50GB 数据盘。模型权重和数据不需要复制到数据盘。

还可以上传下面四个很小的历史结果作为复现参照，总计约 28MB：

- `/home/czh/code/DCVC/output/e24_spatial_budget_summary_jockey`
- `/home/czh/code/DCVC/output/e24_spatial_quality_reds_val000_b050_nearest`
- `/home/czh/code/DCVC/output/e24_seedvr2_spatial_reds_val000_b050_nearest`
- `/home/czh/code/DCVC/output/e25_seedvr2_roi_reds_val000_b050`

放到 `assets/reference/` 即可。这些只是回归参照，不能代替云端 fresh decode。

## 不要上传什么

- 不要上传整个 `/home/czh/data/adaptive_chunk_coding/REDS`。其中含有历史缓存和本轮禁止读取的验证集。
- 不要上传整个 `/home/czh/code/DCVC/checkpoints`。里面约 6.9GB 的 SDXL 是旧路线遗留，本轮不用。
- 不要上传 `seedvr2_ema_3b.pth`。它约 13.6GB；E20–E25 实际使用的是上表的 BF16 权重。
- 不要上传 HT-L、LD、PnP-VCVE、旧 latent predictor checkpoint 或整个 12GB 的 `output/`。
- 不要上传本机 conda 环境或已经编译的 `.so`。CUDA 扩展必须在 A800 服务器上重新编译。
- 不要再次上传主仓库；服务器上按用户安排从 GitHub clone。

## 服务器上的使用方法

先确认三块盘的实际挂载。AutoDL 常见的数据盘路径是 `/root/autodl-tmp`，但必须以
`df` 的真实输出为准：

```bash
df -h / /root/autodl-fs /root/autodl-tmp .
```

若 50GB 数据盘确实是 `/root/autodl-tmp`，建议使用：

```bash
export DCVC_PERSIST=/root/autodl-fs/DCVC
export DCVC_FAST=/root/autodl-tmp/DCVC

mkdir -p "$DCVC_FAST"/{envs,cache,tmp,torch_extensions}
mkdir -p "$DCVC_PERSIST/runs"

export TMPDIR="$DCVC_FAST/tmp"
export CONDA_PKGS_DIRS="$DCVC_FAST/cache/conda-pkgs"
export PIP_CACHE_DIR="$DCVC_FAST/cache/pip"
export TORCH_EXTENSIONS_DIR="$DCVC_FAST/torch_extensions"
export HF_HOME="$DCVC_PERSIST/cache/huggingface"
export TORCH_HOME="$DCVC_PERSIST/cache/torch"
```

仓库也应位于 50GB 数据盘。创建环境时使用显式前缀，避免把环境装进系统盘：

```bash
conda create -p "$DCVC_FAST/envs/dcvcuf" python=3.12 -y
conda activate "$DCVC_FAST/envs/dcvcuf"
```

把上传的两份第三方源码复制或解包到仓库的 `third_party/`，再在服务器上编译：

```bash
mkdir -p third_party/SeedVR2 third_party/cutlass
rsync -a "$DCVC_PERSIST/assets/sources/SeedVR2/" third_party/SeedVR2/
rsync -a "$DCVC_PERSIST/assets/sources/cutlass/" third_party/cutlass/
```

模型、数据和正式输出则保留在文件存储，通过符号链接接入仓库预期位置：

```bash
mkdir -p checkpoints third_party/SeedVR2/ckpts third_party/mmagic/checkpoints data

ln -s "$DCVC_PERSIST/assets/checkpoints/cvpr2026_image.pth.tar" \
  checkpoints/cvpr2026_image.pth.tar
ln -s "$DCVC_PERSIST/assets/checkpoints/cvpr2026_video_hts.pth.tar" \
  checkpoints/cvpr2026_video_hts.pth.tar
ln -s "$DCVC_PERSIST/assets/seedvr2/seedvr2_ema_3b_bf16.safetensors" \
  third_party/SeedVR2/ckpts/seedvr2_ema_3b_bf16.safetensors
ln -s "$DCVC_PERSIST/assets/seedvr2/ema_vae.pth" \
  third_party/SeedVR2/ckpts/ema_vae.pth
ln -s "$DCVC_PERSIST/assets/seedvr2/pos_emb.pt" third_party/SeedVR2/pos_emb.pt
ln -s "$DCVC_PERSIST/assets/seedvr2/neg_emb.pt" third_party/SeedVR2/neg_emb.pt
ln -s "$DCVC_PERSIST/assets/basicvsrpp/basicvsr_plusplus_c128n25_ntire_decompress_track1_20210223-7b2eba02.pth" \
  third_party/mmagic/checkpoints/basicvsr_plusplus_c128n25_ntire_decompress_track1_20210223-7b2eba02.pth
ln -s "$DCVC_PERSIST/assets/data/REDS" data/REDS
ln -s "$DCVC_PERSIST/runs" output
```

需要 Jockey 回归时再创建：

```bash
mkdir -p data/test_sequences/PNG
ln -s "$DCVC_PERSIST/assets/regression/jockey" data/test_sequences/PNG/jockey
```

创建链接前先确认同名路径不存在，不要用强制覆盖。SeedVR2 必须显式传入
`--dit-checkpoint third_party/SeedVR2/ckpts/seedvr2_ema_3b_bf16.safetensors`，以复现 E20–E25。

所有正式 manifest、码流、日志、checkpoint、指标和可视化都写入 `output/`，也就是
`/root/autodl-fs/DCVC/runs`。若慢盘随机 I/O 让任务明显停顿，可以在
`$DCVC_FAST/tmp/<run-id>` 保留不超过 8GB 的单样本临时目录；每个样本结束就把完整结果写回
`runs` 并删除临时目录。不要把整个 REDS 或整轮输出复制到 50GB 数据盘。

运行期间同时监控三个挂载点。文件存储使用率达到 80%，或 50GB 数据盘使用率达到 80%，就停止
新样本、保存断点并汇报；不要靠删除已有正式结果继续掩盖容量问题。
