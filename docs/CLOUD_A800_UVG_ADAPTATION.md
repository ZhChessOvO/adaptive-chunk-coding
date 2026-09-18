# A800 单卡 UVG 跨域适配

## 目的

v5 控制器只在 REDS train 上学习。37 条联合评估中，Enhance 在 REDS 和 UVG 上都稳定
有效，但 Generate 在 UVG 上平均略有害。本轮不微调 DCVC-UF 或 SeedVR2，而是给轻量
控制器补充少量 UVG 反事实标签，让它学会不同视频风格下何时不应选择 Generate。

这不是把 UVG 全部并入训练。REDS 仍是训练主体，UVG 只做约一成的跨域适配。

## 数据划分

| 角色 | UVG 序列 | 本轮用法 |
|---|---|---|
| v6 适配训练 | Beauty、Bosphorus、HoneyBee、Jockey、ShakeNDry | 恢复更多时段，生成 teacher 与 ROI 成本标签 |
| v6 留出测试侧 | ReadySetGo、YachtRide | 不进入 v6 标签、训练或参数选择 |

七条 UVG 的第 0--16 帧都已经参加过 v5 联合评估。因此这里的“留出”只表示**不参与
v6 训练**，不能写成从未看过的盲测。旧结果无需撤回；后续报告应分别列出 REDS、UVG
训练侧序列、UVG 留出序列和合并结果。

QST 当前没有可登记的官方 clip ID，本轮不使用含糊的旧别名代替。

## 适配样本怎样组成

每条训练侧序列取 4 个彼此分开的 17 帧时段，每个时段取左、中、右三个 512×512 裁剪：

- 600 帧序列从第 32、160、288、416 帧开始；
- 300 帧 ShakeNDry 从第 32、96、160、224 帧开始；
- 三个裁剪的左上角为 `(128, 284)`、`(704, 284)`、`(1280, 284)`；
- 共 5×4×3=60 个样本、1,020 张 PNG、960 个区域标签。

原 REDS teacher 有 500 个窗口、8,000 个区域标签。加入 60 个 UVG 窗口后，UVG 约占
窗口总数的 10.7%，足以提供域信息，又不至于覆盖 REDS 主任务。

## 下载、空间和断点续跑

准备脚本为 `demo/prepare_uvg_adaptation_samples.sh`。它从 Ultra Video Group 官方服务器
直接下载，只处理五条训练侧序列，并显式清除代理和镜像环境变量。下载按 1 MiB 分块保存，
网络中断后可以继续已有分块；每完成一条序列就写完成标记。

也可以在别处下载完整归档后放到
`/root/autodl-fs/DCVC/downloads/uvg-adaptation/`。文件必须使用下面的短文件名；准备脚本会先
核对真实字节数，确认完整后才清理同名断点分块，不会把半截文件当成可用数据。

| 短文件名 | 官方下载地址 | 预期字节数 |
|---|---|---:|
| `Beauty.7z` | `https://ultravideo.fi/video/Beauty_1920x1080_120fps_420_8bit_YUV_RAW.7z` | 925,430,047 |
| `Bosphorus.7z` | `https://ultravideo.fi/video/Bosphorus_1920x1080_120fps_420_8bit_YUV_RAW.7z` | 680,772,328 |
| `HoneyBee.7z` | `https://ultravideo.fi/video/HoneyBee_1920x1080_120fps_420_8bit_YUV_RAW.7z` | 906,770,507 |
| `Jockey.7z` | `https://ultravideo.fi/video/Jockey_1920x1080_120fps_420_8bit_YUV_RAW.7z` | 770,631,599 |
| `ShakeNDry.7z` | `https://ultravideo.fi/video/ShakeNDry_1920x1080_120fps_420_8bit_YUV_RAW.7z` | 460,046,003 |

压缩包和原始 YUV 只作为临时文件。每条序列的 12 个样本全部通过图片解码、尺寸和数量
检查后，立即删除对应 7z 和 YUV。长期保留内容位于：

```text
/root/autodl-fs/DCVC/data/UVG_adaptation/
```

日志、资源快照和完成状态位于：

```text
/root/autodl-fs/DCVC/runs/a800_uvg_adaptation_20260919/
```

长任务从 tmux 启动：

```bash
tmux new-session -d -s uvg_adaptation_download \
  'cd /root/autodl-tmp/adaptive-chunk-coding && \
   bash demo/prepare_uvg_adaptation_samples.sh'
```

脚本每分钟记录已完成序列、样本、断点分块、耗时和三块盘空间。完成后必须确认：

```bash
test -f /root/autodl-fs/DCVC/data/UVG_adaptation/dataset.complete
wc -l /root/autodl-fs/DCVC/data/UVG_adaptation/uvg_adaptation_samples.jsonl
du -sh /root/autodl-fs/DCVC/data/UVG_adaptation
df -h /root /root/autodl-tmp /root/autodl-fs
```

`wc` 应为 60。`summary.json` 还会记录 manifest SHA-256、各序列样本数、真实普通文件字节
和留出边界。

## 训练时怎样使用

数据恢复完成后，仍在一张 A800 上依次执行：

1. 对 60 个 UVG 窗口生成冻结三路质量 teacher 标签；
2. 对相同窗口实测 SeedVR2 ROI 成本；
3. 把这些标签与已有 500 个 REDS 标签合并，不复制大文件；
4. 先保留 v5 的 REDS v1 锚点、置信度和预算，只重训上下文／Base-probe 残差专家；
5. 先做便宜的路由变化检查，再在 ReadySetGo、YachtRide 和 REDS 分层上运行真实
   spatial-QP、fresh decode、基线与消融；
6. 将域适配、空间一致性和羽化宽度分别消融，再选择整体最合适的组合。

第 4 步有意先只改变训练数据和残差专家，避免同时重训锚点、改求解器和改融合宽度后
无法解释收益。若这种小改动不足，再把“完整重训 v1 锚点”作为单独消融，而不是静默混入
同一个版本。

对应的可断点流水线为 `demo/run_stage_c_a800_uvg_adaptation.sh`。数据准备完成后在 tmux
中启动：

```bash
tmux new-session -d -s uvg_adaptation_train \
  'cd /root/autodl-tmp/adaptive-chunk-coding && \
   bash demo/run_stage_c_a800_uvg_adaptation.sh'
```

质量 teacher、ROI teacher、合并 manifest、残差专家 checkpoint 和 37 条 route-only
比较分别落在 `/root/autodl-fs/DCVC/runs/a800_uvg_adaptation_20260919/` 下。route-only 只
回答动作是否变化，不能当作画质结论；只有它通过完整性检查后，才启动真实 spatial-QP、
SeedVR2、fresh decode 和消融评估。

## 不变的科研边界

- DCVC-UF、SeedVR2 3B BF16 和 spatial-QP codec 继续冻结；
- 不进入多卡训练；
- teacher 标签和训练样本可以用于方法改进，但不能再包装成独立测试证据；
- 正式结论只认真实落盘字节、fresh decode、完整耗时和固定可视化；
- 下载归档和原始 YUV 完成转换后不保留，大文件不进入 Git。
