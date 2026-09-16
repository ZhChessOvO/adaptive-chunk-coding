# 云端 Codex 启动提示词

下面内容可直接粘贴到已经完成 Git clone 的服务器 Codex 会话中：

```text
请接手本仓库的 A800 80GB 云端单卡试跑。Git 仓库已经由用户克隆完成。

开始前请完整阅读根目录的 AGENTS.md、README.md、
docs/CLOUD_STORAGE_AND_UPLOAD.md 和 docs/CLOUD_A800_PILOT.md，并以它们作为
当前主线、数据边界、环境恢复、磁盘布局和实验范围的权威说明。若能访问 Notion，
再核对文档中链接的项目总览、02、07、09、11、12；无法访问时不要因此阻塞。

服务器的系统盘是 /root（30GB），数据盘是 /root/autodl-tmp（50GB），较慢的
文件存储是 /root/autodl-fs（200GB）。用户已经把以下文件直接上传到
/root/autodl-fs/，没有整理子目录：

- cvpr2026_image.pth.tar
- cvpr2026_video_hts.pth.tar
- seedvr2_ema_3b_bf16.safetensors
- train_sharp.zip
- val_sharp.zip

先核对 GPU、MIG 状态、三块盘、当前仓库位置和这五个文件。不要重复下载它们。
仓库、conda 环境、第三方源码和编译产物放在 /root/autodl-tmp；数据、模型和
所有正式实验输出放在 /root/autodl-fs。不要在 /root 安装大环境或保存大文件。

train_sharp.zip 可以完整解压到文件存储；val_sharp.zip 本轮只解压 000..005。
其他数据边界按 AGENTS.md 和 CLOUD_A800_PILOT.md 执行。两个 ZIP 解压后先保留，
不要自行删除。

其余所需内容直接从官方来源下载：SeedVR2 和 CUTLASS 源码放数据盘上的仓库中；
SeedVR2 的 ema_vae.pth、pos_emb.pt、neg_emb.pt 以及 BasicVSR++ 权重放文件存储。
当前试跑继续使用已经上传的 BF16 DiT，不下载官方 FP32 DiT。下载前检查并移除
conda、pip、Hugging Face 和 Git 的第三方镜像覆盖。

按文档在 /root/autodl-tmp 建立带显式路径的 conda 环境，在 A800 上重新编译两个
DCVC-UF 扩展，并用符号链接把文件存储中的权重、数据和 output 接入仓库。慢盘可以
直接承载数据和正式输出；只有在随机 I/O 明显阻塞时，才允许在数据盘使用不超过
8GB 的当前样本临时缓存，写回后立即清理。

环境验证后，按 CLOUD_A800_PILOT.md 持续完成最小 smoke test、8–16 个样本的
吞吐校准、500–1000 个反事实 teacher labels、轻量预算条件控制器训练和开发集评估。
不要只给计划，请在已授权范围内持续推进。冻结 DCVC-UF 和 SeedVR2，不启动多卡
生产、SeedVR2 微调、spatial-QP codec 微调或其他大规模训练。

只有在缺少必要数据或凭据、预计超过 12 小时、任一磁盘将达到 80%、可能触碰数据
边界，或下一步确实需要大规模训练时才停下来告诉我。长任务必须可断点续跑，记录
进度、显存、耗时和三个挂载点的空间。

所有正式 manifest、码流、日志、controller checkpoint、指标和
GT／Base／SeedVR2／三路拼接／action map 可视化保存到
/root/autodl-fs/DCVC/runs。完成后检查真实落盘字节、fresh decode、基线与
Generate／Enhance 分离消融；大文件不进 Git。提交并推送代码和文档，给出简短结论、
关键结果、资源消耗、下一阶段估计及继续／停止建议；若已连接 Notion，再同步项目页面。
```
