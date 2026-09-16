# 云端 Codex 启动提示词

下面内容可直接粘贴到已经完成 Git clone 的服务器 Codex 会话中：

```text
请接手本仓库的 A800 80GB 云端单卡试跑。Git 仓库已经由用户克隆完成。

开始前请完整阅读根目录的 AGENTS.md、README.md、
docs/CLOUD_STORAGE_AND_UPLOAD.md 和 docs/CLOUD_A800_PILOT.md，并以它们作为当前
主线、数据边界、环境恢复、磁盘布局和实验范围的权威说明。若能访问 Notion，再核对
文档链接的项目页面；无法访问时不要因此阻塞。

服务器的系统盘是 /root（30GB），数据盘是 /root/autodl-tmp（50GB），较慢的
文件存储是 /root/autodl-fs（200GB）。以下五个文件已经直接上传到文件存储根目录：

- cvpr2026_image.pth.tar
- cvpr2026_video_hts.pth.tar
- seedvr2_ema_3b_bf16.safetensors
- train_sharp.zip
- val_sharp.zip

先核对 GPU、三块盘、仓库和上传文件，然后严格按云端存储文档恢复目录。仓库、环境、
源码和编译放 /root/autodl-tmp；数据、模型和正式输出放 /root/autodl-fs。不要重复
下载已有文件。train_sharp.zip 可完整解压，val_sharp.zip 本轮只解压 000..005；
两个 ZIP 先保留。其他缺失依赖从官方来源直接下载，下载前先检查并移除镜像覆盖。
继续使用已上传的 SeedVR2 BF16 DiT，不另下 FP32 DiT。

随后按 README 完成环境安装、扩展编译和 smoke test，再按 CLOUD_A800_PILOT.md
持续完成已授权的单卡试跑。不要只给计划；在授权范围内自主推进到可复现的结果。
冻结 DCVC-UF 和 SeedVR2，不自行进入多卡生产或大模型／codec 微调。

只有在缺少必要数据或凭据、预计超过 12 小时、任一磁盘将达到 80%、可能触碰数据
边界，或下一步确实需要大规模训练时才停下来告诉我。长任务必须可断点续跑，记录
进度、显存、耗时和三个挂载点的空间。

所有正式结果保存到 /root/autodl-fs/DCVC/runs。完成后按文档检查真实落盘字节、
fresh decode、基线、消融和固定可视化；大文件不进 Git。提交并推送代码和文档，
给出简短结论、关键结果、资源消耗和下一阶段建议；若已连接 Notion，再同步项目页面。
```
