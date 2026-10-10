<div align="center">

# RouterVC

**基于 DCVC-UF 的可伸缩区域增强与生成协作**

</div>

## 文档入口

当前主线是 **同一次高质量 UF 表示的粗基础层＋区域细化补包＋可选生成协作**。
新 B 是高质量表示的粗版本，不是重新编码的 QP8；小包优化和冻结 G 检查之后，适配新 B/E 上的独立双 Router。
方法、实验状态、结果与可视化只在 Notion 维护，不在 Git 另写研究报告：

- [项目首页](https://app.notion.com/p/3d58b22ebd8d815483aad4e1471ee933)
- [00 当前整体方法](https://app.notion.com/p/3e78b22ebd8d81828124c50e8e74c2ca)
- [1010 新版完整结果：RD、发送端消融、画面与资源](https://app.notion.com/p/3f58b22ebd8d8145a663d10ffabeb5c7)
- [1007 新 E：已授权计划](https://app.notion.com/p/3f28b22ebd8d8143b641c38667998b54)
- [1008 小包优化、连续区域与 G 接回](https://app.notion.com/p/3f38b22ebd8d8139b98aeb3f79198e02)
- [1008.1 新 B/E 双 Router：标签、训练与接收端对照](https://app.notion.com/p/3f38b22ebd8d81b9a240f004b5f22c81)
- [1007.1 新 E：实施、真实字节和画面](https://app.notion.com/p/3f28b22ebd8d81489980feaa14ceae3c)
- [04 旧 E 完整系统结果：RD、画面和显存](https://app.notion.com/p/3f18b22ebd8d81fabf9fc4ccdb4cf5ae)
- [03 实验导航：所有版本、曲线与固定画面](https://app.notion.com/p/3e78b22ebd8d819cbad8cd249c7566b0)

下文 spatial-QP 研究叙述为历史版本，环境安装说明仍可参考。不要按旧段落自动重启历史队列；
其运行入口、源码固定要求和复现边界见 `AGENTS.md` 及对应 Notion 实验页。

## 已完成入口：新 B/E 发送端适配与整链路评价

2026-10-10已完成结果复核与Notion整理，当前等待讨论，不启动下一轮训练。
以下训练命令保留用于复现，不表示任务仍在运行；优先查看`evaluation/complete.json`。
阶段完成文件中历史`evaluation_pending`字段不代表后续评价未完成。

接收端适配与固定E对照已完成；具体曲线和画面见上方1008.1。固定经复核选定的core R_g，
为不同结构、能看原片的R_s重新测量最终补包收益；UF、width3和G保持不变。
研究说明与实际进度只维护在
[1009.1 新发送端](https://app.notion.com/p/3f48b22ebd8d81ec8328e21aa31f4192)。

```bash
# tmux中：tests先做CPU合同，smoke含真实新标签、精确断点和独立接收。
bash tools/run_latent_sender.sh tests
bash tools/run_latent_sender.sh smoke
# 匹配烟测通过后，120窗新标签 → 两组独立R_s各120轮；相同命令续跑。
bash tools/run_latent_sender.sh train
bash tools/run_latent_sender.sh verify
# 另一个tmux：等待发送端完成再做真实字节整链路对照；等待时不占GPU锁。
bash tools/run_latent_system_review.sh --wait
# 独立评价链路烟测，以及完成后的只读核验。
bash tools/run_latent_system_review.sh --smoke --wait
bash tools/run_latent_system_review.sh --verify-only
```

正式目录为`/root/autodl-fs/DCVC/runs/routervc_latent_sender_20261009`；本轮紧凑输入缓存
也放该目录的`input_cache/`，不给接近容量保护线的数据盘继续加压。接收头和E包均按真实
文件大小收费，不发送动作mask。每次选择后都实际解码混合latent，不能使用RGB区域粘贴。
实际运行阶段以`teacher.progress.json`、`router/progress.json`及完成收据为准；标签制备
不等于参数优化已开始。不要修改已绑定源码后强行混入原目录。
整链路评价保存在同目录的`evaluation/`：原片可见／去原片R_s、固定补包顺序、关G，
以及原生UF的I32／P0至P48锚点。补包预算按候选E总字节的比例截取，不是区域数量；
13个历史诊断窗口不当作完整测试集。评价逐点保存，可用相同命令续跑。
仅修图表排版时，可在tmux用评价入口的`--replot-to /absolute/new/output`从已保存
像素／指标重画到新目录；不重跑推理、不覆盖已完成结果。烟测重画另加`--smoke`。
此次只读报告在`report_20261010/`，包含真实前缀、同流关G、重复解码收据复核及训练图。
需要复核报告时，在tmux中运行以下CPU命令；不启动codec/G推理或训练：

```bash
/root/autodl-tmp/DCVC/envs/dcvcuf/bin/python -m tools.storage_guard \
  --log /root/autodl-tmp/DCVC/tmp/latent_sender_report_guard.jsonl --min-inodes 5000 -- \
  /root/autodl-tmp/DCVC/envs/dcvcuf/bin/python -m tools.latent_sender_report
```

## 已完成入口：新 B/E 接收端适配

保持 width3、UF 和现有 G 权重不变，重新测量实际 latent 补包画面的 G 收益。
先适配独立 core R_g，再经对照选择接收端，为不同结构的发送端 R_s 准备最终收益标签。
不要用旧 E 标签、RGB 区域粘贴或旧 Router 的预测冒充新监督。

```bash
# 必须在 tmux 中。all = 真实烟测 → 120 窗新标签 → 120 轮 core R_g。
bash tools/run_latent_routers.sh tests
bash tools/run_latent_routers.sh all
# 同命令恢复；独立阶段也可 smoke / train，完成后 verify。
bash tools/run_latent_routers.sh verify
# 深层只读核验：真实包、每条实测标签、压缩张量、优化器终态与最佳模型。
python -m tools.verify_latent_routers /root/autodl-fs/DCVC/runs/routervc_latent_routers_20261008/formal
# 另一个 tmux 等待正式接收端完成，再自动对照，不启动发送端训练。
bash tools/run_latent_receiver_review.sh --wait
```

正式产物 `/root/autodl-fs/DCVC/runs/routervc_latent_routers_20261008`；无损压缩的
Router 输入缓存 `/root/autodl-tmp/DCVC/cache/routervc_latent_routers_20261008`。
`RVLRG001` 是新的接收策略封装：只含必要模型身份与共享控制，无动作图；G 按区域编号
固定噪声，每个生成区都只读同一个未生成 Y。独立接收进程不读原片、未发送包或发送端模型。
按区域保存 teacher 进度，按训练更新原子保存模型／优化器／随机状态；三盘及 inode 保护
触发后先检查空间，再原命令恢复。完成收据和状态优先于 tmux 窗口是否仍存在。

## 已完成入口：冻结 UF 的 latent 粗细分层

下面这些已完成诊断保持width3、UF和既有G权重不变，不训练新模型。`RVLPACK2`把每个区域内部的
多个细化熵流合为一条，概率表不变，包编号含chunk和区域。`RVLGEN01`使用预共享的固定
中心四区域G策略进行成对诊断，不是新条件下已训练的Router；必要profile头仍计费。
所有长任务在tmux内执行，同命令恢复；完成后优先`verify`，不重复已完成推理：

```bash
bash tools/run_latent_packets.sh test
bash tools/run_latent_packets.sh run --mode single --limit 4
bash tools/run_latent_packets.sh run --mode continuous --limit 6
# 连续检查完成后依次运行G烟测、G比较和同17帧UF锚点；没有训练步骤
bash tools/run_latent_followup.sh wait
# 完成后只读核验
bash tools/run_latent_packets.sh verify --mode single --limit 4
bash tools/run_latent_packets.sh verify --mode continuous --limit 6
bash tools/run_latent_generate.sh verify
bash tools/run_latent_followup.sh native verify
# tmux中的CPU图表，复用接收像素，不重新推理
CUDA_VISIBLE_DEVICES='' python -m tools.plot_latent_followup --wait
CUDA_VISIBLE_DEVICES='' python -m tools.plot_latent_resources
```

正式目录`/root/autodl-fs/DCVC/runs/routervc_latent_20261008`。新格式与下方已完成的
RVL1/RVLC1/RVLR1并存；不要修改已绑定源码后继续写入旧实验目录。

核心原型在 `routervc/latent/`，运行和检查在 `tools/`；不覆盖历史代码、模型或已安装的
CUDA 扩展。`native_bridge.cpp` 是绑定本机二进制／头文件的私有研究接口，尚非通用部署API。
实际模型身份、数值路径、样本和配置由各阶段 `protocol.json` 及码流头绑定。
G 与双 Router 最终仍保留；下面是10月7日隔离检查的复现入口，不启动新训练。

```bash
# 项目Python环境中的CPU回归
python -m unittest tools.test_latent_split tools.test_latent_stream \
  tools.test_latent_chain tools.test_latent_regional tools.test_latent_reports

# 长任务必须在tmux中；相同命令支持断点恢复。
# 阶段A/B/C已完成时优先用verify，不重复推理。
bash tools/run_latent_probe.sh verify
bash tools/run_latent_stream.sh verify
bash tools/run_latent_chain.sh verify
bash tools/run_latent_baseline.sh verify
# 区域阶段也已完成；新诊断目录首次／恢复才用run
bash tools/run_latent_regional.sh verify

# CPU图表（同样在tmux）；完成后重入只核验，不重画。
CUDA_VISIBLE_DEVICES='' python -m tools.plot_latent_probe
CUDA_VISIBLE_DEVICES='' python -m tools.plot_latent_diagnostics --wait
CUDA_VISIBLE_DEVICES='' python -m tools.plot_latent_regional --wait
```

正式目录 `/root/autodl-fs/DCVC/runs/routervc_latent_20261007`。新运行入口使用同一GPU互斥锁、
三盘／inode保护和逐样本原子保存。代码／模型／输入改变时拒绝混入既有目录；新配置需新目录。
RVL1 是单P8两层流，RVLC1 是连续B参考检查，RVLR1 是单P8区域包原型；它们不冒充互通格式。
区域计数不等于字节比例，完整E端点一致不等于任意局部像素独立或RD优于UF。研究解释、
曲线和所有失败／成功观察只在上方Notion页面维护。

## 已完成旧 E 入口：非对称双 Router

整体计划见[03.22 双 Router](https://app.notion.com/p/3ef8b22ebd8d81e39a48efa151655a0f)，
执行状态见[03.22.1 接收端](https://app.notion.com/p/3f08b22ebd8d8141a824faf4872bc9e2)。
发送端 R_s 读取原片并分配补包；接收端 R_g 只读实际重建并选择生成，两端不是同一个模型。
两版 R_g 的120轮训练、固定E评价和图表均已完成；不要重复跑已完成的推理。
用户已选定 core 最佳模型；接收端阶段的结果和完整图集见
[接收端结果](https://app.notion.com/p/3f18b22ebd8d81639ef5d4fd3989e8c9)。
发送端两版120轮及117点整链路评价也已完成；整体图集在上方04旧版结果页，仍不是全测试集评价。
该阶段已结束，当前转入1007新E研究；不要据以下历史入口自动启动旧训练或交替适配。

```bash
# 历史接收端入口：长任务在 tmux 中执行，已完成任务优先只读核验。
bash demo/run_routervc_receiver.sh test
bash demo/run_routervc_receiver.sh smoke
bash tools/run_routervc_receiver_guarded.sh train
# 全部训练结束后只读核验
bash tools/run_routervc_receiver_guarded.sh verify
```

正式目录 `/root/autodl-fs/DCVC/runs/routervc_receiver_20261005`；
`formal/router/progress.json` 记录真实更新，`formal/router/resume.pt` 为权威恢复点。
core/halo 是两个接收端输入候选，不是发送／接收两端；UF/E/G 固定。
数据盘紧凑缓存最多驻留8窗于内存，单卡互斥、30秒心跳、逐区域标签与逐窗口检查点。
新接收格式 RVRC/v1 只需 R_g，不需要 R_s、原片或未收候选；不发送动作mask。
真实RD与固定画面已完成，研究结果仅在Notion记录。

训练后固定E评价与CPU出图已完成，以下入口支持核验完成状态（不会重复推理）：

```bash
bash tools/run_routervc_receiver_guarded.sh eval --wait
bash tools/run_routervc_receiver_guarded.sh report --wait
```

保护入口在原队列外监控三盘容量与 inode（文件条目）余量，不修改锁定的研究源码或
恢复点。默认保留至少5,000个文件条目、字节使用率低于80%；触发后同命令续跑。
详细存储检查、已授权旧输出清理及当前可恢复范围见
[`CLOUD_STORAGE_AND_UPLOAD.md`](docs/CLOUD_STORAGE_AND_UPLOAD.md)。

独立发送端实现与显式选定R_g后的启动说明见
[03.22.2 发送端](https://app.notion.com/p/3f08b22ebd8d8164a6b2cd6061dd3fcb)。
CPU入口 `bash demo/run_routervc_sender.sh test`；长任务使用带三盘／inode保护的
`bash tools/run_routervc_sender_guarded.sh smoke|train|verify`，实际选一个子命令。
10月6日真实烟测及正式队列已完成；优先使用 `verify`，不要重复训练。
首次启动需要显式 `--receiver`、`--receiver-sha256`，正式阶段另加
`--receiver-complete`；准确绑定见 `AGENTS.md`。同阶段恢复可省略已记录绑定参数。
正式训练先准备全部固定标签，再更新source／zero_source两组权重；不能把标签准备
或结构烟测写成已经开始正式优化，也不能把旧共享模型当作完整双Router结果。

训练后评价和CPU报告已完成；下列入口支持读取完成凭据而不重复推理：

```bash
bash demo/run_routervc_sender_evaluate.sh test
bash demo/run_routervc_sender_evaluate.sh run --wait
bash demo/run_routervc_sender_evaluate.sh report --wait
# 完成后的只读核验，不重跑推理
bash demo/run_routervc_sender_evaluate.sh verify
```

固定同一R_g/G策略，比较旧E选择、source和zero_source；候选真实熵编码包复用，
不将缓存选包耗时称为完整编码耗时。曲线、数字与方法讨论继续只放Notion。

当前方法与原生UF的汇总图、全部13窗固定帧／中心细节及资源图，可从已完成测量重建：

```bash
# CPU-only，仍需在tmux中运行；不更新模型，不重新执行生成。
CUDA_VISIBLE_DEVICES='' python -m tools.plot_routervc_overview
python -m unittest tools.test_plot_routervc_overview
```

输出到发送端结果目录的 `overview_20261007`；包含实际字节、原图来源与产物校验。
UF画面仅按最近实际码率选择，图上标明码率差；历史基线计时与当前测量分开说明。

补包标签的CPU诊断入口（在tmux中运行）：

```bash
python -m tools.plot_routervc_sender_targets \
  --stage /root/autodl-fs/DCVC/runs/routervc_sender_20261005/formal \
  --output /root/autodl-fs/DCVC/runs/routervc_sender_20261005/target_diagnostics --wait
```

它等待全部标签落盘，只读取已有分数和码流；不生成新标签、不更新模型、不选择权重。
TRAIN／验证与REDS／UVG分别统计。最终收益与直接重建收益的差值还包括G选区和噪声
分配的变化，不能单独当作“生成协同增益”。完成后重入只核验，不重画或改写结果。

## 已完成入口：混合画面 Router 训练

方案与进度只记录于[03.21 混合画面 Router](https://app.notion.com/p/3ef8b22ebd8d8136941ad2d5b7c1cf43)。
UF/E/G固定；三组等容量Router分别使用孤立区域、真实混合重建、混合重建＋生成邻域。
长任务均在tmux中运行，同命令恢复；先烟测再正式训练：

```bash
bash demo/run_routervc_mixed.sh test
bash demo/run_routervc_mixed.sh smoke
bash demo/run_routervc_mixed.sh train
# 全部训练完成后，只读核验已有模型和标签
bash demo/run_routervc_mixed.sh verify
```

正式目录`/root/autodl-fs/DCVC/runs/routervc_mixed_router_20261004`；`formal/router/progress.json`
记录真实更新步数，`formal/router/resume.pt`包含全部三组模型、优化器和轮内位置。
各组保留`best.pt`与`last.pt`；第一轮逐段测标签后训练，后续复用固定标签。
三组120轮已完成，作为旧共享对照和新R_g初始化／标签来源，不代表双Router已完成。
新模型使用独立输入格式，不能直接传入旧接收器；真实码流评价另行接续。
紧凑缓存放数据盘，不下载新数据、不发送动作mask、不训练内容保护。

## 最新完成入口：轻补包 Router 适配

结果、RD图和画面入口见 [03.20.1 结果](https://app.notion.com/p/3ef8b22ebd8d815382d5e9a655d9cf21)，
方案与记录见 [03.20 轻补包 Router](https://app.notion.com/p/3ef8b22ebd8d810bb1d7ec68e38de801)。
标签、两版训练、156点评价和图表均已完成并核验；已有完成标志时只读重放，不重新推理。
只训练原结构的Router，不改UF/E/G；q2补包保持真实熵编码，不增加动作mask。
以下任务在tmux中执行；相同命令恢复，检查完成项后复用：

```bash
bash demo/run_routervc_light_router.sh test
bash demo/run_routervc_light_router.sh smoke
bash demo/run_routervc_light_router.sh run
# 独立tmux窗口；不占GPU地等两臂正式训练结束，再运行同q2包库的旧/新比较：
bash demo/run_routervc_light_evaluate.sh run
# 项目Python环境内，独立tmux中的CPU图表任务；等待评价，不重新推理：
CUDA_VISIBLE_DEVICES='' python -m demo.routervc_light_report --wait
# 标签和训练完成后的CPU诊断；读取已有质量和训练曲线，不推理、不重选checkpoint：
CUDA_VISIBLE_DEVICES='' python -m demo.routervc_light_label_report
```

正式根目录为`/root/autodl-fs/DCVC/runs/routervc_light_router_20261004`；
`teacher_smoke`核对真实解码及恢复，`teacher`保留实测标签，`router`保存两版120轮模型，
`evaluation`保存真实码流与fresh接收结果，`label_diagnostics`保存标签／训练诊断。
小型视觉缓存放数据盘。
评价中的E比例指q2候选库真实包字节比例，与03.19沿用q1位置的实验不同。
旧代码/模型/结果不覆盖；这不是内容保护训练，也不自动选择部署模型。

## 已完成阶段入口：固定权重的传输与执行效率

方案、结果与图仅记录于 [03.19 传输与执行效率](https://app.notion.com/p/3ef8b22ebd8d81fbb362f658f696a241)。
不修改历史码流或模型。以下长任务在tmux中运行，共享单卡互斥锁；同命令断点恢复，
已有`complete.json`时只读核验，不重新推理：

```bash
bash demo/run_routervc_entropy_audit.sh run
bash demo/run_routervc_qstep_probe.sh run
bash demo/run_routervc_schedule_probe.sh
# 项目Python环境内的CPU几何分析：
CUDA_VISIBLE_DEVICES='' python -m demo.routervc_generation_schedule
# CPU汇总、图像；完成后同命令只读复验：
CUDA_VISIBLE_DEVICES='' python -m demo.routervc_efficiency_report
```

输出位于`/root/autodl-fs/DCVC/runs/routervc_efficiency_20261004`的`entropy_audit`、
`q2_probe`、`G_pair_probe`和`G_geometry`。三个shell入口都支持`test`运行CPU合同测试。
q2实验沿用旧q1的补包位置；E25/E50不是新包库的字节比例。G合并使用新的共享执行指纹，
不多发mask，原小块低显存模式保留。显存比较须取全部G调用中的峰值，不能只读旧解码器
最后一次调用的顶层峰值字段；完整worker与G阶段峰值分开。

## 已完成阶段入口：混合视野与内容监督准备

方案与进度只维护在 [03.18 实施与结果](https://app.notion.com/p/3ee8b22ebd8d813c9dbedd66dd5eb4a6)。
使用 REDS 完整画面与现有 UVG 裁剪，不等待原始 UVG；清单区分两类视野。
旧码流已经不发逐区域 G 图，不能再次虚减字节。新模块不修改历史实验源码。

```bash
# 项目 Python 环境内运行；prepare、probe 与审计均在 tmux 中运行。
python -m demo.routervc_fullview_data inventory --splits val \
  --output /root/autodl-fs/DCVC/runs/routervc_revision_20261003/fullview_manifest.json
python -m demo.routervc_fullview_data prepare \
  --manifest /root/autodl-fs/DCVC/runs/routervc_revision_20261003/fullview_manifest.json \
  --output /root/autodl-fs/DCVC/runs/routervc_revision_20261003/data_smoke \
  --sample-ids reds-val-000-f000-n17-fullview --max-hours 2
bash demo/run_routervc_fullview_probe.sh \
  --input /root/autodl-fs/DCVC/runs/routervc_revision_20261003/data_smoke/samples/reds-val-000-f000-n17-fullview/frames \
  --output /root/autodl-fs/DCVC/runs/routervc_revision_20261003/fullview_probe_verified --verify-only
CUDA_VISIBLE_DEVICES='' python -m demo.routervc_byte_audit --verify-only
```

上面的 probe 命令只读复验已完成结果，不重新推理。首次运行改用新的输出目录，去掉
`--verify-only`，可设置 `--max-hours 6`；相同配置和源码下用相同命令续跑。
旧 `fullview_probe` / `fullview_probe_recovered` 保存修复前后的运行记录，不覆盖其协议。
准备输出绑定样本选择，
换选择使用新输出目录。这里的 probe 仅检查一个 1024×576 完整视野窗口，不是原生分辨率
或测试集结论。`routervc_content_objective.py` 是离线标签接口，不代表语义标注或训练已完成。

混合训练输入及新版视觉本体入口（长任务均在tmux）：

```bash
python -m demo.routervc_mixedview_data manifest \
  --output /root/autodl-fs/DCVC/runs/routervc_revision_20261003/mixedview_manifest.json
python -m demo.routervc_mixedview_data prepare \
  --manifest /root/autodl-fs/DCVC/runs/routervc_revision_20261003/mixedview_manifest.json \
  --output /root/autodl-fs/DCVC/runs/routervc_revision_20261003/mixedview_data
bash demo/run_routervc_mixedview_teacher.sh smoke \
  --data /root/autodl-fs/DCVC/runs/routervc_revision_20261003/mixedview_data \
  --output /root/autodl-fs/DCVC/runs/routervc_revision_20261003/mixedview_teacher_smoke
# 上述真实数据烟测完成后，接续正式teacher及两种等容量视觉本体训练：
bash demo/run_routervc_revision_queue.sh
```

teacher只生成B/E/G/EG实测标签，不更新模型；30个相同UVG裁剪的旧标签核验后复用。
队列先训练感知收益对照，尚不含真实文字/面部监督，不自动接入或晋升部署模型。
视觉输入缓存放数据盘，正式模型/结果放文件存储；同命令可恢复。
CPU测试：`bash demo/run_routervc_visual_train.sh test`。

离线文字/面部工具只生成诊断与标签，不向接收端发送信息：

```bash
# tmux、项目Python环境；权重已在数据盘时不要重复下载。
CUDA_VISIBLE_DEVICES='' python -m demo.routervc_content_labels coverage \
  --data /root/autodl-fs/DCVC/runs/routervc_revision_20261003/mixedview_data \
  --output /root/autodl-fs/DCVC/runs/routervc_revision_20261003/content_coverage_pilot120
```

同命令只读复验已完成扫描。自动OCR不是字符真值；未核验的源参考/候选不生成文字错误
训练目标。面部仅测五点几何，不测身份。工具说明、覆盖结果与误检图片见Notion 03.18.2。

新版视觉Router的独立码流入口如下。必须显式指定正式模型，不默认部署烟测权重；
CPU接口测试不能代替正式模型的真实GPU编解码验收。旧入口/码流仍保持原样。

```bash
# 两条命令均在tmux、项目Python环境内运行，路径替换为实际输入/正式模型。
python -m demo.routervc_visual_encode --input /path/to/frames \
  --router /path/to/formal/model.pt --e-ratio .25 --max-g 4 \
  --output /root/autodl-fs/DCVC/runs/my_visual_router/encode
python -m demo.routervc_visual_decode \
  --stream /root/autodl-fs/DCVC/runs/my_visual_router/encode/stream.rtvc \
  --router /path/to/formal/model.pt \
  --output /root/autodl-fs/DCVC/runs/my_visual_router/decode
```

G-off使用`--disable-generation`，无需提供Router/G权重。新旧策略由共享头中的哈希区分；
头部与E包仍全部计费，不新增E/G/保护mask。保留单本体，未训练的语义输出不参与决策。

正式双臂训练后的接续评价（独立tmux窗口）：

```bash
bash demo/run_routervc_visual_evaluate.sh \
  --output /root/autodl-fs/DCVC/runs/routervc_revision_20261003/visual_evaluation_recovered
# 全部完成后校验已有结果，不重新推理或计算指标：
bash demo/run_routervc_visual_evaluate.sh \
  --output /root/autodl-fs/DCVC/runs/routervc_revision_20261003/visual_evaluation_recovered --verify-only
```

默认等待`routervc_revision_20261003/visual_router`的两组正式完成标志，拒绝smoke模型。
等待最多48小时且不占GPU锁；之后单次评价最多12小时，含GPU排队和指标计算。
相同命令恢复已核验的点；源码、模型或输入变化须使用新输出目录。
本机修复后结果在`routervc_revision_20261003/visual_evaluation_recovered`；原`visual_evaluation`
保留启动失败记录，不覆盖其源码绑定。具体样本、比较设置与状态见
[03.18.3 扩展比较](https://app.notion.com/p/3ee8b22ebd8d811f8a67f1a7b0a43c8f)。
CPU测试：`bash demo/run_routervc_visual_evaluate.sh test`（tmux内）。

完成后的独立分析与补充对照（不覆盖原169点；长任务均在tmux中）：

```bash
# 只读复核、分组RD；可选从保存的增强RGB计分，不重新生成或改原始指标
bash demo/run_routervc_visual_report.sh --score-enhanced
# 补足原生UF质量索引40/48/56的真实码率覆盖
bash demo/run_routervc_uf_extension.sh
# 两版Router相同G8预算下的E=0真实码流消融，自动等待单GPU互斥锁
bash demo/run_routervc_zero_e_probe.sh
# 在已测四状态像素上检查原图含文字/脸的区域，不训练、不发送标签
CUDA_VISIBLE_DEVICES='' python -m demo.routervc_content_review
CUDA_VISIBLE_DEVICES='' python -m demo.routervc_content_review_figures
# 上面评价及E=0完成后，只读合并曲线、同流关G、运行时间与包字节分项
CUDA_VISIBLE_DEVICES='' python -m demo.routervc_visual_ablation_report
```

输出分别在当前revision根目录的`visual_evaluation_analysis`、`visual_uf_rate_extension`、
`visual_zero_E`、`content_candidate_review`、`content_review_visuals`及`visual_ablation_report`。
相同代码/参数恢复逐点结果；先检查完成标志，不重跑已完成推理。G-off评分保留同一码流
字节，不冒充另外压缩的低开销E-only流。文字自动识别仍是诊断，不自动启用内容保护头。
结果、图像和下一步选择只记录于Notion 03.18。

## 已完成运行入口：RouterVC 完整闭环

研究记录与图：[03.17 RouterVC完整闭环](https://app.notion.com/p/3ee8b22ebd8d8112aa83dfe749002a02)。
旧阶段实现与四裁剪开发评价已完成；[整体验证、标准UF曲线与强整帧G对照](https://app.notion.com/p/3ee8b22ebd8d81578d66c6e03b5070c9)。
其中120个RD点不是120段独立测试视频；更广泛评价在03.18推进。
四状态数据及两组Router训练已完成。新入口接入预算选包、接收端共享G策略、生成边界整理与真实码流评价。
单一本体，不恢复两个专家；Base／只E／只G／E→G均保留。必要策略头计费，不再发送逐区域G图。

| 操作 | 命令 |
| --- | --- |
| CPU测试 | `bash demo/run_routervc.sh test` |
| tmux中真实解码、前缀、回退与重复检查 | `bash demo/run_routervc.sh smoke --max-hours 3` |
| tmux中多预算/本体/边界对照 | `bash demo/run_routervc.sh run --max-hours 12` |
| 完成后只读验证 | `bash demo/run_routervc.sh verify` |
| UF／全G／固定路由消融 CPU 测试 | `bash demo/run_routervc_baselines.sh test` |
| tmux中自动补齐基线（先烟测，再正式） | `bash demo/run_routervc_baselines.sh queue` |
| 完成后基线只读验证 | `bash demo/run_routervc_baselines.sh verify` |
| tmux中排队CPU收尾：等正式／补充／恢复检查，审计、分析及短预览 | `bash demo/run_routervc_finalize.sh --max-hours 24` |
| tmux中标准UF原生码率对比（等待收尾完成，仅CPU） | `bash demo/run_routervc_native_uf_report.sh --wait-workflow` |

正式结果：`/root/autodl-fs/DCVC/runs/routervc_20261003`；烟测加`_smoke`。
相同命令恢复，已完成码流和指标校验后复用。当前按E包字节预算与G调用上限分配；
固定排序的E包可真正追加，G会随实际重建重算，不保证LPIPS逐包单调。
边界整理暂不合并生成调用，实际耗时单独测量。当前评价复用候选缓存，不能把它的
发送端时间写成从原视频开始的完整编码时间。详细能力范围和最新运行状态看Notion。

### 从自己的帧序列编码与独立接收

在项目环境、tmux中使用独立公共入口；它负责GPU互斥、资源记录与原子续跑。
输入是按文件名排序的PNG目录，或含`source`键的`uint8 [T,H,W,3]` RGB NPZ。
不隐式裁剪、缩放或补齐帧尾；要求`T>=17`且`T=1+8n`，宽高为8的倍数且在64～8192内。
完整G路径已验证512×512、384×256及17/33帧；它不能直接接受任意尺寸，例如原尺寸720p/1080p。
`--max-g 0`仅取消G的额外几何约束，高分辨率E路径尚未完成真实码流／显存验证。
4×4是当前Router拓扑，不是码流的普适限制。

```bash
# 示例路径须替换为实际输入和新的输出目录；两条命令均在tmux内执行。
bash demo/run_routervc_codec.sh encode \
  --input /path/to/frames --count 33 \
  --output /root/autodl-fs/DCVC/runs/my_routervc/encode \
  --e-ratio 0.5 --max-g 4 --mode prefix
bash demo/run_routervc_codec.sh decode \
  --stream /root/autodl-fs/DCVC/runs/my_routervc/encode/stream.rtvc \
  --output /root/autodl-fs/DCVC/runs/my_routervc/decode
```

`--e-ratio`是全部候选E包字节的比例，不是总码率；也可用`--e-budget`指定E包字节上限。
发送端先实际编码、解码全部16个区域的候选，再按预算选择；即使E预算为0也会准备全部候选。
`encode.json`记录完整准备与选包耗时。准备阶段在UF基础流完成、完整E候选库完成后分别保存；
若在E候选准备中途重启，需要重做该阶段，而不是从单个E包继续。
接收端没有源图参数，`fresh/reconstruction.npz`保存底图、增强图与最终输出。
`--disable-generation`可直接显示收到E后的画面，`--allow-incomplete-tail`只容忍末尾未收全的包，
不会把半个熵载荷当有效增强。相同命令续跑，配置变化使用新输出目录。
收到追加包后，须使用新的decode输出目录；当前会对完整前缀重新解码，
不是保持在线解码状态、只计算新收到的增强。
换E额度时可用`--prepared-dir`复用经过hash核验的候选缓存；保持同一输入、模型与G策略，
`prefix`模式的较高预算流才是原字节串的追加。不要用`independent`模式声称逐字节前缀。

独立CPU审计（完成后，tmux中）只验证保存证据，不重新推理或计算指标：

```bash
CUDA_VISIBLE_DEVICES='' /root/autodl-tmp/DCVC/envs/dcvcuf/bin/python \
  -m demo.routervc_audit --output /root/autodl-fs/DCVC/runs/routervc_20261003
```

基线补充输出在正式目录的`supplement/`；所有研究解释、结果与图片仍只维护在Notion。

### 查看已有结果的运动对照

补充评价完成后，在tmux中用CPU制作固定REDS／UVG并排MP4，不重新推理或计算指标：

```bash
CUDA_VISIBLE_DEVICES='' /root/autodl-tmp/DCVC/envs/dcvcuf/bin/python \
  -m demo.routervc_preview \
  --root /root/autodl-fs/DCVC/runs/routervc_20261003/supplement \
  --profile formal --repeats 1 \
  --output /root/autodl-fs/DCVC/runs/routervc_20261003/supplement/preview_single
```

输出含MP4、固定封面和校验manifest。`--repeats 1`减少上传大小；烟测可改用`_smoke`目录与
`--profile smoke`。展示帧率不代表原始帧率或解码速度，MP4转码字节和画面不参与RD／LPIPS评价。
既有预览保持不变；改变展示参数时使用新输出目录。

包含强整帧G对照的五面板预览（同样在tmux中，仅CPU、复用保存结果）：

```bash
CUDA_VISIBLE_DEVICES='' /root/autodl-tmp/DCVC/envs/dcvcuf/bin/python \
  -m demo.routervc_fullframe_preview
```

默认输出`supplement/preview_fullframe/`，展示原图、原生UF QP32、Router G4/G8和单ROI整帧G。
UF字幕用真实原生字节，其余用完整码流字节；不会覆盖已有预览或重新计算指标。

## 已完成运行入口：四状态 Router 数据准备

范围与进度：[03.16 四状态路由准备](https://app.notion.com/p/3ed8b22ebd8d81a8915ec00064643818)。
先用已完成的联合版 E/G 测量 Base／只E／只G／E→G，之后训练单一本体 Router；
这里的数据队列本身不更新模型。没有恢复旧两专家，也不重新运行已完成的训练。

| 操作 | 命令 |
| --- | --- |
| CPU 协议测试 | `bash demo/run_four_state.sh test` |
| tmux 中两样本、断点与 fresh 接收端烟测 | `bash demo/run_four_state.sh smoke --max-hours 3` |
| tmux 中120条数据、真实字节与画质标签 | `bash demo/run_four_state.sh run --max-hours 24` |
| tmux 中完成结果只读核验 | `bash demo/run_four_state.sh verify` |
| CPU Router 单元测试 | `bash demo/run_four_state_router.sh test` |
| tmux 中等待数据、训练并评价初版Router | `bash demo/run_four_state_router_queue.sh --max-hours 24` |

正式目录：`/root/autodl-fs/DCVC/runs/a800_four_state_20261002`，烟测加 `_smoke`。
按区域原子保存，相同命令续跑；进度和三盘/GPU心跳保存在正式目录。
单区域收益表不是已验证的整幅组合质量，正式分配须重新解码/生成验证。
Router队列另存`/root/autodl-fs/DCVC/runs/a800_four_state_router_20261002`，现已完成
两组等容量的240-epoch CPU训练（本体内上下文／局部对照），每5轮原子保存。
它只做分组开发集的收益预测与表格预算评价，不自动宣称真实路由RD或共享接收策略已完成。
输入只有底图、实际候选重建和E覆盖；LPIPS为主，PSNR/时序为辅助；仍只用一个本体。

## 已完成运行入口：E＋G 联合训练与真实码流评价

配方与实际状态：[03.15 E与G联合训练](https://app.notion.com/p/3ed8b22ebd8d8158be1dd92e3d59b99f)。
继承已完成的区域对齐版；单A800，UF固定，不训练Router。旧权重与接收入口保留。

| 操作 | 命令 |
| --- | --- |
| CPU 回归 | `bash demo/run_online_eg.sh test` |
| tmux 中的梯度、断点与真实解码检查 | `bash demo/run_online_eg.sh smoke` |
| tmux 中的两组各3000步训练 | `bash demo/run_online_eg.sh train --max-hours 24` |
| 评价 CPU 回归 | `bash demo/run_online_eg_evaluate.sh test` |
| tmux 中的真实字节评价及固定图 | `bash demo/run_online_eg_evaluate.sh run --max-hours 8` |
| 完成后的只读续跑检查 | `bash demo/run_online_eg_analysis.sh`（tmux，禁止重新推理/计算指标） |
| 续跑检查完成后的交叉图/生成曲线放大图 | `/root/autodl-tmp/DCVC/envs/dcvcuf/bin/python -m demo.online_eg_cross_report`（CPU，只读取已保存结果） |

正式目录：`/root/autodl-fs/DCVC/runs/a800_online_eg_20261002`，烟测另加 `_smoke`。
先运行joint，随后自动运行fixed；每25步原子保存E/G与优化器，相同命令续跑。
两组训练均已完成，`train.complete.json`与`training_audit.json`已落盘，不重新训练。
92点评价、报告和只读恢复检查也已完成，正常无需重跑。
结果与图：[03.15.1 联合训练评价](https://app.notion.com/p/3ed8b22ebd8d81a5a9b3de25abd5830a)。
`evaluation.resume_audit.json`记录禁止重新推理/计算指标的恢复核验；下一阶段已获同意，见03.16。
需要复现时，评价可单独执行`evaluate`或`report`，已完成点校验后复用。
真实码流、结果与图保存在正式目录的`evaluation/`；研究记录仍只在Notion维护。

## 已完成运行入口：ROI 区域训练对齐

配方与实际状态：[03.14 区域训练对齐](https://app.notion.com/p/3ec8b22ebd8d815e9a51f784f515360d)。
保持旧训练/接收源码不变；本轮冻结 UF/E，不训练 Router。

| 操作 | 命令 |
| --- | --- |
| CPU 回归 | `bash demo/run_roi_condition_evaluate.sh test` |
| tmux 中的短训练、断点与解码检查 | `bash demo/run_roi_condition.sh smoke` |
| tmux 中的三组训练 | `bash demo/run_roi_condition.sh train` |
| tmux 中排队评价与作图 | `bash demo/run_roi_condition_evaluate.sh run --max-hours 12` |
| 旧结果只读预检 | `bash demo/run_roi_condition_evaluate.sh history`（tmux） |

正式目录：`/root/autodl-fs/DCVC/runs/a800_roi_condition_20261001`；烟测另加 `_smoke`。
每25步原子保存，同一命令可恢复；GPU互斥锁保证训练和真实UF解码不同时运行。
评价支持单独 `evaluate`／`report`，已完成点校验后复用；排队时间与实际评价耗时分开统计。
不要因为 tmux 已启动就自动结束当前会话，后续安排遵从用户最新指示。

## 历史主线（Old version）

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
下一步用它重建 Generate teacher 并重训 router。该阶段只重做旧协议的标量 QP8 fresh
decode 与 Generate 标签，逐字段复用 Base、Enhance 和 ROI 成本；协议见
[`docs/CLOUD_A800_SEEDVR2_LORA_TEACHER.md`](docs/CLOUD_A800_SEEDVR2_LORA_TEACHER.md)。
如果后续仍不合适，再保持 codec、路由、预算和评估输入不变，只替换 Generate 恢复后端。

租用 A800 时先用 `nvidia-smi -L` 和 `nvidia-smi --query-gpu=name,memory.total --format=csv` 核对实际可见的是完整 80GB 设备，而不是 MIG 切片。当前服务器的系统盘是 `/root`（30GB），数据盘是 `/root/autodl-tmp`（50GB），较慢的 200GB 文件存储是 `/root/autodl-fs`。环境、编译和常用只读模型放数据盘；数据集和正式输出放文件存储。用户上传的五个文件最初直接平铺在文件存储根目录，当前真实位置、缺失下载、解压边界和链接方式见云端存储文档。

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

当前 AutoDL 服务器最初在 `/root/autodl-fs/` 上传了 BF16 DiT，校验后已移到数据盘的常用模型目录；不要再执行上面的整组下载。只在确实缺失时从官方仓库补下 `ema_vae.pth`、`pos_emb.pt` 和 `neg_emb.pt`，保存位置按 `docs/CLOUD_STORAGE_AND_UPLOAD.md` 执行。实际推理必须用单进程 `torchrun` 启动：

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
- `demo/stage_c_seedvr2_lora_teacher.py`、
  `demo/run_stage_c_a800_seedvr2_lora_teacher.sh`：只重建受 LoRA 0.50 影响的 Generate
  teacher 值，复用 Base／Enhance／ROI 标签，随后重训保守共识 router 并重新应用固定的
  空间一致性项；协议见
  [`docs/CLOUD_A800_SEEDVR2_LORA_TEACHER.md`](docs/CLOUD_A800_SEEDVR2_LORA_TEACHER.md)；
- `demo/stage_c_seedvr2_lora_router_evaluation.py`、
  `demo/run_stage_c_a800_seedvr2_lora_router_eval.sh`：在固定 37 条 REDS／UVG 上做旧／新 router
  × 冻结／LoRA 0.50 的 2×2 真实码流评估，只按精确动作图复用，并核对 adapter 强度、
  fresh decode、非 Generate 像素和固定可视化；实验叙述与结果只在 Notion 主线页面维护；
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
