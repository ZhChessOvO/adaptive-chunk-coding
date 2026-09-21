# A800 单卡 LoRA × router 真实码流评估

## 目的

LoRA 0.50 改变了 Generate 的恢复能力，teacher 重建和 router 重训又会改变区域动作。最终
评估不能只比较“旧系统”和“新系统”两个总数，否则无法区分收益来自哪里。本轮固定同一批
30 条 REDS + 7 条 UVG，用一个简单的 2×2 回答：

| | 冻结 SeedVR2 | LoRA 0.50 |
|---|---|---|
| 旧 v6 router | 已完成的 v6 基线 | 只看恢复器变化 |
| 新 router | 只看路由变化 | 完整性能增强版 |

这 37 条已经用于开发和误差分析，因此本轮回答机制和版本选择问题，不包装成新的独立测试。
不设置硬性的晋级门槛。

## 固定条件

- DCVC-UF 继续使用冻结权重 alpha=0；
- spatial-QP 语法、`Generate/Base/Enhance = QP 8/16/32`、4×4 网格不变；
- 每条仍为 17 帧 512×512，使用相同源视频和随机 seed；
- ROI 上下文 64 像素、处理倍率 1.0、16 像素羽化；
- LoRA adapter SHA-256 和推理强度 0.50 写入计划并在每个 batch 中复核；
- 所有改变的动作图必须重新写入真实 spatial-QP 码流并 fresh decode。

## 怎样避免重复计算

复用只发生在两种能逐项证明等价的情况：

1. 同一样本的新旧 16 个动作完全相同；
2. 动作图没有任何 Generate 区域，此时冻结与 LoRA 后端都不会被调用。

旧 v6 码流和 fresh decode 已经通过正式复核，因此“旧 route + LoRA”直接读取那份 codec
输出，只重做 ROI 恢复。新 route 只编码一次，冻结与 LoRA 两个恢复后端共享同一份 fresh
decode。controller 分数接近不能触发复用。

## 运行入口

teacher／router 正式任务完成后执行：

```bash
cd /root/autodl-tmp/adaptive-chunk-coding
tmux new-session -d -s a800_lora_router_eval \
  'bash demo/run_stage_c_a800_seedvr2_lora_router_eval.sh'
```

正式目录：

```text
/root/autodl-fs/DCVC/runs/a800_seedvr2_lora_router_eval_20260921
```

恢复时运行同一个命令。每个 backend 的 codec、ROI、恢复和评价都有独立 JSON／完成标志，
已完成任务会跳过。tmux 每分钟记录进度、显存和三块盘，每 5 秒记录 GPU 样本。

## 完成检查

汇总必须同时通过：

- 所有流的实际文件大小重新核对；
- 所有新码流逐像素 fresh decode；
- 非 Generate 像素逐像素不变；
- LoRA checkpoint SHA-256 和强度与冻结计划一致；
- 只按精确动作图或无 Generate 情形复用；
- 37 条均生成固定 2×2 输出、动作图和差异图；
- 分别汇总 REDS、UVG、UVG 训练侧与未参与训练侧；
- 报告 LPIPS、PSNR、时序、真实字节、完整 ROI 时间、显存、边界和连通块。

## 当前状态（2026-09-21）

评估入口、自检和“无 Generate 时仍记录 LoRA 身份／强度”的恢复测试已通过。当前不抢占
正在运行的 teacher 重建；待其生成新 route 后冻结正式计划并启动 tmux。
