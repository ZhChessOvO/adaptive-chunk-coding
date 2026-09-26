# Project instructions for coding agents

Before changing code or starting an experiment, read these files in order:

1. `README.md`
2. `docs/CLOUD_STORAGE_AND_UPLOAD.md`
3. `docs/CLOUD_A800_PILOT.md`

## Current direction override (2026-09-26)

The user approved the scalable pivot and implementation plan. This section
supersedes every historical "current", "next", or authorization statement below.
Use one immutable full-frame DCVC-UF QP8 base stream, append regional residual
enhancement packets, and keep optional generation outside the base reference
loop. Do not resume spatial-QP sweeps or external-baseline queues automatically.
First implement and fresh-decode a simple transform-residual mechanism baseline;
then prepare mixed REDS/UVG base caches and train a base-conditioned enhancement
codec. The approved learned candidate now uses one optional enhancement layer
per region and UF-aligned 8-frame chunk, conditioned on actual decoded UF
features and base temporal context in analysis, entropy modeling and synthesis.
Do not make a second enhancement layer a prerequisite; preserve the two-level
transform prototype as historical mechanism evidence. Retrain the router only
after its new actions have real measured labels.
The transform baseline is not the proposed learned model or an efficiency claim.

Method and research records live in Notion, not duplicate Git experiment docs:
- Method: https://app.notion.com/p/3e78b22ebd8d81828124c50e8e74c2ca
- Technical design: https://app.notion.com/p/3e78b22ebd8d8181b12fec5580d401a6
- Implementation: https://app.notion.com/p/3e78b22ebd8d81fabb93d6849dd03415
- Old version: https://app.notion.com/p/3e78b22ebd8d817596eef1b4179a6bb4

Keep one A800, tmux/resume, honest data roles, real on-disk byte accounting,
fresh decode, fixed visualizations, and the storage rules below. Once a long
job is stable, keep working on useful next steps and update Notion. The latest
user instruction supersedes the former automatic handoff: do not end a session
merely because a tmux job started; hand off when requested or discussion is needed.
Long training
and later codec/generator adaptation are allowed, not mandatory for each stage.
Coordinate new large downloads with the user. Preserve historical code/results.

## Historical regional-routing handoff (superseded)

The historical research line is budget-conditioned regional Generate / Base /
Enhance routing around DCVC-UF. It is not the historical latent-prediction
line. Preserve all E01-E18 and latent-predictor code as history, but do not
make it a prerequisite for the current work.

The cloud scope remains one A800 80GB.  On 2026-09-20 the user explicitly
authorized the next single-card sequence: validate continuous long-video
coding and overlapping restoration, then run spatial-QP-aware DCVC-UF and
SeedVR2 fine-tuning (including long runs when feasible), regenerate teachers,
and retrain the router.  The 33-frame long-video mechanism smoke is complete;
the 1000-step spatial-QP-aware DCVC-UF fine-tune and its 110-task endpoint
comparison are complete.  It improves REDS rate-distortion but over-adapts on
UVG.  The fixed 0.25/0.50/0.75 interpolation protocol is also complete and
selected alpha=0 (the frozen codec) by the predeclared combined LPIPS BD-rate
rule; see `docs/CLOUD_A800_SPATIAL_QP_INTERPOLATION.md`.  The SeedVR2 LoRA
real-cache, backward, adapter-save, and adapter-reload smoke checks pass.  Its
formal single-A800 run then completed all 560 cache samples and 1000/1000
training steps.  The 37-sample fixed-input evaluation is also complete: the
full-strength adapter improves combined and REDS LPIPS, strongly improves
PSNR, and has mixed UVG LPIPS because some fine-motion textures are
over-smoothed.  The follow-up 0.25/0.50/0.75 inference-strength sweep selected
0.50 as the practical default: it is effectively tied with 0.75 on the
balanced LPIPS diagnostic, but improves 34/37 samples, has the best temporal
average, and is safer on Jockey, ShakeNDry, and YachtRide.  See
`docs/CLOUD_A800_SEEDVR2_LORA_EVAL.md` and
`docs/CLOUD_A800_SEEDVR2_LORA_STRENGTH.md`.  The fixed 0.50 ROI/long-video
reintegration is also complete on the same 33-frame stream: LPIPS, PSNR,
temporal error, and boundary-band error all improve; all non-Generate pixels
remain exact and all 32 adjacent-frame temporal errors improve.  See
`docs/CLOUD_A800_SEEDVR2_LORA_ROI_LONG.md`.  The scalar-QP8 Generate teacher
rebuild, router retraining, and fixed 37-sample old/new-router x
frozen/LoRA-0.50 real-stream 2x2 are now complete.  LoRA supplies the main
quality gain.  The retrained router has a small favorable interaction with
LoRA, but its marginal LPIPS gain is outweighed by more Generate fragments,
slower execution, and slightly weaker PSNR/temporal diagnostics.  The current
balanced performance version therefore keeps the old v6 route and uses LoRA
0.50; the retrained router remains an interaction ablation.  The four-point
real Enhance-budget curve for this selected version is complete.  All 37
samples respond to every adjacent budget change; bytes and PSNR rise
monotonically, while LPIPS improves at every 0->0.25 and 0.25->0.50 step and at
33/37 final steps.  The 9/4, 17/8, and 33/16 temporal-window sensitivity is
also complete: 9/4 is slower and slightly weaker, while 17/8 and 33/16 are
nearly tied.  Keep 17/8 as the general low-latency default and retain 33/16 as
an optional offline-throughput mode.  The bounded spatial-grid and rectangular-
aspect sensitivity check is complete and keeps 4x4 as the general default.
The domain-balanced spatial-QP codec v2 stage is complete.  Fresh 1000-step
25%-UVG and 50%-UVG candidates were compared with the frozen codec and the
REDS-dominant v1 on fixed actual-stream REDS/UVG curves; all 72 logical points
passed stream-size and fresh-decode checks.  Increasing the UVG share only
partly reduced the UVG penalty and also weakened the REDS gain, so the selected
codec is the frozen DCVC-UF.  Keep the three adapted variants as training-recipe
ablations; do not spend time running their rejected mixed-route candidates.
The current performance configuration is frozen DCVC-UF + spatial-QP + the old
v6 route + SeedVR2 LoRA 0.50.  The active stage is external-baseline comparison:
first inventory checkpoints and implementations already present locally, then
define a common multi-rate, actual-stream protocol and clearly separate locally
measured curves from paper-reported numbers.  Per the user's 2026-09-22 scope
decision, compare only DCVC-UF within the DCVC family for now; defer DCVC,
DCVC-DC, DCVC-HEM, DCVC-TCM, DCVC-FM, and DCVC-RT until the paper-table stage.
Prioritize genuinely different generative-video baselines instead.  Check
existing files before any download and coordinate large missing downloads with
the user because server network throughput is slow.  If external curves reveal
a specific restoration bottleneck, a later experiment may compare another
backend while holding the selected codec, routes, budgets, and evaluation
inputs fixed.  Do not start multi-GPU production work without a new user
decision.

The current codec adaptation protocol is documented in
`docs/CLOUD_A800_SPATIAL_QP_FINETUNE.md`.  It uses REDS training data plus the
60 already prepared UVG adaptation windows, region-weighted RD loss, uniform-QP
rehearsal, atomic step checkpoints, and actual-stream fresh-decode validation.
Do not substitute the upstream from-scratch schedule without recording a new
protocol decision.

Respect the data ledger exactly:

- Training may use REDS `train/000..239`.
- REDS `val/000..005` is development data, never an independent test set.
- REDS `val/006..011` and `val/012..023` have already influenced earlier
  experiments. They may be reused for analysis or paper evaluation, but must
  not be described as newly independent evidence.
- REDS `val/024..029` may be used when the experiment records the frozen
  method, exact frame window, crop, and whether the result later influenced a
  design choice. Do not invent a permanent "sealed" boundary.
- UVG, QST and other external videos belong in the cross-distribution paper
  evaluation together with REDS validation data. Report per-dataset results
  as well as the combined result; do not label them as an application test.
- Before every experiment, record each sample's role as training,
  development, previously used evaluation, or new evaluation. Once a result
  changes the method or hyperparameters, update that role instead of still
  calling it independent.

Report all actual on-disk stream bytes, including action maps, headers, masks,
and auxiliary payloads. Codec results must be reproducible by fresh decode
without source frames or hidden command-line state. Never report true-fill as
a result. Keep Generate, Enhance, and joint contributions separate. Evaluate
LPIPS and visual quality as the primary Generate criteria, while also reporting
PSNR diagnostics, temporal quality, complete decode time, peak VRAM, and a
small fixed visual comparison set.

Long jobs must be resumable, emit progress heartbeats, save manifests and
checkpoints, and enforce time and disk limits. Store datasets, checkpoints,
third-party repositories, streams, generated media, and experiment outputs
only in ignored paths. Do not commit large artifacts.

On the current cloud host, `/root` is the 30GB system disk,
`/root/autodl-tmp` is the 50GB data disk, and `/root/autodl-fs` is the slower
200GB file store. The five user uploads originally arrived as flat files under
`/root/autodl-fs`: two DCVC-UF checkpoints, the SeedVR2 BF16 DiT,
`train_sharp.zip`, and `val_sharp.zip`. They may since have been moved,
linked, extracted, or deleted after verification. Always inventory current
files first, reuse existing extracted data, and download only genuinely
missing material from the official source. Extract datasets and keep formal
outputs on the file store; keep the repository, conda environment, source
builds, frequently reused immutable model weights, and at most 8GB of
per-sample scratch on the data disk. The detailed
layout and recovery commands live in `docs/CLOUD_STORAGE_AND_UPLOAD.md`.
