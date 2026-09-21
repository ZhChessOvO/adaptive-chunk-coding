# Project instructions for coding agents

Before changing code or starting an experiment, read these files in order:

1. `README.md`
2. `docs/CLOUD_STORAGE_AND_UPLOAD.md`
3. `docs/CLOUD_A800_PILOT.md`

The current research line is budget-conditioned regional Generate / Base /
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
`docs/CLOUD_A800_SEEDVR2_LORA_ROI_LONG.md`.  The active stage is to regenerate
Generate teacher values with frozen DCVC-UF plus SeedVR2 LoRA 0.50, then
retrain the router without changing the selected codec or data roles.  Rebuild
only the scalar-QP8 Generate input and Generate-dependent labels; reuse the
old Base, Enhance, feature, byte, and measured ROI-cost fields exactly.  The
LoRA latent cache uses all-Generate spatial-QP syntax and failed the required
frozen-teacher replay, so it must not be substituted for the old scalar-QP8
teacher input.  See `docs/CLOUD_A800_SEEDVR2_LORA_TEACHER.md`.  Do not infer
display-quality improvement from training loss alone.  If measured adaptation
remains unsuitable, a later
experiment may compare another generation/restoration backend while holding
the selected codec, routes, budgets, and evaluation inputs fixed.  Do not start
multi-GPU production work without a new user decision.

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
builds, and at most 8GB of per-sample scratch on the data disk. The detailed
layout and recovery commands live in `docs/CLOUD_STORAGE_AND_UPLOAD.md`.
