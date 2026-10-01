# Project instructions for coding agents

Before changing code or starting an experiment, read these files in order:

1. `README.md`
2. `docs/CLOUD_STORAGE_AND_UPLOAD.md`
3. `docs/CLOUD_A800_PILOT.md`

## Current direction override (2026-09-26)

### Cooperation update (2026-09-28; supersedes the next-stage queue below)

The user approved optional same-region Enhance -> Generate, retaining all four
cases: neither, E only, G only, and E followed by G. Low importance does not
require generation. First run the bounded paired probe documented at
https://app.notion.com/p/3e78b22ebd8d8172bd29faf1b7dffc23:
same base/region/E payload, G(base) versus G(received enhanced RGB), original
SeedVR2 versus existing LoRA strengths, then limited context/processing-size
diagnostics. Keep the existing A enhancement model, BF16 assets, immutable UF
reference chain, one optional E layer, single A800, tmux and real fresh decode.
Do not immediately generate mutually exclusive B/E/G teacher labels or train
the router; first measure conditional cooperative benefits. Feature adapters
and joint fine-tuning follow evidence from this probe, not an arbitrary gate.
Preserve the ACSG v1 decoder and its pinned profile byte-for-byte. New cooperation
uses ACSG v2 in separate modules; explicit ROI controls in this diagnostic are
charged and do not claim the future receiver-derived policy is already trained.
Method/results remain in Notion, not duplicate Git experiment documents.

The paired cooperation probe above is now complete. Do not restart its queue
automatically. Results and the next generation-condition adaptation step are at
https://app.notion.com/p/3e98b22ebd8d81e8b0cef5542c73d70d;
consult that page and the roadmap before new training. No new router is trained.

Generation-condition adaptation is now authorized (2026-09-28). Its separate
entrypoint is `bash demo/run_conditioned_generation.sh`: `smoke` validates
actual enhancement-prefix inputs, differentiable frozen-VAE image losses and
exact interrupted/resumed LoRA weights/optimizer; `evaluate --smoke --output
/root/autodl-fs/DCVC/runs/a800_conditioned_generation_20260928_smoke` validates
fresh adapter reload, old-weight receiver equivalence and model-free G-off.
Then `train` runs paired 1000-step latent-only/image-objective adaptations,
and `evaluate` compares the final adapters on the same four development clips.
All GPU stages must run serially in tmux. Reuse the 90 REDS/30 UVG training
cache; keep UF/E frozen and preserve all older stream profiles. The new
ACSG2 receiver uses a distinct profile plus transmitted adapter hash; do not
overwrite the legacy adapter. Progress and recipe, not a duplicate Git report:
https://app.notion.com/p/3e98b22ebd8d8143a89de2d0ea97e5a0.

The paired generation-condition adaptation and all 27 fresh decodes are now
complete; do not restart training automatically. `report` rechecks artifacts
and produces the CPU-only digest, fixed-prefix curves and incremental-E plots.
Results and the feature-condition discussion are at
https://app.notion.com/p/3e98b22ebd8d813dbca5d606f633d9e0.
Keep the image-objective adapter as a research candidate, not a silent change
to old receiver profiles.

The user has now approved the feature-condition continuation (2026-09-28).
Use `bash demo/run_feature_condition.sh smoke` and then `train`, in tmux.
The separate receiver observes decoded P8 delta from the same E packets,
leaving E RGB and UF references unchanged. Its small zero-output-initialized
adapter aligns eight learned phases to the local five VAE temporal positions;
absent packets and the I-frame fallback supply no fabricated delta. This first
profile supports native-scale, aligned crops, not arbitrary resizing. Train
paired RGB-only and RGB-plus-feature 1000-step continuations from the completed
image-objective adapter at strength one, with identical crops/noise/losses.
Reuse the 90 REDS/30 UVG caches; keep original UF/E/DiT/VAE frozen. Both LoRA and
the small feature adapter may train. Do not jointly tune E or start the router.
Both 1000-step arms are now complete (2026-09-29); do not restart training.
Use `evaluate` for the paired real-stream evaluation in tmux: 24 paired points,
four same-LoRA branch-off ablations, and three fresh-decode/recovery checks.
It resumes validated points without redoing them. `report` rechecks artifacts
and creates CPU-only summaries, curves, fixed crops and sequence previews.
These four clips remain development evidence, not independent generalization.
Do not infer interface benefit from a comparison to a less-trained adapter;
compare equal-step RGB and feature arms, then the same LoRA with the branch off.
Discuss the results before another training recipe, joint E update or router.
Interface and training configuration:
https://app.notion.com/p/3e98b22ebd8d81d2b386d5b4b5b4a7c9.
Evaluation status and figures:
https://app.notion.com/p/3ea8b22ebd8d81fabc2bd92ee8cfb87c.

The user approved the interface-only continuation on 2026-09-29. Preserve the
small positive result above; do not turn it into a failed performance gate.
Use `bash demo/run_feature_interface.sh smoke`, then `run`, in tmux. The latter
serializes two 3000-effective-update arms, 38 fresh decodes and CPU reporting.
Freeze the completed RGB LoRA plus UF/E/DiT/VAE; train only the 39,984-parameter
interface. Compare received delta against an equally trained RGB/coverage-only
zero-delta control with paired samples/crops/noise/objectives. Same-weight
zero/shuffle/off receiver ablations distinguish content, alignment and capacity.
No-E is structurally exact and verified, not a meaningless training update.
Use separate hashed receiver profiles and keep every previous module/output.
Reuse the existing 120 mixed caches and assets. Atomic checkpoints every 25
updates, exact resume smoke, shared GPU exclusion and resource heartbeats apply.
Do not start joint E training or the router. Recipe/status/results live only at
https://app.notion.com/p/3ea8b22ebd8d8176a6b2ef459c3f068e.
The paired interface-only training, all 38 evaluations and CPU reporting are
now complete. Do not restart the queue. Results and the next discussion:
https://app.notion.com/p/3ea8b22ebd8d8159bf55ebdd198cae13.
`bash demo/run_feature_interface_analysis.sh` reaudits artifacts and resumes four
observational numerics replays; its `posterior` command resumes two VAE sampling
observations. All six reproduce saved pixels exactly; no weights are updated.
Preserve the pinned training/receiver files at baed308. Training conditions use
posterior-mode BF16 caches, while the receiver samples its VAE posterior and
adds interface corrections in FP32 before DiT autocast. Do not call the old
receiver's addition-site statistics final BF16 measurements. First discuss a
fixed-weight condition-path comparison (RGB and both interfaces, same diffusion
noise) before new training or a different internal injection architecture.

The user has now approved that fixed-weight comparison (2026-09-29). Use
`bash demo/run_condition_path.sh run` in tmux; `test` runs CPU regressions and
`report` reaudits existing outputs. Separate `condition_path_*.py` modules and
hashed bundles select posterior sample or mean. Both execute the upstream VAE
posterior draw, then use the selected condition; paired RNG states and actual
diffusion-noise tensors must match. Keep the existing FP32-add/BF16-DiT cast
policy unchanged to isolate one factor. Compare frozen RGB, trained zero-feature
and real-feature interfaces: 24 full-E points, 6 first-clip no-E points, repeat
and model-free G-off checks (32 total). All old modules, weights and results
remain unchanged. No new training, internal injection architecture, joint E or
router work is implied. Scope/status/figures live at
https://app.notion.com/p/3ea8b22ebd8d817a97e9e1a618b7e37c.
This comparison is complete: 32 fresh decodes, 48 exactly paired noise windows,
15 exact old-output replays, and all 32 points validated through the no-recompute
resume branch. Do not restart it as new training. Posterior mean does not remove
the content-dependent interface tradeoff; keep prior models/profiles. Discuss
the proposed independent internal conditioning branch before implementing a new
architecture or training recipe. An initial dtype-confounded two-decode attempt
is preserved separately as diagnostic evidence, excluded from formal results.

The user approved independent internal conditioning on 2026-09-30. Separate
`internal_condition_*.py` modules preserve every old profile. Use
`bash demo/run_internal_condition.sh smoke`, then `run`, inside tmux. Freeze
the completed RGB LoRA plus UF/E/DiT/VAE; train three paired 3000-step arms:
legacy input addition, internal received features, and internal zero-feature
control. The latter two have equal 124,992-parameter interfaces, injecting once
before block 24 of 32; the input architecture retains 39,984 parameters.
All use posterior-mean BF16 conditions/noise, FP32 interface arithmetic and
identical cached samples/crops/noise/objectives. This unifies those choices,
not the remaining cached-full-frame versus inference-ROI context difference.
Verify image gradients, exact 3+3 versus 6-step weights/optimizer, zero-init,
no-E and model-free G-off before formal training. The run queue serializes
training, 58 fresh decodes, content/alignment ablations and CPU figures; atomic
25-step checkpoints, GPU exclusion and 30-second resource heartbeats remain.
No new assets, joint E update or router training. Recipe/status live only at
https://app.notion.com/p/3eb8b22ebd8d81a3b573f5c4906975ba.
All three 3000-step arms, 58 fresh decodes and the report are now complete.
Do not restart this queue or promote the internal branch automatically.
Results, fixed visuals and the next discussion are at
https://app.notion.com/p/3eb8b22ebd8d8196b4fced245fb8e9b5.
`bash demo/run_internal_condition_analysis.sh` checks all completed resume
branches without permitting inference or metric recomputation, preserves the
formal timing/summary, and summarizes content ablations from existing files.
Keep the formal training/receiver source pins at 98c094f unchanged. Discuss
the proposed equal-budget RGB/internal-real/internal-zero LoRA coadaptation
before new training; it is not yet launched. No E or router updates are implied.

The user approved that three-arm LoRA coadaptation on 2026-09-30 and requested
handoff once formal training is verified running. Use the separate entrypoint
`bash demo/run_joint_condition.sh smoke`, then `train`, inside tmux. All three
arms start from the same completed RGB LoRA and newly zero-output-initialized
internal interface: RGB LoRA only, actual-packet interface plus LoRA, and
zero-packet-content interface plus LoRA. Each trains 3000 steps, with paired
samples/crops/noise, one-third none/partial/full conditions, and 25% UVG sampling.
Preserve the prior internal receiver/model modules exactly; the bundle carries
updated LoRA weights and its distinct weight hash without changing decoding
behavior. No-E rehearsal updates LoRA but skips the interface optimizer; no-E
must equal that SAME new LoRA with its branch off, not the old RGB checkpoint.
Freeze UF/E/base DiT/VAE. Smoke checks exact 3+3/6-step LoRA, branch and optimizer
resume, all-arm image gradients, six fresh reload/fallback decodes and pairing.
The formal queue runs internal, zero, RGB serially, saves every 25 steps and
audits completion. Evaluation/figures follow in the next session, not an
unimplemented automatic stage. Protocol/status:
https://app.notion.com/p/3eb8b22ebd8d81debf34d68b813ddd34.

All three coadaptation arms completed 3000 steps and passed the paired training
audit. Do not restart training. The separate follow-up entrypoint is
`bash demo/run_joint_condition_evaluate.sh run` in tmux. It performs 58 new
fresh decodes: 36 main, 12 same-weight full-prefix off/zero/shuffle controls,
8 own-LoRA no-E equivalence checks, one repeat and one model-free G-off.
Initial RGB's 12 old mean/BF16 points are artifact-checked reuse, not new decodes.
`evaluate` and CPU `report` are also available separately; completed points are
validated before reuse, and the original evaluation summary/timing is preserved.
The queue keeps the global GPU mutex and 30-second resource heartbeats.
Preserve a8348c9 training and earlier receiver/model source pins. No E or Router
training or automatic model promotion follows evaluation. Report the paired
RGB budget control separately from same-weight feature-content ablations.
All 58 fresh decodes, original figures and the no-recompute resume audit are
now complete. Notion read/write succeeded again on 2026-10-01, and the pending
training/evaluation records were synchronized. Results and the next discussion:
https://app.notion.com/p/3ec8b22ebd8d81458221c2e47c864595.
`bash demo/run_joint_condition_analysis.sh` (tmux, CPU only) replays all completed
resume branches with decoding, metric recomputation and formal-summary writes
forbidden. It preserves original hashes/timing and writes independent pixel
comparisons and figures under `evaluation/supplement`; `evaluation/analysis.json`
records the audit. CPU regression: `python -m unittest
demo.test_joint_condition_analysis`, alongside the existing 44-test evaluation
suite. Preserve the pinned training/receiver/evaluation/report files unchanged.
The same-weight interface-off result is a post-hoc research candidate, not a
promoted model; E still decodes and conditions generation through enhanced RGB.
The user approved ROI training/inference alignment on 2026-10-01. Use the new
`bash demo/run_roi_condition.sh smoke`, then `train`, inside tmux; preserve every
historical pinned trainer, receiver and report. This is a paired 3000-step
RGB/internal-real/internal-zero LoRA coadaptation, from the SAME initial adapter
and sample/crop/noise schedule as the previous joint round, not three additional
continuations from its endpoints. Freeze UF/E/base DiT/VAE; no Router training.
The new path crops RGB before receiver-identical mean/BF16 VAE encoding, with a
256-pixel processing window, central 128-pixel image supervision and 64-pixel
context. Latent losses retain the full window. Thus this tests an ROI pipeline
bundle (encoding order AND image-loss scope), not a single-factor causal claim.
Smoke checks exact real-receiver condition tensors including rectangular/33-frame
windows, 3+3 versus 6-step weight/optimizer resume, all-arm gradients, reloads and
fallbacks. Local feature caches are separate and shared across arms; atomic
checkpoints every 25 steps, global GPU mutex and 30-second resource records.
Formal output: `/root/autodl-fs/DCVC/runs/a800_roi_condition_20261001`.
Execution, experiment navigation and figures live in Notion:
https://app.notion.com/p/3ec8b22ebd8d815e9a51f784f515360d.
Do not duplicate the research report in Git. Do not automatically end the user
session merely because tmux starts; the latest request is to keep making useful
progress and organize Notion during training, stopping for substantive discussion.
Follow-up evaluation uses `bash demo/run_roi_condition_evaluate.sh run` in tmux;
the global GPU mutex lets it queue safely behind training. It performs 58 new
decodes and validates/reuses 12 initial RGB points plus 40 prior equal-budget
joint endpoints (36 main, four full-prefix interface-off). The latter are
historical measurements, not new decodes or independent validation. Report both
RGB/real/zero arm pairing and same-LoRA content controls, with fixed old/new
visuals; do not invent a partial interface-off curve. Waiting for training is
excluded from evaluation elapsed time and sampled evaluation GPU peak. `history`
is a CPU-only artifact-reader preflight; `evaluate`/`report` are separate stages,
and `test` checks pairing, resume-without-recomputation and historical scopes.

Router simplification confirmed by the user on 2026-09-28: keep one router
backbone followed by generation-region boundary/fragmentation reduction; do
not restore the former backbone-plus-two-experts design. Context may be used
inside the backbone. Investigate its injection point and ablations when router
training begins; this probe's SeedVR2 crop context is not router context.
Boundary reduction is distinct from output feathering and must retain all four
optional E/G states. This decision does not start router training now.

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
The user has selected the feature-correction candidate as the default next-stage
Enhance implementation: keep UF frozen, predict a display-only feature delta,
and anchor the correction as H(F+delta)-H(F) using the frozen UF reconstruction
head. Retain the RGB-output candidate as a comparison, not the default. This
is a research choice despite mixed metrics, not a claim of universal dominance.
q0.5/q1/q2 denote quantization steps, not model versions or additional layers.
Use demo/run_feature_head_pilot.sh for its reproducible completed pilot; results
and next-stage decisions live at https://app.notion.com/p/3e78b22ebd8d819a86cae48b4fc3d689.
The user has approved improving patch efficiency before router integration:
compact framing, less spatial padding, then measured payload/RD improvements.
ACSE v2 is a lossless binary framing variant, not a second enhancement layer.
The pad16 feature-head model has a distinct format ID; old model/stream behavior
is retained. Use demo/run_patch_efficiency.sh in tmux (diagnostic, then train);
the paired adaptation recipes share GPU 0 and have separate resume states.
Progress and results: https://app.notion.com/p/3e78b22ebd8d81cda7eefa104f8debd7.
The paired 20,000-step continuation, real-stream evaluations and timing are now
complete. Use train_l4/final.pt (A) under a800_patch_efficiency_20260926 as the
next-stage fidelity-oriented Enhance default; retain train_l2 (B) as the
rate-focused comparison. This is a development choice, not universal RD dominance.
Do not restart the completed training pipeline. Its recorded code is fd4e381;
reproducing its strict resume config requires that version, not a modified wrapper.
The next bounded probe measures every complete prefix of the same q=1 encoding,
not a sweep of separately encoded q values. Run demo/run_chunk_enhancement.sh
prefix-probe in tmux; it validates per-prefix artifacts on resume and keeps GT
only in the evaluator. These four previously used clips and their packet-benefit
labels are development diagnostics, not a trained router or independent evidence.
Prefix progress: https://app.notion.com/p/3e78b22ebd8d81e6bde2d5d32be18a8e.
The prefix and bounded generation integration probes are complete. The latter is
`bash demo/run_chunk_enhancement.sh generate-probe` in tmux; atomic per-point
resume, checksummed protocol and native-GPU exclusion are built in. It compares
Base/Enhance/Generate/combined/full-frame Generate on the same four development
clips with fixed geometric actions, not a trained or semantic router. ACSG v1
adds charged, versioned generation controls around unchanged ACSE v2 bytes;
the base and enhanced pixels remain independent of generation. Use the existing
local SeedVR2 BF16 and LoRA-0.50 assets, no new downloads or training. Run
`generate-test` for the 33-test CPU regression suite. Implementation/results:
https://app.notion.com/p/3e78b22ebd8d81379f43cb1a34f44792.
All 28 fresh decodes (including repeats and generation-disabled fallback) and
the artifact audit passed. Use `generate-audit` to recheck completed outputs;
do not restart this pilot as new training. The next stage is mixed REDS/UVG
labels for the new Base/Enhance/Generate actions, followed by transparent
allocation and a lightweight router. Whole-frame LPIPS improves with the fixed
combination, but not every metric improves; do not claim whole-system UF RD
dominance or semantic reliability from four fixed geometric masks.
Do not treat same-q byte savings with changed pixels as equal-quality savings.
Native UF replay failed base-pixel hashes under concurrent GPU training and
passed again when training was briefly paused. Keep native UF coding/decoding
and timings exclusive from other GPU workloads; do not relax pixel validation.
The evaluator queues until cached-feature training exits and serializes complete
evaluations (including child decoders) with a parent-held lock. The exact native
CUDA root cause is not yet established. Cached-feature training itself does not
run the native base decoder.
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
