# Project instructions for coding agents

Before changing code or starting an experiment, read these files in order:

1. `README.md`
2. `docs/CLOUD_STORAGE_AND_UPLOAD.md`
3. `docs/CLOUD_A800_PILOT.md`

## Current direction override (2026-09-26)

### Completed new-latent system review (2026-10-10; latest instruction)

The user requested result verification and Notion updates only, then discussion.
Do not start another training run or promote/change models. The sender's formal
labels, paired 120-epoch fits and `evaluation/complete.json` are complete; old
stage-local `evaluation_pending` flags are historical, not a reason to rerun.
Sender and system `verify` commands passed. A separate CPU-only report is in
`routervc_latent_sender_20261009/report_20261010`; the reproduction entry is
`python -m tools.latent_sender_report` under `tools.storage_guard` in tmux.
It checks prefixes, same-stream G-off, repeats, and summarizes existing records;
it never runs inference or changes completed measurement artifacts. Keep all
previously pinned sources intact. Unit tests: `tools.test_latent_sender_report`.
Current results, resources, training plots and the complete 13-view gallery:
https://app.notion.com/p/3f58b22ebd8d8145a663d10ffabeb5c7.
Only operational handoff belongs in Git; research conclusions stay in Notion.

### New-latent source-aware sender continuation (2026-10-09)

The new-latent core R_g teacher, 120-epoch fit and 156-point fixed-E review are
complete. Do not repeat them. The independent read-only audit is in
`runs/routervc_latent_routers_20261008/receiver_audit_20261009`; its entry is
`python -m tools.latent_receiver_audit` in tmux. Keep pinned receiver code intact.
The adapted core was selected from the completed review for the already approved
R_s continuation, including its REDS full-E LPIPS exception. Exact checkpoint:
`routervc_latent_routers_20261008/formal/router/core/best.pt`, SHA256
`62f02fe9a1d0fe008c0f93c5c6bf0747c7a470cf2d8eba90633795cf0bb7fe7b`.
Notion results and all fixed views:
https://app.notion.com/p/3f38b22ebd8d81b9a240f004b5f22c81.
Sender implementation and progress:
https://app.notion.com/p/3f48b22ebd8d81ec8328e21aa31f4192.

New `routervc/latent/sender*.py` modules reuse the independent multiscale R_s
architecture and paired optimizer engine, never old feature-patch labels or
RGB pasting. Every parent/child Y is decoded from real latent packets, fixed
R_g reruns, then frozen G produces final whole-picture marginal labels. Two
states and three measured additions per state give eight renderings and six
signed labels per window. Unknown candidates stay masked. Preserve 90/30
REDS/UVG views and the 96/24 grouped split. Fit source/zero_source for 120 epochs
from paired random R_s initialization; do not share R_g weights or scales.

Use `bash tools/run_latent_sender.sh tests|smoke|train|all|verify` in tmux;
`all` requires and verifies the real smoke before preparing formal labels and
fitting. Check completion receipts before resuming. Output and compact cache
are both in `runs/routervc_latent_sender_20261009` because the data disk is near
its 80% guard. No datasets or previous outputs are deleted. Teacher saves each
whole received state; fitting saves both optimizers/RNG after each paired update.
Native GPU mutex, 30-second resources and byte/inode guards remain mandatory.
R_s plans a budget-independent conditional ordering using actual mixed decodes,
without executing G; budgets truncate complete two-P8 bundles. No I E packet,
extra E/G/protection mask, source feature or sender model is transmitted.
Keep width3, UF, E format, G and selected R_g frozen. Adaptive width and further
fusion/joint training in the user's `1008交流` note are future discussion items,
not authority to silently change this experiment. Do not end merely at launch.

The automatic post-fit review is `bash tools/run_latent_system_review.sh --wait`
in a separate tmux; it waits without the GPU mutex, then verifies the completed
sender before GPU evaluation. Validate that path with `--smoke --wait` first.
It compares source/zero_source/fixed-order/source-G-off at four actual E-byte
caps with native UF I32/P0..48 on the same 13 reused diagnostics. Formal output
is `evaluation/` under the sender root; `--verify-only` audits completed files.
Do not confuse E-byte fractions with E-region counts or this diagnostic with
full benchmarks. No model is promoted or trained by the evaluator. Resource
timings include fresh-process setup/saving; sender order timings exclude bank
preparation. CPU entropy tests need the established environment PATH and
`TORCH_EXTENSIONS_DIR`, not a new extension build under the system disk.
Both real smokes have passed (18 sender fresh decodes and 22 system-review
fresh decodes); the formal queue started on October 9 at 10:05 Asia/Shanghai.
Read `formal/teacher.progress.json` before assuming optimization has begun.
The evaluator was restarted only while waiting, to load a figure-only caption
fix; its queue has an explicit intentional-restart receipt. The sender was not
interrupted. Original smoke images remain untouched; the CPU-only redraw is
in `evaluation_smoke_layout/`. Replot finished measurements with
`--replot-to /absolute/new/output`, never overwrite completed artifacts.

### New-latent dual-Router adaptation (2026-10-08; latest authorization)

The user approved the next step after compact E and frozen-G diagnostics:
re-measure labels on new width3 B/E, adapt independent core R_g, then bind that
receiver to new final-marginal labels for source-aware R_s. This supersedes the
previous no-new-Router-training boundary, not the frozen UF/E/G weights.
Preserve asymmetric networks, the 90 REDS resized full views + 30 UVG crops
and existing 96/24 Router sequence split. Do not use old feature-patch labels
or paste full-E RGB cells onto B: each mixed Y must be entropy-decoded from
the actual latent packets. First actions are two-P8 region bundles; I has no
new-E packet. No extra E/G/protection masks or untrained semantic heads.

New modules: `routervc/latent/{routing,router_data,receiver_fit}.py`;
new tools: `latent_router_queue.py`, `latent_router_worker.py`, and
`run_latent_routers.sh`. RVLRG001 has a separate receiver/G identity envelope;
region-addressed noise and ungenerated Y condition every selected G core.
Long jobs require tmux, shared native lock, byte/inode guard, per-cell atomic
teacher journals and optimizer/RNG checkpoints. Run `bash
tools/run_latent_routers.sh smoke|train|all|verify` inside tmux; `tests` runs
the focused CPU contracts. `all` runs verified smoke, 120-window teacher,
then 120 epochs of core-only R_g. It does NOT automatically start R_s before
reviewing receiver results. Reuse completed receipts, never rerun blindly.
Formal outputs: `runs/routervc_latent_routers_20261008`; losslessly compressed
Router input cache: `/root/autodl-tmp/DCVC/cache/routervc_latent_routers_20261008`.
Research plan/progress: https://app.notion.com/p/3f38b22ebd8d81b9a240f004b5f22c81.
Only operational instructions belong here; research tables/figures in Notion.
`bash tools/run_latent_receiver_review.sh --wait` in a separate tmux waits for
formal R_g completion and compares old/new core plus G-off on the existing
13 diagnostic views, with identical fixed E prefixes and actual header bytes.
It never starts R_s or promotes a receiver automatically. Source and model
bindings are explicit; new profile results must not be mixed with RVLGEN01's
fixed-center G4 diagnostic. Figures and training traces are generated on finish.

### Compact packets and G reconnection (2026-10-08; completed prior stage)

The user approved: reduce small-packet overhead, check continuous regional E,
then reconnect existing frozen G for a bounded paired comparison. Keep width3
and frozen UF; do NOT start B-clarity optimization, new G/Router training, or
deploy old Routers as if they had been adapted to the new latent representation.
This supersedes the October 7 G-off boundary for the new diagnostic only.
Research evidence and progress live in Notion:
https://app.notion.com/p/3f38b22ebd8d8139b98aeb3f79198e02.

New files `compact_entropy.py`, `compact_rans.cpp`, `packet_format.py`,
`packet_codec.py`, and `generation.py` under `routervc/latent/` leave October 7
profiles unchanged. RVLPACK2 uses the SAME conditional CDFs in one rANS state
per region, with addressed `(chunk, region)` packets. Its inner B is unchanged.
The separate RVLGEN01 envelope declares a shared fixed-center G4 diagnostic,
not a learned receiver Router; geometry and noise are paired across E prefixes.
Every header is charged, no action/protection mask is sent. G-off must not read
G model assets. G display never enters the B reference loop.

Results root: `runs/routervc_latent_20261008`. tmux entrypoints:
`tools/run_latent_packets.sh run|verify --mode single --limit 4` and
`--mode continuous --limit 6`; `tools/run_latent_generate.sh run|verify`;
`tools/run_latent_followup.sh native run|verify` for matched 17-frame UF anchors.
`run_latent_followup.sh wait` waits for continuous completion before G smoke,
the frozen G comparison and native anchors; it never trains models. Same
commands resume artifact-checked samples. Read completion receipts first.
All builds and GPU work use the configured environment, tmux, shared native
lock and three-mount/inode protection. New long REDS windows are 41/38 frames;
UVG remains existing crops. These are reused diagnostics, not full benchmarks.
CPU reports: `python -m tools.plot_latent_followup --wait` and
`python -m tools.plot_latent_resources`, also in tmux. Counter resets inside G
require maximum memory across ALL calls; fresh loading time is not throughput.

The October 8 packet, continuity, frozen-G and matched-native stages are now
complete and independently verified. Reports are in `report/` and `resources/`
under that run root. Do not restart completed inference or infer authority for
new training from this entry; consult the Notion result and the user's next
decision before adapting the two Routers to the new B/E distribution.

### Frozen-UF latent scalability (2026-10-07; prior isolation stage)

The user authorized the 1007 plan after the completed dual-Router result review:
https://app.notion.com/p/3f28b22ebd8d8143b641c38667998b54.
Build B and one E from ONE high-quality UF encoding at fixed q*: B is its coarse
representation, NOT another QP8 stream. Preserve frozen UF weights and its native
decoder. First verify symbols and the native-full-same-context endpoint, then
real entropy streams/fresh decode, B-only reference continuity, and regional E.
G/Router are OFF ONLY for isolation of this first E probe. Final scope still
includes sender-selected regional E, receiver-selected G and B/E/G/EG states.
Do not remove them, restore old feature-delta enhancement, or retrain them now.
Keep all historical pinned modules/models/results unchanged; use separate
`routervc/latent/` and `tools/` modules/profiles. The native inspection bridge is
version-bound research instrumentation, not a change to the installed extension.
Long builds/evaluation require tmux, global GPU exclusion, atomic resume,
three-mount/inode heartbeats and real byte accounting. Continue useful work
after tmux launch; discuss substantive choices. Research records only in Notion.
All older "latest", pending and automatic-handoff paragraphs are historical.

October 7 execution: A, B, C and the matched native-QP anchors are complete in
`runs/routervc_latent_20261007`; use their `verify` entries, not repeated inference.
The user accepts width3's blur and prioritizes measured savings / regional E,
not a stricter coarse-picture quality gate. RVLR1 tests width3/q*=48, one P8,
fixed center-out 4x4 latent tiles and E0/4/8/12/16; this is NOT trained routing or
byte-percentage allocation. Regional evaluation (28 fresh decodes) and its CPU
report are complete; inspect their completion receipts. No new training, G or UF updates are authorized by
these diagnostics. Region packets locate themselves, no extra action mask.
Entropy packets are independently decodable from B; native synthesis remains
spatially coupled. Do not claim received-tile pixels match the full endpoint.
Preserve the completed source/profile hashes, including the new sidecar modules.
Results hub: https://app.notion.com/p/3f28b22ebd8d81489980feaa14ceae3c.

### Asymmetric dual Routers (2026-10-05; authorized, latest)

October 7 continuation: R_s source/zero_source training, 117 fresh evaluation
points, prefix/repeat checks, and both CPU reports are complete. The user now
prioritizes consolidating the current system's RD, qualitative comparisons and
resource use, not designing another training run. Do not automatically start
on-policy alternation. Current result hub:
https://app.notion.com/p/3f18b22ebd8d81fabf9fc4ccdb4cf5ae.
Supplemental CPU entry: `python -m tools.plot_routervc_overview` in tmux;
completed runs verify inputs/artifacts without redrawing. Preserve all pinned
training/evaluation/report code. The October 6 launch descriptions below are
historical; earlier-stage pending flags do not override later completion files.

October 6 continuation: both R_g arms finished 120 epochs, and fixed-E evaluation
and reporting are complete. Inspect `formal/router/complete.json`,
`evaluation/complete.json`, and `report/complete.json` together: pending fields
in earlier-stage receipts are historical, not reasons to rerun inference.
Results/figures: https://app.notion.com/p/3f18b22ebd8d81639ef5d4fd3989e8c9.
The user explicitly selected core and authorized continuing R_s. Bind
`runs/routervc_receiver_20261005/formal/router/core/best.pt`, SHA256
`53b5b5fb9a8b5ee0c834e8af3cfbdf7c95a0e031c8c55a1dc882bb541deaca02`;
the completed training receipt is in the same `formal/router` directory.
The real R_s smoke passed: 16 final renderings / 12 marginals, exact paired
model/AdamW/RNG resume, eight actual-G teacher/fresh checks, four sender-prefix
checks (12 prefix pairs plus eight G-on/off fresh decodes). Formal queue started
October 6 15:21 Beijing in `routervc_sender_train`: first prepare all labels,
then optimize both arms; do not equate teacher rendering with optimizer steps.
Use `bash tools/run_routervc_sender_guarded.sh` with `smoke`, `train`, or
completion-only `verify`. Core is not a claim of a major RD gain; retain old
shared and halo controls. CPU regression: 443 RouterVC + 20 maintenance/plot tests.

October 5 maintenance is historical: inode exhaustion interrupted R_g at paired
update 42; verified frame TAR archival and an external storage guard recovered
the original experiment. The user SUBSEQUENTLY authorized permanent retirement
of the listed obsolete runs, including those four TARs and their JSONs. They
are no longer restorable from server archives; no downloaded backup exists.
Exact deletion/completion receipts: `runs/maintenance_20261005/retirement/`.
Datasets, current dependencies, and the retained sample ledgers were preserved.
Do not rerun retirement or assume old Notion/local artifact paths still exist.
The guard preserves >=5,000 free file entries and the original 80% byte limit;
resource logs stay on the fast disk. Never edit pinned research code to add it.

The user approved implementing the updated two-network plan after the completed
03.21 training. First audit idle RAM and modestly clean obsolete file-store
artifacts, never datasets. That maintenance is complete: high RAM was reclaimable
page cache, not zombies; the exact eleven-file, 1.532 GiB cleanup is recorded in
`/root/autodl-fs/DCVC/runs/maintenance_20261005/cleanup_manifest.json`.
Do not drop caches or kill idle tmux shells to claim a memory improvement.

Authoritative design: https://app.notion.com/p/3ef8b22ebd8d81e39a48efa151655a0f.
Execution: https://app.notion.com/p/3f08b22ebd8d8141a824faf4872bc9e2.
Keep research prose/results in Notion, not duplicate Git reports. Continue useful
implementation/evaluation work after tmux launch unless the user requests a
handoff or a material decision requires discussion; old exit-on-start directions
below describe previous sessions, not this authorization.

R_g is a lightweight, independent receiver-only G utility model with three
trained outputs. Its only inputs are decoded B, actual received Y and packet
coverage. Remove old E and untrained semantic outputs. R_s will be a DIFFERENT
source-aware, multiscale spatiotemporal network using all frames of the current
window, original/reconstruction differences, candidate packets and actual bytes.
No shared parameters or old expert branches. The sender's measured final-gain
labels must rerun fixed R_g after adding a packet, not sum isolated E/G gains.
R_g with fixed E and then R_s are complete; measured on-policy alternation is
a possible later step, not the current task. Freeze UF/E/G and retain qE=2.

Current receiver entry: `bash demo/run_routervc_receiver.sh test`, `smoke`,
`train`, and completion-only `verify`. Long jobs require tmux and the single-GPU
mutex. Root: `/root/autodl-fs/DCVC/runs/routervc_receiver_20261005`; compact cache:
`/root/autodl-tmp/DCVC/cache/routervc_receiver_20261005`. Keep the 90 REDS resized
full views + 30 existing UVG crops, 96/24 sequence split; no downloads. Reuse E0
and old E4/E8 measured G targets, newly measure E2/E12/E16, all REGION counts,
not byte fractions. Six states, core/halo equal-capacity R_g candidates, same
mixed_core best initialization, 120 epochs, G-only Huber + within-view ranking.
Core/halo are candidate input ablations, not sender/receiver or two experts.
The data LRU is bounded at eight windows. Atomic checkpoints preserve both
optimizers and within-epoch position; final export reconstructs JSON history
from the authoritative checkpoint even after a last-epoch crash.

New source-free receiver uses RVRC/v1, only R_g identity, zero E/G/protection
masks, and measured header/packet bytes. G-off needs no R_g/G assets and no-E
needs no E asset. Do not modify any pinned old source or silently change an old
profile. Future R_s is not a receiver dependency. True prefix ordering must be
fixed independently of later truncation budgets. Local G regret is not whole-
video RD or semantic protection; fixed-E fresh decode and whole-view comparison
remain necessary before promotion. Preserve B/E/G/EG options and negative gains.

03.21 is verified complete (120 epochs per arm, 3,840 measured mixed G labels);
keep its best/last/control models and labels. It is historical shared-model
training, not a completed asymmetric system. The first interrupted receiver
smoke attempt was stopped for a final-checkpoint export fix, not quality failure;
its obsolete directory was subsequently retired with the user's authorization.

Historical receiver launch verification at 2026-10-05 15:14 Beijing: the two-window smoke
completed 96 new and 96 reused G labels, exact GPU model/AdamW resume, and 14
fresh/teacher checks (six actually run G). New RVRC headers measure 310 bytes.
All 354 RouterVC CPU regressions passed. Formal core/halo optimization started
in `routervc_receiver_train`; both optimizer steps advanced and the saved
weights reload finite and changed. Inspect `formal/router/progress.json` and
`formal/router/resume.pt`, not GPU utilization alone. Labels are prepared lazily
and later epochs replay them. Sender architecture GPU smoke is synthetic only,
not completed sender training. Receiver training and its evaluation are now complete.

Fixed-E evaluation and CPU reporting completed in `routervc_receiver_eval` and
`routervc_receiver_report`; waiting never held the GPU mutex. Read-only completion
checks are preferred; the existing idempotent entries are
`bash demo/run_routervc_receiver_evaluate.sh run --wait` / `report --wait`.
Reuse 39 authenticated 03.20 points, fresh-decode core/halo on the exact same
E0/E25/E50 packets (78 new points + eight checks), then generate separate REDS/UVG
curves and all 13 fixed visualizations. These remain diagnostic windows, not a
full independent benchmark. Keep old seed 20261003 for that matched comparison.

Source-aware R_s implementation/preparation is tracked at
https://app.notion.com/p/3f08b22ebd8d8164a6b2cd6061dd3fcb.
Its 142,897-parameter 3D network passed synthetic GPU exact resume; 430 combined
RouterVC CPU tests passed before launch; real sender smoke is now complete,
and the formal sender teacher/training queue has started. The new
`run_routervc_sender.sh` supports `test`, `smoke`, `train`, `verify`. First launch
requires explicit `--receiver`, `--receiver-sha256`, and for formal training
`--receiver-complete`; subsequent same-stage launches resume recorded bindings.
Do not launch it before selecting completed R_g from the fixed-E comparison.
Smoke and formal sender stages must bind the SAME R_g/G. A preliminary smoke
using temporary receiver weights must use a separate output root, never weaken
the matching check or overwrite the immutable final-root protocol.

Sender root/cache: `runs/routervc_sender_20261005` on fs and
`DCVC/cache/routervc_sender_20261005` on tmp. Complete the fixed final labels
before fitting TRAIN-only scales or optimizing either source/zero_source arm.
Keep the LRU bound of two; do not save all formal full-RGB renderings. The
conditional encoder reuses authenticated q2 candidates and a saved budget-
independent order; cache reuse is not fresh encoding time. On smoke recovery
reuse `plan.json` rather than regenerating its timing-bearing payload. Recipe,
limits and next-stage decisions live in Notion, not a second Git research log.

Sender evaluation and CPU report wait in `routervc_sender_eval` and
`routervc_sender_report`, holding no GPU mutex while waiting. Entry:
`bash demo/run_routervc_sender_evaluate.sh run --wait`, `report --wait`, or
completion-only `verify`. Evaluation pins new code only when formal R_s finishes;
do not alter it once `evaluation/protocol.json` exists. It compares source,
zero_source and fixed old E selections with the EXACT training R_g/G policy
(seed 20261005), so old E is re-decoded rather than reusing 20261003 scores.
117 fresh points, 13 source-free full-E candidate decodes, eight checks, and 78
literal prefix pairs precede separate REDS/UVG curves and all 13 fixed visuals.
Only old entropy packets are reused: allocation time is NOT full encoding time.
Same E caps are not necessarily equal realized rates. No automatic sender
promotion/on-policy adaptation; inspect results and discuss meaningful choices.

`routervc_sender_targets` is a CPU-only label diagnostic waiting for ALL formal
labels. Entry: `python -m tools.plot_routervc_sender_targets --stage <formal>
--output <separate-output> --wait` inside tmux. It has seven unit checks plus
real-smoke render/read-only replay coverage. It uses saved metrics/streams only,
never changes labels, training or selection. Direct-Y and final-R_g/G marginal
differences include changed G inputs, selected regions AND ordinal noise; do not
call them an isolated generation-synergy estimate. Keep TRAIN/validation apart.

### Mixed-reconstruction Router continuation (2026-10-04; completed, historical)

The user approved the next objective after 03.20 and requests session handoff
once formal optimization is demonstrably running. Freeze UF/E/G and retain qE=2.
Use the same 90 REDS resized full views + 30 existing UVG crops and 96/24
sequence split. No downloads, content heads, extra masks, experts or G merging.
Three equal-capacity continuations start from the same completed q2 global/local
Router: isolated_core control, mixed_core, mixed_halo. Each runs 120 epochs.
Mixtures use deterministic nested 4/8 REGION subsets, not byte-budget fractions.
Measure conditional G on actual mixed Y; do not sum old isolated G labels.
Halo input uses G's clipped 64px neighborhood at the unchanged 64px CNN input;
this includes a context/detail sampling tradeoff, not a pure capacity increase.

Use `bash demo/run_routervc_mixed.sh test`, then `smoke`, then `train` in tmux.
First epoch prepares each missing fixed-teacher sample before optimizing it;
later epochs replay those same labels. This is not on-policy or pseudo-label
training. Atomic checkpoints contain all three models/optimizers and the exact
within-epoch cursor. G results are separately atomic per region. Preserve G's
precision flags across Router updates. Select best on held-out mixed-Y local
conditional G4/G8 regret with dataset-equal averaging, loss tie-break; also keep
last epoch. This is not a whole-video RD metric or content-protection result.

Output: `/root/autodl-fs/DCVC/runs/routervc_mixed_router_20261004`.
Compact cache: `/root/autodl-tmp/DCVC/cache/routervc_mixed_20261004`.
Notion 03.21: https://app.notion.com/p/3ef8b22ebd8d8136941ad2d5b7c1cf43.
The queue enforces the shared GPU mutex, 30-second resource heartbeats and disk
limits. Smoke checks paired optimization/restart/best-state equality, actual
partial-E fresh decode, fixed diffusion noise and model-free G-off. Do not call
teacher preparation alone formal training: verify an advancing optimizer cursor
and loadable checkpoint, update Notion/Git, then end this session as requested.
Evaluation and deployment-profile integration are for the next session; do not
automatically launch them or feed these separately formatted models to old
receivers. Historical pinned modules/weights/results remain unchanged.

Launch handoff: all 313 RouterVC CPU tests passed. The two-window smoke measured
64 mixed G labels, checked exact GPU mid-epoch model/optimizer/best-state resume,
and passed six fresh/G-off checks. The separate `routervc_mixed_interleave_check`
passed REDS/UVG before/after live Router updates with exact G pixels/noise and
restored precision flags. Formal `routervc_mixed_train` tmux is running: all
three optimizers advanced and the epoch-one cursor checkpoint reloads with
finite, changed weights. `formal/router/resume.pt` is authoritative; progress
is in `formal/router/progress.json`, child logs in `queue/formal_training.log`.
Resume with the same `train` command after a restart, never repeat completed
teacher cells. On return inspect these files/complete markers before acting.
The additional interleave audit is `python -m torch.distributed.run --standalone
--nproc-per-node=1 demo/routervc_mixed_interleave_check.py` in the project CUDA
environment/tmux; it is read-only when complete. Stop this session after Notion
handoff as requested, leaving training running; do not auto-launch evaluation.

### Cheap-packet Router adaptation (2026-10-04; completed, latest)

The user approved Router-only training after the completed efficiency results.
Freeze UF and completed joint E/G weights; teach the same single visual body
the measured qE=2 E and conditional G gains. Reuse authenticated unchanged B/G
teachers; re-encode/decode q2 E and regenerate EG. Keep the 90 REDS resized
full-view + 30 UVG crop pool and its 96/24 sequence-group split. No new semantic
heads, experts, merged G scheduling, E/G fine-tuning or downloads in this stage.
Train both equal-capacity global/local arms for the same 120 epochs; use actual
q2 bundle bytes to allocate, then evaluate against the old Router using q2 too.
Do not attribute cheaper payload gains to Router retraining. Charge all headers,
preserve source-free fresh decoding, literal within-q2 prefixes, no extra masks.

Use `bash demo/run_routervc_light_router.sh smoke` then `run` inside tmux.
Both commands resume and share the GPU mutex through their child jobs. The
run queue prepares teachers and trains both Routers; evaluation is separate.
Formal root: `/root/autodl-fs/DCVC/runs/routervc_light_router_20261004`.
Fast compact visual cache: `/root/autodl-tmp/DCVC/cache/routervc_light_20261004`.
Current scope/progress: https://app.notion.com/p/3ef8b22ebd8d810bb1d7ec68e38de801.
`bash demo/run_routervc_light_evaluate.sh run` waits without the GPU mutex for
both formal models, then compares old/new and global/local on the same q2 bank
and 0/25/50 percent E-byte caps, G8. Byte-identical old E0 results are authenticated
reuse; other points are fresh decodes. This differs from the old q1-location q2
probe. Do not call equal byte caps equal realized bpp. Report source groups
separately, model-free G-off, true prefixes and all-G-call memory maxima.
All historical pinned files remain unchanged. Do not exit merely because tmux
starts; continue useful work and update Notion until discussion is needed.

Completion handoff: teacher, paired 120-epoch Routers, all 156 evaluation points,
four extra repeat/G-off checks, 104 literal prefixes and the joined report are
complete. Evaluation/report read-only re-entry passed; inspect complete markers
instead of restarting inference. The CPU-only `routervc_light_label_report`
reads saved labels/training histories into `label_diagnostics`; it never selects
a different checkpoint. 304 RouterVC CPU tests passed. Results/RD and all fixed
images/route maps are linked near the top of Notion 03.20; direct result page:
https://app.notion.com/p/3ef8b22ebd8d815382d5e9a655d9cf21.
No model was promoted and no new GPU task is running. Discuss the next objective
(mixed-reconstruction supervision/context placement and validation selection)
before new training or changing frozen UF/E/G. These are proposals, not approval.

### Fixed-weight transmission/execution efficiency (2026-10-04; completed)

The user approved the post-03.18 efficiency probes and explicitly values VRAM
savings independently of decoder speed. Retain the original low-memory tiled
G option; do not impose a speedup gate or infer consumer-GPU compatibility from
A800 PyTorch allocation statistics. UF and E already use native rANS; do not
describe the next stage as adding missing entropy coding.

Current output: `/root/autodl-fs/DCVC/runs/routervc_efficiency_20261004`.
Progress, results and figures belong in Notion 03.19 only:
https://app.notion.com/p/3ef8b22ebd8d81fbb362f658f696a241.
The entropy audit is complete: inspect `entropy_audit/complete.json` and resume
read-only, not another inference run. The q2 probe keeps the original q1 E25/E50
packet selections; those names are NOT fractions of the new q2 bank. It uses
the existing q-conditioned E weights and the same receiver policy, not retraining
or automatic promotion of the global/local Router. Each q has true packet
prefixes; q1 versus q2 is independent re-encoding, never a prefix claim.

The bounded-pair G probe uses the old q1 E50/G8 streams and a new shared policy
fingerprint. Only adjacent selected cores merge, up to two cells and 1.5x the
old maximum processing area; original feather/writeback support stays fixed.
This area cap is NOT a VRAM guarantee. Preserve unmerged seed ordinals; different
merged shapes do not have pixelwise paired noise. No E/G/protection mask is sent.
Do not modify pinned historical codec/receiver code or old result JSONs.

Memory correction: SeedVR2 resets CUDA peak counters per ROI. Old top-level
decode.json peak fields can therefore represent only the last G call. Use the
maximum across ALL recorded G calls for like-for-like historical comparisons.
The scheduled receiver additionally accumulates peak allocations before every
reset, including UF/E and loading. Do not equate these with total device VRAM.

Queues: `run_routervc_qstep_probe.sh run`, `run_routervc_schedule_probe.sh`,
and `run_routervc_entropy_audit.sh run`. Check complete markers first; all
queues require tmux, share the single-GPU mutex and resume atomic results.
Do not exit only because tmux starts. New long training/changed objectives still
require discussing the efficiency results first; frozen probes are authorized.

Completion handoff: all three probes, the CPU geometry report and joined
`report/complete.json` are complete and read-only verified. Do not restart
completed inference or training. The q2 probe has 26 points and 13 true prefixes;
the pair schedule has 13 points plus two repeated fresh decodes. These are
separate changes, not a tested q2-plus-merged combination. Keep both execution
options and both visual Router arms; no semantic heads/model were promoted.
Results and the 13-window five-column gallery are linked near the top of
Notion 03.19.1: https://app.notion.com/p/3ef8b22ebd8d81698ef2e655e9295450.
The CPU-only joined report is `python -m demo.routervc_efficiency_report`;
completed replay verifies original hashes without rewriting metrics or timing.
292 RouterVC CPU tests passed. The next training priority is a discussion point,
not authority to launch low-rate E/G or new Router training automatically.

### Full-view and content-aware Router revision (2026-10-03; latest)

The user approved full-view/global-local Router inputs and content-importance
supervision while preserving one backbone and B/E/G/EG. Importance and content
harm come from offline training labels, never transmitted protection hints.
Sender selects E packets; receiver derives G from decoded B/Y/coverage. No
separate E/G mask or protection map is sent. Packet coordinates, timing and
necessary headers still count; RTVC v1 already has zero explicit G-map bytes,
so do not invent a new saving by subtracting zero a second time. Optional user
no-generation regions are receiver-local settings, not the default wire method.

First establish full-frame data and broader current-model evaluation, plus
honest content-label contracts; do not call these steps completed Router
training. Keep prior E/G weights and pinned modules unchanged. Add isolated
modules for this revision; defer physical demo/ refactoring. Preserve local
ROI-aligned training when later exploring mixed spatial scales. Only actual
content annotations/metrics may justify semantic labels; unknown is not safe.
The user confirmed the first content-aware stage targets readable text/digits
and face geometry (not identity recognition). General subject importance is a
later extension, not a prerequisite. Use per-category masked supervision when
only partial labels exist; never fabricate absent/safe labels to fill a table.
The 25,338-parameter visual backbone and its paired global/local perceptual
baseline are separate from content-supervised training. Do not deploy untrained
semantic heads or describe teacher generation as a completed Router training.

REDS originals remain available; existing UVG PNGs are 512px crops. The user
explicitly approved proceeding with REDS full-view + existing UVG crops for
mixed training, without waiting for full UVG uploads. Preserve their distinct
view provenance and report REDS full-view and UVG crop evaluation separately.
Restoring full UVG is an optional later improvement, not a current blocker.
No large server downloads. If archives eventually arrive, remove them only
after verified extraction and retain the sole full-frame source. Never label
UVG crops as full-view. Distinguish resized full-frame from native resolution,
and keep padding out of metric/valid-pixel denominators.

Implementation, decisions and progress belong only in Notion 03.18:
https://app.notion.com/p/3ee8b22ebd8d813c9dbedd66dd5eb4a6.
Output: /root/autodl-fs/DCVC/runs/routervc_revision_20261003. Keep tmux/resume,
single-GPU exclusion, real bytes/fresh decode and three-mount heartbeats.
Current entrypoints: run_routervc_mixedview_teacher.sh for the measured teacher,
run_routervc_visual_train.sh for the perceptual baseline, and
run_routervc_revision_queue.sh for their supervised sequence after teacher smoke.
Visual input caches may live on /root/autodl-tmp/DCVC/cache/routervc_visual_20261003;
formal weights and results stay on autodl-fs. Check actual completion markers and
Notion before starting a queue; do not repeat completed inference.
The independent run_routervc_visual_evaluate.sh queue waits for both formal
visual Router models without holding the GPU lock, then checks fresh/repeat/
G-off decoding and evaluates fixed REDS full-view and UVG crop samples. It
rejects smoke weights, does not promote a model, and uses boundary_lambda=0
to isolate the global/local comparison. See Notion 03.18.3 for the scope;
resume verified points instead of restarting inference.

Completion/resume handoff (2026-10-04): the mixed-view teacher, both formal
120-epoch visual Routers, and all 169 first-batch evaluation points are complete.
The independent UF quality-index 40/48/56 supplement and bounded 26-point
zero-E/G8 control are complete and read-only verified. Check
`visual_zero_E/complete.json` instead of restarting it.
`visual_ablation_report/complete.json` identifies the completed, verified
CPU-only joined report; do not overwrite any source evaluation.
The saved-pixel G-off scores are not new decodes or cheaper E-only streams.
Original outputs/timing and source pins stay unchanged; do not restart training
because older status paragraphs say it is pending. Results, fixed images and
the next research decision live in Notion 03.18.3, not new Git experiment docs.
Offline content review is diagnostic only: unverified OCR is not character
truth, missing detections are unknown, and repeated regions/frames are not
independent examples. No semantic heads were trained or enabled. Discuss the
next training/architecture objective before promoting either Router or changing
the frozen completed E/G weights; retain single-backbone B/E/G/EG and zero
transmitted masks. Region-context overlap is processing-volume evidence, not
FLOPs or a demonstrated scheduler speedup.

### RouterVC complete workflow (2026-10-03; completed earlier phase)

The user named the method RouterVC and authorized completing its implementation
workflow. The 120-window four-state teacher and both 240-epoch CPU Routers are
complete; do not restart them. Keep both context/local candidates for real-stream
comparison, since their table allocation regret is effectively close. Continue
with one body, optional G-boundary reduction, true E packet prefixes, a shared
receiver-derived G policy, and fresh whole-video evaluation. Preserve all four
B/E/G/EG states and the completed joint E/G weights; UF remains frozen.

New `routervc_*.py` files do not alter historical source pins. RTVC v1 wraps
unchanged ACSE2 packets in a charged 310-byte strategy/model header; it transmits
no per-region G map. The receiver re-predicts G from actual mixed B/Y/received
coverage. Sender-side isolated-table predictions are only approximate planning,
not a mixed-video oracle. Distinguish independently optimized E budgets from
the fixed greedy packet ordering used for literal byte prefixes. Boundary
penalties change G choices but do not merge generator calls in this profile;
fewer connected components alone is not a measured speedup.

Use `bash demo/run_routervc.sh test`, then tmux `smoke`, `run`, or `verify`.
Outputs: `/root/autodl-fs/DCVC/runs/routervc_20261003` (smoke adds `_smoke`).
Record actual candidate preparation costs separately from cached evaluation.
Initially validate 17/33 frames and rectangular inputs; explicit implementation
limits are not universal method limits. Do not silently resize or pad unsupported
G geometry, claim semantic reliability from LPIPS, or call reused component
training clips independent system tests. Keep single-A800 GPU exclusion,
atomic resume, three-disk/VRAM heartbeats, real bytes and fixed visualizations.
Do not end the session simply because tmux starts: proceed until the requested
workflow is verified, a substantive user decision is needed, or handoff is asked.
Research records and progress belong only in Notion:
https://app.notion.com/p/3ee8b22ebd8d8112aa83dfe749002a02.

Completion handoff: this research workflow, the baseline supplement, partial-tail
recovery checks, CPU audit and native-UF report are now complete. Inspect
`workflow/complete.json`, `audit.json`, `supplement/complete.json`, and
`supplement/native_uf_report/complete.json` before resuming; use the read-only
verification commands instead of restarting training or completed evaluation.
The public source-frame entrypoint is `bash demo/run_routervc_codec.sh` inside
tmux. Its bounded geometry and preparation resume granularity are in README;
literal byte prefixes are not an online incremental-state decoder.
Preserve all completed source pins. Do not auto-promote context or boundary
penalties, or mistake 16 separate G ROIs for the one-ROI full-frame baseline.
The full-frame companion is CPU-only `python -m demo.routervc_fullframe_preview`
and writes only `supplement/preview_fullframe/` by default. Final results,
visuals and the methodological choices to discuss next are maintained at:
https://app.notion.com/p/3ee8b22ebd8d81578d66c6e03b5070c9.

### Completed four-state Router preparation (2026-10-02)

The user accepted the post-03.15 recommendation: joint E/G is the main candidate,
new E plus control G remains an alternative. Proceed with a true Base/E/G/EG
utility-and-cost table, then the single-backbone Router and later G-boundary
reduction. This supersedes the older pending-discussion/no-Router handoff below;
do not restore the two experts. Do not retrain completed UF/E/G models here.

Use `bash demo/run_four_state.sh smoke`, then `run`, inside tmux. Formal output:
`/root/autodl-fs/DCVC/runs/a800_four_state_20261002`; smoke adds `_smoke`.
Reuse 90 REDS + 30 UVG component-training windows, not independent evidence.
Measure all four states on 16 native 128px cells per 17-frame 512px window.
E is real q1 I+P8+P8 packet bundles; G keeps RGB-first cropping, 64px context,
16px feather and completed joint LoRA. Shared base/G overhead must be counted
once; explicit G controls remain charged. A receiver-derived G policy is not
implemented by this teacher. Do not describe an isolated tile montage as a
decoded mixed stream: neighboring E can alter G context, so compositions need
fresh evaluation before treating the table as an exact allocation oracle.

Source-free receiver execution reuses model residency only after fresh-process
pixel/noise equality, including an image-border and an interior cell. Save each
completed region atomically; replay validates saved artifacts without inference.
Keep the global GPU mutex, 30-second resource heartbeats and historical source
pins unchanged. No new downloads, multi-GPU work or arbitrary quality gates.
This queue prepares labels only; Router training/evaluation has separate code.
After the exact CPU resume smoke, `bash demo/run_four_state_router_queue.sh`
may wait in its own tmux session for the complete teacher, then produce the
teacher figures, train paired 240-epoch context/local utility backbones on CPU,
and evaluate grouped-holdout table regret at six byte/ROI-call budgets. Both
models have 6,982 parameters; context enters after local embeddings with equal
parameter count. Inputs contain only B, decoded candidate Y and E coverage,
not source X. Predict direct-E gains and G gains CONDITIONAL on Y (LPIPS primary;
PSNR/temporal auxiliary). Only the target cell is supervised in isolated-E views.
Never add the independent G(B) gain to E and call that E-to-G prediction.
Use the simple sequence-group split only for Router development; these are
component-training clips, not independent codec/generator evidence. Exact table
budget allocation is not an exact full-video oracle. G controls remain charged;
a deployed receiver-derived policy and actual routed RD are not completed by
this CPU pilot. Do not auto-promote a model based only on table regret.
Router output: `/root/autodl-fs/DCVC/runs/a800_four_state_router_20261002`.
Each five epochs is an atomic resume boundary. The CPU queue neither competes
for GPU memory nor modifies the in-flight teacher or historical source pins.
Research plan, progress and results live only in Notion 03.16:
https://app.notion.com/p/3ed8b22ebd8d81a8915ec00064643818.

### Completed online E/G joint adaptation (2026-10-02)

The user approved joint E/G training while retaining the completed RGB-first
ROI alignment. Use the separate `bash demo/run_online_eg.sh smoke`, then `train`,
inside tmux. Continue both paired 3000-step arms from the completed 03.14 RGB
LoRA and A/pad16 E model: `joint` updates E plus G LoRA; `fixed` updates only G
LoRA. Preserve all historical source pins and outputs. Freeze UF, its rendering
head, original DiT and VAE weights; do not start Router training or add another
feature interface. Preserve Base/E-only/G-only/E-to-G modes.

The online path must actually carry the generated-image loss through the VAE
encoder to E; detached offline E images are not joint training. Keep RGB-first
256px processing / 128px core / 64px halo, actual packet geometry and rounded
receiver RGB. Verify the gradient from LPIPS alone to E analysis and synthesis,
receiver input equality, exact interrupted/resumed E/G/optimizer states, real
entropy streams, source-free fresh decode, repeats and model-free G-off first.
Training entropy estimates are not actual stream sizes. The updated E changes
packet lengths, so later comparisons must use actual-byte RD, not equal-q claims.

The latest user explicitly requests handoff once formal training is running:
confirm steps advance and an atomic checkpoint loads, update Notion, then end
the session. Do not wait for the training to finish or auto-launch evaluation.
The queue serializes joint then fixed, saves every25 steps and reports resources
every30 seconds; same command resumes after restarting tmux. This supersedes
older statements below that required E frozen or automatic continued monitoring.
Recipe, progress and next-session evaluation live only at:
https://app.notion.com/p/3ed8b22ebd8d8158be1dd92e3d59b99f.

Both online E/G arms have completed 3000 steps and the user has returned to
continue. Do not restart training. Use the separate, tmux-only
`bash demo/run_online_eg_evaluate.sh run` for the real-byte evaluation: 40 E-only,
40 generated points, eight full-q1 crossed E/G controls and four receiver/repeat
checks (92 fresh decodes). Re-encode the original packet geometry/order at
q0.5/1/2, preserving true q1 prefixes. Joint E may change byte counts and RGB
conditions; pair diffusion noise without incorrectly requiring equal conditions.
Keep the completed 03.14 RGB points and native UF QP8/32 as checked historical
references, not new decodes or a dense UF curve. Report local and whole-frame
metrics, E-only utility, actual-byte curves and fixed visuals. No new training,
Router, model promotion or downloads. Preserve cbda3b4 training/receiver pins.
Completed evaluation points must validate without inference/metric recomputation;
`evaluate` and CPU-only `report` are separate modes. Retain resource heartbeats
and the global GPU mutex. Training handoff above belongs to the prior session;
this continuation proceeds through evaluation and discussion of its results.
After the evaluation/report queue finishes, `bash demo/run_online_eg_analysis.sh`
replays all completed points with inference and metric calculation forbidden,
and verifies that original result JSON and elapsed times remain unchanged.
Its CPU-only tmux task may wait behind the same GPU mutex; do not edit pinned
evaluation or training modules while the formal queue is running.
Once the completed-queue audit passes, `demo/online_eg_cross_report.py` produces
CPU-only crossed-model visuals and a generated-output RD zoom from the saved
points. Keep this supplement separate from the pinned report. Crossed E/G
combinations have only full-q1 measurements, not full matched-rate curves.

The 92-point evaluation, report and completed-queue replay are now finished.
`run.complete.json`, `evaluation/audit.json`, and `evaluation.resume_audit.json`
all report completion; the replay preserves 241 JSON files and original timings.
The separate crossed-model report and 34 CPU regressions also pass. Results and
19 figures/previews are in Notion 03.15.1 and its two child galleries:
https://app.notion.com/p/3ed8b22ebd8d81a5a9b3de25abd5830a.
No task remains running. Discuss the next candidate E/G pairing and four-state
Router data preparation with the user before starting new experiments/training;
neither a new Router queue nor model promotion is authorized by this handoff.

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
