# RouterVC

Research implementation of **scalable latent refinement, asymmetric dual routing,
and local generative reconstruction** on frozen DCVC-UF and SeedVR2.

Scientific description, results and decisions live in Notion:

- [Project](https://app.notion.com/p/3d58b22ebd8d815483aad4e1471ee933)
- [00 Method](https://app.notion.com/p/3e78b22ebd8d81828124c50e8e74c2ca)
- [02 Stream/interfaces](https://app.notion.com/p/3e78b22ebd8d8181b12fec5580d401a6)
- [04 RD, images and resources](https://app.notion.com/p/3f18b22ebd8d81fabf9fc4ccdb4cf5ae)
- [90.1 Migration](https://app.notion.com/p/3fb8b22ebd8d81e1863af3ed5ba328de)
- [91 New-server handoff prompt](https://app.notion.com/p/3fb8b22ebd8d81f4a39ec0a786599bbc)

This README contains operational instructions, not a duplicate experiment diary.

## Layout

```text
encode.py / decode.py    Sending and independent receiving
prepare.py / train.py    Offline labels, independent routers and G LoRA training
evaluate.py             Actual-bitstream metrics and native UF comparison
smoke_test.py            CPU regressions and fresh-process GPU codec check
routervc/
  codec/                Latent splitting, rANS, regional packets, B reference chain
  routers/              Distinct sender/receiver architectures and budget selection
  generation/           SeedVR2 runner, LoRA and native-scale regional generation
  fusion/               Core composition and fixed multiband boundary treatment
  data/                 Frame manifests, compact labels and teacher preparation
  training/             Router optimization, G objectives and resumable checkpoints
  utils/                Atomic I/O, precision controls and resource checks
src/                    DCVC-UF models and C++/CUDA extensions
scripts/                Build and asset verification helpers
tests/                  Current release regressions
```

Disposable trials, earlier methods and duplicated `docs/` are removed from main.
Historical source remains available at Git tag `archive/pre-migration-20261016`
(commit `246b9aa`). Do not replay its old queued experiment commands by default.

## Restore and install

Use Linux, an NVIDIA CUDA GPU, Python 3.12, a CUDA toolkit matching PyTorch,
a C++ compiler, CMake and Ninja. The validated server used a single A800 80GB,
PyTorch 2.13.0+cu130 and torchvision 0.28.0+cu130. New environments must be tested;
we do not claim cross-version/cross-GPU bit identity.

Keep environment/builds on the fast data disk. Run long work in tmux:

```bash
tmux new -s routervc
conda create -p /root/autodl-tmp/envs/routervc python=3.12 -y
conda activate /root/autodl-tmp/envs/routervc
unset PIP_INDEX_URL PIP_EXTRA_INDEX_URL HF_ENDPOINT
export PIP_CONFIG_FILE=/dev/null
python -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-research.txt --index-url https://pypi.org/simple
```

Install a matching CUDA toolkit if `nvcc` is missing, and set `CUDA_HOME` to it.
The migration package records the original environment versions but deliberately
does not contain an entire conda environment. Restore the bundled SeedVR2 and
CUTLASS sources to `third_party/SeedVR2` and `third_party/cutlass`. The source
identities and licenses are included in that package; do not redownload them.

```bash
export MAX_JOBS=1
# When using NVIDIA's pip runtime plus conda nvcc, if needed:
export DCVC_CUDA_INCLUDE="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"]+"/nvidia/cu13/include")')"
bash scripts/build_extensions.sh
python scripts/verify_assets.py /path/to/RouterVC/assets/assets.json
bash scripts/run.sh smoke_test.py --gpu --assets /path/to/RouterVC/assets/assets.json
```

`scripts/run.sh` supplies the selected Python environment's executables/libraries,
thread limits and deterministic BLAS settings. If the environment is not active,
set `ROUTERVC_PYTHON=/absolute/path/to/python`. Set `TORCH_EXTENSIONS_DIR`, `TMPDIR`
and `TORCH_HOME` to writable data-disk locations as appropriate. Restore the
bundled LPIPS AlexNet cache to `TORCH_HOME` to avoid an evaluation-time download.

## Shared assets

`--assets` points to a JSON manifest. Paths are relative to that file and SHA256
is checked. Required entries are `uf_i`, `uf_p`, `dit`, `vae`, `positive`, `negative`,
`sender`, `receiver`, `g_formal`, and `g_candidate800`; `seedvr2_source` points to
the upstream source directory. The migration package supplies this manifest.

Formal models are P2b source R_s epoch 8, P2 R_g epoch 108 and the formal LoRA.
`candidate800` is an explicitly selected G-repair candidate, **not** a silently
promoted replacement. Both retain the uploaded BF16 DiT, not FP32 DiT.

## Encode, receive, evaluate

Inputs are alphabetically ordered RGB PNG/JPEG frames. No implicit resize/crop is
performed. Current dual-router windows are exactly 17 frames. Geometry must obey
UF and G alignment; previously evaluated examples are 1024x576 REDS full views
and 512x512 UVG crops. Unsupported G geometry fails rather than silently scaling.

```bash
bash scripts/run.sh encode.py --input /data/frames --assets /path/to/assets.json --width 3 --e-ratio 0.5 --output /runs/sample.rvc
bash scripts/run.sh decode.py --input /runs/sample.rvc --assets /path/to/assets.json --output /runs/decoded
bash scripts/run.sh evaluate.py --source /data/frames --decoded /runs/decoded --stream /runs/sample.rvc --output /runs/metrics.json

# Native, unmodified-reference DCVC-UF comparator:
bash scripts/run.sh evaluate.py --source /data/frames --assets /path/to/assets.json --uf-qp 16 --output /runs/uf16.json
```

Use `--g-variant candidate800` on **both** encoding and decoding to select the
candidate. Decode with `--disable-generation` to output received Y without loading
R_g/G; `--e-ratio 0` with G disabled produces B. `--all-enhancements --e-ratio 1`
is an E endpoint diagnostic, not the learned sender's normal stopping policy.

Width is one of 3/5/7/9. E ratio is a cap on **actual full E bundle bytes**, not
area. R_s may stop early. A 121-byte profile header plus all packet metadata are
charged; no extra action/protection mask is transmitted. R_g uses no original
frames or sender weights. The compact calibrated width selector is exposed in
`routervc.routers.budget`; it selects measured candidate prefixes under a total
byte budget, not unseen evaluation quality. Its calibration table is in assets.

The cleaned release has a distinct `RVC20261` profile. It does not claim to read
historical profile streams. Use the archived Git version for those; never bypass
hash checks. Native binary identities may change on recompilation, requiring new
encodings even with the same model weights.

## Training and data

No training is started automatically. Final compact receiver labels are included in
the migration package (derived low-resolution inputs/targets, **not datasets**).
The sender's old FP32 input cache was already deleted in an earlier cleanup;
its measured target records and split/provenance are preserved, but its training
inputs must be prepared again after restoring data. A new G variant
changes final marginal utilities: prepare new labels before claiming adapted
routers. Do not train on the old labels and claim they describe the new G.

```bash
bash scripts/run.sh train.py receiver --assets /path/to/assets.json --manifest /path/to/training/receiver/labels.json --output /runs/new_rg
bash scripts/run.sh train.py sender --assets /path/to/assets.json --manifest /runs/rs_labels/labels.json --output /runs/new_rs
```

These commands **continue from the retained final routers** with fresh AdamW.
They do not recreate the entire historical training lineage from deleted older
initializations. Receiver and sender architectures/losses remain independent.
`resume.pt` saves optimizer, RNG and cursor; reusing an output directory resumes
the same bound recipe. `--stop-after` is a restart-test tool, not early stopping.

To measure new labels, supply a JSON object with `rows`; each row has
`sample_id`, `dataset` (`REDS` or `UVG`), `sequence`, `router_split` (`train` or
`validation`), `files` (17 paths relative to the manifest), per-file `hashes`,
and `shape` (`[17,H,W,3]`). Optional `E_cap`/`sender_parent_cap` controls teacher
states. Do not split one sequence across training and validation.

```bash
bash scripts/run.sh prepare.py receiver --manifest /data/views.json --assets /path/to/assets.json --output /runs/rg_labels
bash scripts/run.sh prepare.py sender --manifest /data/views.json --assets /path/to/assets.json --output /runs/rs_labels
bash scripts/run.sh prepare.py generator --manifest /data/views.json --assets /path/to/assets.json --g-variant candidate800 --output /runs/g_inputs
bash scripts/run.sh train.py generator --manifest /runs/g_inputs/protocol.json --assets /path/to/assets.json --output /runs/g_continuation
```

Preparation restarts at completed windows. G preparation regenerates actual Y;
no original pixels are substituted for received conditions. Mixed G training
uses 75% 512px inputs / 25% 256px inputs, without enlarging small crops, with
corresponding 384px/128px supervised interiors. All source views need enough
native pixels. DiT/VAE/UF are frozen; only the selected LoRA tensors train.
The previous A800 G recipe used about 30.4GiB allocated VRAM but up to 106GiB
container memory with CPU activation offload. Check RAM before rerunning it.

Datasets are intentionally not in the migration archive. The saved source and
split ledgers describe the existing resized REDS views and cropped UVG samples.
Use the user's local dataset copies and inspect these ledgers; do not claim the
UVG crops are full-frame data or silently replace them with different crops.

## Limits and licensing

This is a research prototype, not a real-time network player. Full-system long
videos, deployment on home GPUs and semantic protection remain unvalidated.
The reported LPIPS improvements do not establish exact detail fidelity, and G
still has periodic-texture failure cases. See Notion for numerical evidence.

Keep `LICENSE.txt`, `NOTICE.txt` and upstream third-party licenses with any
redistribution. This repository derives its UF implementation from Microsoft
DCVC; SeedVR2, CUTLASS and model weights retain their own terms. Do not upload
model weights, datasets, bitstreams or generated media to Git.
