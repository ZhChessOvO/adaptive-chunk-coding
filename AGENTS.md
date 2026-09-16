# Project instructions for coding agents

Before changing code or starting an experiment, read these files in order:

1. `README.md`
2. `docs/CLOUD_STORAGE_AND_UPLOAD.md`
3. `docs/CLOUD_A800_PILOT.md`

The current research line is budget-conditioned regional Generate / Base /
Enhance routing around DCVC-UF. It is not the historical latent-prediction
line. Preserve all E01-E18 and latent-predictor code as history, but do not
make it a prerequisite for the current work.

The authorized cloud scope is the single-card A800 80GB pilot described in
`docs/CLOUD_A800_PILOT.md`: restore the environment, reproduce a small smoke
test, generate a bounded set of counterfactual teacher labels with frozen
DCVC-UF and SeedVR2, and train/evaluate a lightweight controller. Do not start
multi-GPU production label generation, SeedVR2 fine-tuning, or spatial-codec
fine-tuning without a new user decision.

Respect the data ledger exactly:

- Training may use REDS `train/000..239`.
- REDS `val/000..005` is development data, never an independent test set.
- Do not read `val/012..023`.
- Keep `val/024..029` sealed.
- Jockey and other previously used clips are regression material only.

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

On the current cloud host, keep datasets, model weights, and all formal run
outputs under `/root/autodl-fs/DCVC`. Keep the repository, conda environment,
compiled extensions, and at most 8GB of per-sample scratch on the 50GB data
disk. Do not fill the 30GB system disk or copy the full dataset to the data
disk. Prefer the uploaded assets over downloading another copy. Follow
`docs/CLOUD_STORAGE_AND_UPLOAD.md` for the exact layout and limits.
