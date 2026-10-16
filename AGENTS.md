# RouterVC agent instructions

## Read first

Read this file and `README.md`, then the current Notion pages linked there:
00 (method), 02 (interfaces), 04 (results), 90.1 (migration), and 91 (handoff).
Notion is the primary scientific record. Do not duplicate experiment reports in
`docs/` or append chronological experiment diaries to this file.

## Current implementation

- Frozen DCVC-UF I32/P48 integer latents, coarse/fine widths 3/5/7/9.
- Only B advances the temporal reference. Regional E packets refine display.
- Independent, asymmetric sender R_s and receiver R_g; no source at receiver.
- No transmitted E/G/protection masks. Charge all necessary framing and addresses.
- SeedVR2 BF16 DiT/VAE + explicit LoRA; posterior-mean BF16 condition; local G.
- Fixed multiband fusion, not the retired learned fusion or feature-delta branch.
- Full pipeline currently requires 17 frames. B/E-only has longer-chain evidence;
  this does not establish arbitrary-length full-system support.
- Formal G and `candidate800` are separate assets. The candidate still has residual
  periodic texture; never silently replace formal G or relabel historical RD.
- Semantic protection supervision has not been trained. UVG evidence is cropped,
  not full-frame. Do not claim either missing capability as implemented.

## Working rules

- Use root Python entrypoints; put reusable code under functional `routervc/`
  packages. No new `demo/` dumping ground or one-off experiment code on main.
- Long preparation, compilation, training, evaluation and packaging run in tmux.
  Save resumable units/optimizer state, progress, time, GPU/CPU memory and disks.
- One GPU by default. Do not start multi-GPU, extra training, expensive downloads
  or promote a model merely because a historical page says "next".
- Preserve datasets. Never delete a directory solely because it lives in `runs/`:
  some historical runs contain derived training views. Audit exact paths first.
- Keep repository, environment/build/cache on fast local disk; formal results and
  migration archives may use file storage. Never put large assets in Git.
- Models are located through `assets.json` / `ROUTERVC_ASSETS`, with SHA256 checks.
  Do not hard-code the previous server's run directories in runtime code.
- For downloads, clear mirror overrides for that command and use official sources.
  Ask the user before large dataset downloads; they can supply local copies.
- Verify real saved bytes, fresh-process decoding, B-only reference invariance,
  generation-off fallback, current model identities and fixed visual outputs.
- Code hashes are part of release identities. Do not disable identity checking to
  accept old checkpoints/streams. Metadata re-export requires tensor equality and
  recorded provenance. Historical source is at `archive/pre-migration-20261016`.
- Do not restore deleted experiment queues or archived checkpoints by default.

## Tests

Use `bash scripts/run.sh smoke_test.py` for CPU tests and add `--gpu --assets ...`
for a fresh-process B/E round trip. Run G comparisons with an explicit variant.
Current closure checks and migration receipts are in the migration package,
not a second scientific report in the repository.
