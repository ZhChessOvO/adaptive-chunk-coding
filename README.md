<div align="center">

# Adaptive Chunk Coding

**Bandwidth-compute scalable selective latent transmission for chunk video compression**

</div>

## Overview

Adaptive Chunk Coding studies how independently entropy-coded latent blocks can switch between
real transmission and decoder-side prediction according to bandwidth and decoder-compute budgets.
The goal is to provide controllable rate-distortion-compute operating points within one
high-throughput chunk codec.

The implementation is built on
[DCVC-UF](https://github.com/microsoft/DCVC) and currently includes:

- real latent-symbol skipping with an explicit route stream;
- independently readable single- and multi-chunk research containers;
- routed reconstruction propagated across consecutive chunks;
- Oracle routing and matched-rate-distortion analysis;
- routing-granularity comparisons for HT-S latent blocks.

## Research scripts

- `demo/stage0_selective_demo.py`: historical pixel-space three-path proxy.
- `demo/stage05_rate_substitution.py`: historical pixel-space rate-substitution proxy.
- `demo/stage1_token_skipping.py`: true HT-S latent skipping for one P chunk.
- `demo/stage1_multichunk_oracle.py`: multi-chunk bitstream and Oracle experiments.
- `demo/analyze_stage1_oracle_rd.py`: matched-rate-distortion analysis.

## Local assets

Datasets, model checkpoints, generated outputs, experimental bitstreams, CUTLASS build sources,
and local research notes are intentionally excluded from version control.

This repository is research code under active development.
