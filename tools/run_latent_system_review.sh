#!/usr/bin/env bash
set -Eeuo pipefail
repo=/root/autodl-tmp/adaptive-chunk-coding
env_root=/root/autodl-tmp/DCVC/envs/dcvcuf
export TMPDIR=/root/autodl-tmp/DCVC/tmp TORCH_HOME=/root/autodl-fs/DCVC/cache/torch
export TORCH_EXTENSIONS_DIR=/root/autodl-tmp/DCVC/torch_extensions CUDA_HOME="$env_root"
export PATH="$env_root/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="$env_root/lib/python3.12/site-packages/nvidia/cu13/lib:$env_root/targets/x86_64-linux/lib:$env_root/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$repo:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 MAX_JOBS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$repo"
[[ -n ${TMUX:-} ]] || { echo 'Run inside tmux' >&2; exit 2; }
exec python -m tools.storage_guard --log /root/autodl-tmp/DCVC/tmp/latent_system_review_guard.jsonl \
    --min-inodes 5000 -- python -m tools.latent_system_review "$@"
