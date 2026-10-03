#!/usr/bin/env bash
set -Eeuo pipefail
repo=/root/autodl-tmp/adaptive-chunk-coding
env_root=/root/autodl-tmp/DCVC/envs/dcvcuf
export TMPDIR=/root/autodl-tmp/DCVC/tmp TORCH_HOME=/root/autodl-fs/DCVC/cache/torch
export PATH="$env_root/bin:$PATH" PYTHONPATH="$repo:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$repo"
if [[ "${1:-}" == test ]]; then exec python -m unittest demo.test_routervc_analysis -v; fi
if [[ -z "${TMUX:-}" ]]; then
    echo 'Run RouterVC saved-result analysis inside tmux (CPU only).' >&2
    exit 2
fi
exec python demo/routervc_analysis.py "$@"
