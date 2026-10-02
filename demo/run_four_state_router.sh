#!/usr/bin/env bash
set -Eeuo pipefail
repo=/root/autodl-tmp/adaptive-chunk-coding
env_root=/root/autodl-tmp/DCVC/envs/dcvcuf
export PATH="$env_root/bin:$PATH" PYTHONPATH="$repo:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$repo"
if [[ "${1:-}" == test ]]; then
    exec python -m unittest demo.test_four_state_router -v
fi
if [[ -z "${TMUX:-}" ]]; then
    echo 'Run preparation/training inside tmux.' >&2
    exit 2
fi
exec python -m demo.four_state_router "$@"
