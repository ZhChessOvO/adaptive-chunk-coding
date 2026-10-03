#!/usr/bin/env bash
set -Eeuo pipefail
repo=/root/autodl-tmp/adaptive-chunk-coding
fast_root=/root/autodl-tmp/DCVC
env_root="$fast_root/envs/dcvcuf"
export TMPDIR="$fast_root/tmp" TORCH_HOME=/root/autodl-fs/DCVC/cache/torch
export PYTHONPATH="$repo:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$repo"
if [[ "${1:-}" == test ]]; then
    exec "$env_root/bin/python" -m unittest demo.test_routervc_finalize -v
fi
if [[ -z "${TMUX:-}" ]]; then
    echo 'Run the CPU RouterVC completion queue inside tmux.' >&2
    exit 2
fi
exec "$env_root/bin/python" demo/routervc_finalize.py "$@"
