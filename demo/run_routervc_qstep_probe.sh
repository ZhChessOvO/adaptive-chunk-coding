#!/usr/bin/env bash
set -Eeuo pipefail
repo=/root/autodl-tmp/adaptive-chunk-coding
fast_root=/root/autodl-tmp/DCVC
env_root="$fast_root/envs/dcvcuf"
export TMPDIR="$fast_root/tmp" TORCH_HOME=/root/autodl-fs/DCVC/cache/torch
export TORCH_EXTENSIONS_DIR="$fast_root/torch_extensions" CUDA_HOME="$env_root"
export PATH="$env_root/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="$env_root/lib/python3.12/site-packages/nvidia/cu13/lib:$env_root/targets/x86_64-linux/lib:$env_root/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$repo:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$repo"
if [[ "${1:-}" == test ]]; then
    export CUDA_VISIBLE_DEVICES=''
    exec python -m unittest demo.test_routervc_qstep_probe -v
fi
if [[ -z "${TMUX:-}" ]]; then
    echo 'Run the resumable fixed-weight q2 probe in tmux.' >&2
    exit 2
fi
exec python -m demo.routervc_qstep_probe "$@"
