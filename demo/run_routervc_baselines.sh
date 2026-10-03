#!/usr/bin/env bash
set -Eeuo pipefail
repo=/root/autodl-tmp/adaptive-chunk-coding
fast_root=/root/autodl-tmp/DCVC
env_root="$fast_root/envs/dcvcuf"
run_root=/root/autodl-fs/DCVC/runs/routervc_20261003
export TMPDIR="$fast_root/tmp" TORCH_HOME=/root/autodl-fs/DCVC/cache/torch
export TORCH_EXTENSIONS_DIR="$fast_root/torch_extensions" CUDA_HOME="$env_root"
export PATH="$env_root/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="$env_root/lib/python3.12/site-packages/nvidia/cu13/lib:$env_root/targets/x86_64-linux/lib:$env_root/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$repo:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$repo"
mode="${1:-}"
if [[ "$mode" == test ]]; then
    export CUDA_VISIBLE_DEVICES=''
    python demo/routervc_baselines.py test
    exec python -m unittest demo.test_routervc_report demo.test_routervc_baselines -v
fi
if [[ -z "${TMUX:-}" ]]; then
    echo 'Run RouterVC baseline supplements inside tmux; the queue owns the GPU mutex.' >&2
    exit 2
fi
if [[ "$mode" == queue ]]; then
    if (( $# != 1 )); then
        echo 'queue uses fixed smoke/formal roots; use run for custom arguments.' >&2
        exit 2
    fi
    # Fail closed: no formal work after a failed smoke or changed source pins.
    python demo/routervc_baselines.py test
    python -m unittest demo.test_routervc_report demo.test_routervc_baselines -v
    python demo/routervc_baselines.py run --root "${run_root}_smoke" \
        --limit-samples 1 --wait-main-complete --max-hours 24
    python demo/routervc_baselines.py verify --root "${run_root}_smoke" \
        --limit-samples 1 --max-hours 4
    exec python demo/routervc_baselines.py run --root "$run_root" \
        --wait-main-complete --require-smoke "${run_root}_smoke/supplement" --max-hours 24
fi
case "$mode" in
    run|verify) exec python demo/routervc_baselines.py "$@" ;;
    *) echo 'Usage: bash demo/run_routervc_baselines.sh test|queue|run|verify [options]' >&2; exit 2 ;;
esac
