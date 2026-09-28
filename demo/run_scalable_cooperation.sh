#!/usr/bin/env bash
# Run the experiment in tmux. Every point has atomic, checksummed resume.
set -Eeuo pipefail
repo=/root/autodl-tmp/adaptive-chunk-coding
fast_root=/root/autodl-tmp/DCVC
env_root="$fast_root/envs/dcvcuf"
persist=/root/autodl-fs/DCVC
export TMPDIR="$fast_root/tmp" TORCH_HOME="$persist/cache/torch"
export TORCH_EXTENSIONS_DIR="$fast_root/torch_extensions" CUDA_HOME="$env_root"
export PATH="$env_root/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="$env_root/lib/python3.12/site-packages/nvidia/cu13/lib:$env_root/targets/x86_64-linux/lib:$env_root/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$repo:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$repo"
case "${1:-probe}" in
  test)
    exec python -m unittest demo.test_scalable_cooperation demo.test_scalable_generation demo.test_chunk_enhancement demo.test_scalable_format demo.test_patch_efficiency demo.test_patch_prefix_probe -v ;;
  report)
    shift
    exec python demo/scalable_cooperation_report.py "$@" ;;
  probe)
    if [[ $# -gt 0 ]]; then shift; fi
    exec python demo/scalable_cooperation_experiment.py "$@" ;;
  *) exit 2 ;;
esac
