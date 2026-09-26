#!/usr/bin/env bash
# Run long commands in tmux; each experiment mode supports atomic resume.
set -Eeuo pipefail
repo=/root/autodl-tmp/adaptive-chunk-coding
fast_root=/root/autodl-tmp/DCVC
env_root="$fast_root/envs/dcvcuf"
persist=/root/autodl-fs/DCVC
export TMPDIR="$fast_root/tmp"
export TORCH_HOME="$persist/cache/torch"
export TORCH_EXTENSIONS_DIR="$fast_root/torch_extensions"
export CUDA_HOME="$env_root"
export DCVC_CUDA_INCLUDE="$env_root/lib/python3.12/site-packages/nvidia/cu13/include"
export CPATH="$env_root/lib/python3.12/site-packages/nvidia/cu13/include:$env_root/targets/x86_64-linux/include:${CPATH:-}"
export LIBRARY_PATH="$env_root/lib/python3.12/site-packages/nvidia/cu13/lib:$env_root/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export PATH="$env_root/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="$env_root/lib/python3.12/site-packages/nvidia/cu13/lib:$env_root/targets/x86_64-linux/lib:$env_root/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$repo:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export MAX_JOBS=4
cd "$repo"
if [[ ${1:-} == build ]]; then
  cd src/layers/extensions/inference
  # Existing dependencies only; no index access or duplicate downloads.
  exec python -m pip install --no-deps --no-index --no-build-isolation .
fi
if [[ ${1:-} == test ]]; then
  exec python -m unittest demo.test_chunk_enhancement demo.test_scalable_format demo.test_patch_efficiency -v
fi
if [[ ${1:-} == efficiency ]]; then
  shift
  exec python demo/patch_efficiency_experiment.py "$@"
fi
if [[ ${1:-} == efficiency-timing ]]; then
  shift
  exec python demo/patch_efficiency_timing.py "$@"
fi
if [[ ${1:-} == prefix-probe ]]; then
  shift
  exec python demo/patch_prefix_probe.py "$@"
fi
if [[ ${1:-} == generate-test ]]; then
  exec python -m unittest demo.test_scalable_generation demo.test_chunk_enhancement demo.test_scalable_format demo.test_patch_efficiency demo.test_patch_prefix_probe -v
fi
if [[ ${1:-} == generate-probe ]]; then
  shift
  # All model files are already local. Prevent accidental network lookups.
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  exec python demo/scalable_generation_experiment.py "$@"
fi
if [[ ${1:-} == generate-audit ]]; then
  shift
  exec python demo/audit_scalable_generation.py "$@"
fi
if [[ ${1:-} == evaluate ]]; then
  shift
  exec python demo/chunk_enhancement_evaluate.py "$@"
fi
exec python demo/chunk_enhancement_experiment.py "$@"
