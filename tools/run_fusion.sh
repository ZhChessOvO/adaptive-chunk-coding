#!/usr/bin/env bash
set -euo pipefail
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
action=${1:-tests}
shift || true
if [[ "$action" == tests ]]; then
  python -m unittest tools.test_fusion_boundaries tools.test_fusion_blend tools.test_fusion_model tools.test_fusion_stream tools.test_fusion_metrics tools.test_fusion_finish tools.test_fusion_report "$@"
  exit
fi
[[ -n ${TMUX:-} ]] || { echo 'tmux required' >&2; exit 2; }
case "$action" in
  p0) export CUDA_VISIBLE_DEVICES=''; module=tools.latent_boundary_report ;;
  controls) module=tools.fusion_pilot ;;
  prepare) module=tools.fusion_prepare ;;
  train) module=tools.fusion_train ;;
  visuals) export CUDA_VISIBLE_DEVICES=''; module=tools.fusion_visuals ;;
  review) module=tools.fusion_review ;;
  boundary-quality) export CUDA_VISIBLE_DEVICES=''; module=tools.fusion_boundary_quality ;;
  finish) export CUDA_VISIBLE_DEVICES=''; module=tools.fusion_finish ;;
  report) export CUDA_VISIBLE_DEVICES=''; module=tools.fusion_report ;;
  *) echo "unknown action: $action" >&2; exit 2 ;;
esac
python -m tools.storage_guard --log /root/autodl-tmp/DCVC/tmp/fusion_guard.jsonl --min-inodes 5000 -- python -m "$module" "$@"
