#!/usr/bin/env bash
# Invoke in tmux: bash demo/run_scalable_stage1.sh mechanism|cache [output-root]
set -Eeuo pipefail
repo=/root/autodl-tmp/adaptive-chunk-coding
fast_root=/root/autodl-tmp/DCVC
env_root="$fast_root/envs/dcvcuf"
persist=/root/autodl-fs/DCVC
mode=${1:-mechanism}
if [[ "$mode" != "mechanism" && "$mode" != "cache" ]]; then
  echo "mode must be mechanism or cache" >&2
  exit 2
fi
run_root=${2:-"$persist/runs/a800_scalable_${mode}_20260926"}
plan="$persist/runs/a800_scalable_mechanism_20260926/plan.json"
export TMPDIR="$fast_root/tmp"
export TORCH_HOME="$persist/cache/torch"
export HF_HOME="$persist/cache/huggingface"
export TORCH_EXTENSIONS_DIR="$fast_root/torch_extensions"
export CUDA_HOME="$env_root"
export PATH="$env_root/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="$env_root/lib/python3.12/site-packages/nvidia/cu13/lib:$env_root/targets/x86_64-linux/lib:$env_root/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$repo:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
mkdir -p "$run_root/logs"
cd "$repo"
exec 9>"$run_root/run.lock"
flock -n 9 || { echo "another job owns this run directory" >&2; exit 1; }
exec > >(tee -a "$run_root/logs/run.log") 2>&1
python -m unittest demo.test_scalable_format -v
python demo/scalable_experiment.py plan --plan "$plan"
python demo/scalable_experiment.py run --plan "$plan" --mode "$mode" --output "$run_root" --max-hours 8
