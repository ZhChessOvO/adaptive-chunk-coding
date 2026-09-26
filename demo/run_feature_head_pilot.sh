#!/usr/bin/env bash
# Launch inside tmux. Re-running resumes verified training/evaluation outputs.
set -Eeuo pipefail
repo=/root/autodl-tmp/adaptive-chunk-coding
pilot=/root/autodl-fs/DCVC/runs/a800_feature_head_20260926
python=/root/autodl-tmp/DCVC/envs/dcvcuf/bin/python
cd "$repo"
mkdir -p "$pilot"
# Do not launch the formal recipe without its actual decode/resume checks.
"$python" -c 'import json,sys; assert all(json.load(open(p))["passed"] for p in sys.argv[1:])' \
  "$pilot/smoke_verified/smoke.json" "$pilot/resume_audit.json"
bash demo/run_chunk_enhancement.sh train --architecture uf_head \
  --steps 20000 --save-every 500 --max-hours 12 \
  --warmup-steps 2000 --rate-ramp-steps 3000 --initial-gain 8 --lambda-scale 4 \
  --output "$pilot/train" >> "$pilot/train.log" 2>&1
bash demo/run_chunk_enhancement.sh evaluate --checkpoint "$pilot/train/final.pt" \
  --output "$pilot/evaluation" >> "$pilot/evaluation.log" 2>&1
"$python" demo/feature_head_report.py --root "$pilot" > "$pilot/report.log" 2>&1
# The same environment as the coding wrapper; no package installation.
env_root=/root/autodl-tmp/DCVC/envs/dcvcuf
export LD_LIBRARY_PATH="$env_root/lib/python3.12/site-packages/nvidia/cu13/lib:$env_root/targets/x86_64-linux/lib:$env_root/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$repo:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
"$python" demo/feature_head_timing.py --root "$pilot" > "$pilot/timing.log" 2>&1
