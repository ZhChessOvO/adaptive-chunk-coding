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
