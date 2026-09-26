#!/usr/bin/env bash
# Invoke inside tmux. Every stage is restartable; no external downloads.
set -Eeuo pipefail
repo=/root/autodl-tmp/adaptive-chunk-coding
root=/root/autodl-fs/DCVC/runs/a800_patch_efficiency_20260926
py=/root/autodl-tmp/DCVC/envs/dcvcuf/bin/python
cd "$repo"
mkdir -p "$root"
bash demo/run_chunk_enhancement.sh test > "$root/tests.log" 2>&1
if [[ ${1:-diagnostic} != train ]]; then
bash demo/run_chunk_enhancement.sh efficiency prepare --output "$root" > "$root/prepare.log" 2>&1
bash demo/run_chunk_enhancement.sh efficiency audit --output "$root/compact" > "$root/compact.log" 2>&1
bash demo/run_chunk_enhancement.sh evaluate --compact --checkpoint "$root/pad16_initial.pt" \
  --output "$root/padding_evaluation" > "$root/padding_evaluation.log" 2>&1
fi
if [[ ${1:-diagnostic} == diagnostic ]]; then
  exit 0
fi
"$py" -c 'import json,sys; [json.load(open(p)) for p in sys.argv[1:]]' \
  "$root/compact/audit.complete" "$root/padding_evaluation/evaluate.complete"
# Same crop/q/sample schedule and initialization; only the distortion weight
# differs. The 4x branch controls for additional training and mixed geometry.
common=(--architecture uf_head_pad16 --initialize "$root/pad16_initial.pt" \
  --mixed-rectangles --learning-rate 0.00005 --save-every 500 --max-hours 12 \
  --feature-manifest /root/autodl-fs/DCVC/runs/a800_chunk_enhancement_20260926/features.json)
for control in split continuous; do
  if [[ "$control" == split ]]; then
    bash demo/run_chunk_enhancement.sh train "${common[@]}" --lambda-scale 2 --limit 1 \
      --steps 50 --output "$root/resume_$control" > "$root/resume_$control.log" 2>&1
  fi
  bash demo/run_chunk_enhancement.sh train "${common[@]}" --lambda-scale 2 --limit 1 \
    --steps 100 --output "$root/resume_$control" >> "$root/resume_$control.log" 2>&1
done
"$py" demo/audit_chunk_resume.py "$root/resume_split/resume.pt" \
  "$root/resume_continuous/resume.pt" "$root/resume_audit.json" > "$root/resume_audit.log" 2>&1
train_one() {
  local weight=$1
  bash demo/run_chunk_enhancement.sh train "${common[@]}" --lambda-scale "$weight" \
    --steps 20000 --output "$root/train_l$weight" >> "$root/train_l$weight.log" 2>&1
  bash demo/run_chunk_enhancement.sh evaluate --compact --checkpoint "$root/train_l$weight/final.pt" \
    --output "$root/evaluation_l$weight" >> "$root/evaluation_l$weight.log" 2>&1
}
# Independent paired recipes share GPU 0, not multiple GPUs. Each has its own
# optimizer, RNG, checkpoints and heartbeat; host RAM is well within capacity.
train_one 4 &
control_pid=$!
train_one 2 &
rate_pid=$!
status=0
wait "$control_pid" || status=1
wait "$rate_pid" || status=1
if [[ "$status" != 0 ]]; then exit "$status"; fi
"$py" demo/patch_efficiency_report.py --root "$root" > "$root/report.log" 2>&1
bash demo/run_chunk_enhancement.sh efficiency-timing --root "$root" > "$root/timing.log" 2>&1
"$py" -c 'from pathlib import Path; from demo.scalable_codec import atomic_json; from demo.scalable_experiment import resources; import sys; atomic_json(Path(sys.argv[1])/"pipeline.complete", resources())' "$root"
