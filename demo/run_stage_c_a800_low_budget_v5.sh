#!/usr/bin/env bash
set -Eeuo pipefail

repo_root=/root/autodl-tmp/adaptive-chunk-coding
baseline_root=/root/autodl-fs/DCVC/runs/a800_pilot_20260917
v4_root=/root/autodl-fs/DCVC/runs/a800_low_budget_v4_20260918
run_root=/root/autodl-fs/DCVC/runs/a800_low_budget_v5_20260918
python_bin=/root/autodl-tmp/DCVC/envs/dcvcuf/bin/python

export PATH=/usr/local/cuda/bin:/root/autodl-tmp/DCVC/envs/dcvcuf/bin:$PATH
export PYTHONPATH=$repo_root:/root/autodl-tmp/DCVC/DCVC:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/root/autodl-tmp/DCVC/envs/dcvcuf/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export TORCH_HOME=/root/autodl-fs/DCVC/cache/torch
export HF_HOME=/root/autodl-fs/DCVC/cache/huggingface
export CUDA_VISIBLE_DEVICES=0

mkdir -p "$run_root/logs" "$run_root/selection" "$run_root/controller" \
  "$run_root/routes"
cd "$repo_root"
exec > >(tee -a "$run_root/logs/run.log") 2>&1

heartbeat_pid=
gpu_monitor_pid=
cleanup() {
  if [[ -n "$heartbeat_pid" ]]; then kill "$heartbeat_pid" 2>/dev/null || true; fi
  if [[ -n "$gpu_monitor_pid" ]]; then kill "$gpu_monitor_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

(
  while true; do
    printf 'HEARTBEAT low_budget_v5 utc=%s ' "$(date -u +%FT%TZ)"
    df -h /root /root/autodl-tmp /root/autodl-fs | tail -n +2 | tr '\n' ';'
    printf '\n'
    sleep 60
  done
) >> "$run_root/logs/heartbeat.log" 2>&1 &
heartbeat_pid=$!

(
  while true; do
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw \
      --format=csv,noheader,nounits || true
    sleep 5
  done
) >> "$run_root/logs/gpu_samples.csv" 2>&1 &
gpu_monitor_pid=$!

cp "$0" "$run_root/logs/executed_run_stage_c_a800_low_budget_v5.sh"

echo "START low_budget_v5 utc=$(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version \
  --format=csv,noheader,nounits
df -h /root /root/autodl-tmp /root/autodl-fs
git status --short
git rev-parse HEAD

if [[ ! -s "$run_root/selection/selection_summary.json" ]]; then
  "$python_bin" demo/stage_c_a800_low_budget_v5.py select \
    --oof-predictions "$v4_root/selection/oof_predictions.npz" \
    --train-teacher-manifest \
      "$baseline_root/pilot/teacher_train/manifest.json" \
    --train-roi-cost-manifest \
      "$baseline_root/pilot/roi_cost_train/manifest.json" \
    --output-dir "$run_root/selection"
fi

"$python_bin" - "$run_root/selection/selection_summary.json" <<'PY'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(json.dumps({
    "hard_promotion_gate": value["selection_policy"]["hard_promotion_gate"],
    "v1_best_candidate": value["v1_best_candidate"],
    "selected_candidate": value["selected_candidate"],
}, ensure_ascii=False, indent=2))
PY

if [[ ! -s "$run_root/controller/controller_v5.pt" ]]; then
  "$python_bin" demo/stage_c_a800_low_budget_v5.py train \
    --train-teacher-manifest \
      "$baseline_root/pilot/teacher_train/manifest.json" \
    --train-roi-cost-manifest \
      "$baseline_root/pilot/roi_cost_train/manifest.json" \
    --selection-summary "$run_root/selection/selection_summary.json" \
    --v1-checkpoint "$baseline_root/pilot/controller/controller.pt" \
    --output-dir "$run_root/controller"
fi

if [[ ! -s "$run_root/routes/manifest.json" ]]; then
  "$python_bin" demo/stage_c_a800_low_budget_v5.py route \
    --sample-manifest \
      "$baseline_root/pilot/sample_ledger/development_samples.jsonl" \
    --base-probe-root "$baseline_root/formal/development" \
    --checkpoint "$run_root/controller/controller_v5.pt" \
    --output-dir "$run_root/routes"
fi

"$python_bin" - "$run_root/routes/manifest.json" \
  "$baseline_root/pilot/routes/mlp/manifest.json" \
  "$v4_root/routes/manifest.json" \
  "$run_root/route_comparison.json" <<'PY'
import json
import os
import sys
from pathlib import Path

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

v5_manifest = read(sys.argv[1])
v1_manifest = read(sys.argv[2])
v4_manifest = read(sys.argv[3])

def route_map(manifest):
    output = {}
    for entry in manifest["entries"]:
        route = read(entry["path"])
        name = route["selected_variant"]
        output[entry["sample_id"]] = route["variants"][name]["actions"]
    return output

v5 = route_map(v5_manifest)
v1 = route_map(v1_manifest)
v4 = route_map(v4_manifest)
rows = []
for sample_id in sorted(v5):
    rows.append({
        "sample_id": sample_id,
        "v5_changed_tiles_vs_v1": sum(a != b for a, b in zip(v5[sample_id], v1[sample_id])),
        "v5_changed_tiles_vs_v4": sum(a != b for a, b in zip(v5[sample_id], v4[sample_id])),
    })
result = {
    "experiment": "v5 development route comparison before quality evaluation",
    "sample_count": len(rows),
    "per_sample": rows,
    "changed_tiles_vs_v1_total": sum(row["v5_changed_tiles_vs_v1"] for row in rows),
    "changed_tiles_vs_v4_total": sum(row["v5_changed_tiles_vs_v4"] for row in rows),
    "quality_metrics_used": False,
}
target = Path(sys.argv[4])
temporary = target.with_suffix(".json.tmp")
temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, target)
print(json.dumps(result, indent=2))
PY

touch "$run_root/TRAINING_AND_ROUTES_COMPLETE"
echo "COMPLETE low_budget_v5 utc=$(date -u +%FT%TZ)"
du -sh "$run_root"
df -h /root /root/autodl-tmp /root/autodl-fs
