#!/usr/bin/env bash
set -euo pipefail

repo_root=/root/autodl-tmp/adaptive-chunk-coding
baseline_root=/root/autodl-fs/DCVC/runs/a800_pilot_20260917
run_root=/root/autodl-fs/DCVC/runs/a800_low_budget_v2_20260918
python_bin=/root/autodl-tmp/DCVC/envs/dcvcuf/bin/python

export PATH=/usr/local/cuda/bin:/root/autodl-tmp/DCVC/envs/dcvcuf/bin:$PATH
export PYTHONPATH=$repo_root:/root/autodl-tmp/DCVC/DCVC:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/root/autodl-tmp/DCVC/envs/dcvcuf/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=0

mkdir -p "$run_root/logs" "$run_root/selection" "$run_root/controller" \
  "$run_root/routes"
cd "$repo_root"

heartbeat_pid=
gpu_monitor_pid=
cleanup() {
  if [[ -n "$heartbeat_pid" ]]; then kill "$heartbeat_pid" 2>/dev/null || true; fi
  if [[ -n "$gpu_monitor_pid" ]]; then kill "$gpu_monitor_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

(
  while true; do
    printf '%s low-budget-v2 alive\n' "$(date --iso-8601=seconds)"
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

cp "$0" "$run_root/logs/executed_run_stage_c_a800_low_budget_v2.sh"

"$python_bin" - "$run_root/start_snapshot.json" <<'PY'
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

def disk(path):
    value = shutil.disk_usage(path)
    return {"total": value.total, "used": value.used, "free": value.free}

out = {
    "time_utc": datetime.now(timezone.utc).isoformat(),
    "gpu": subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
         "--format=csv,noheader,nounits"], text=True).strip(),
    "disks": {path: disk(path) for path in (
        "/root", "/root/autodl-tmp", "/root/autodl-fs")},
    "git_head": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(),
}
Path(sys.argv[1]).write_text(json.dumps(out, indent=2) + "\n")
PY

if [[ ! -s "$run_root/selection/selection_summary.json" ]]; then
  "$python_bin" demo/stage_c_a800_low_budget_v2.py cross-validate \
    --train-teacher-manifest \
      "$baseline_root/pilot/teacher_train/manifest.json" \
    --train-roi-cost-manifest \
      "$baseline_root/pilot/roi_cost_train/manifest.json" \
    --output-dir "$run_root/selection"
fi

if ! "$python_bin" - "$run_root/selection/selection_summary.json" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1]))
gate = value["gate"]
print(json.dumps(gate, ensure_ascii=False, indent=2))
raise SystemExit(0 if gate["pass"] else 3)
PY
then
  touch "$run_root/TRAINING_ONLY_GATE_FAILED"
  exit 0
fi

if [[ ! -s "$run_root/controller/controller_v2.pt" ]]; then
  "$python_bin" demo/stage_c_a800_low_budget_v2.py train \
    --train-teacher-manifest \
      "$baseline_root/pilot/teacher_train/manifest.json" \
    --train-roi-cost-manifest \
      "$baseline_root/pilot/roi_cost_train/manifest.json" \
    --selection-summary "$run_root/selection/selection_summary.json" \
    --output-dir "$run_root/controller"
fi

if [[ ! -s "$run_root/routes/manifest.json" ]]; then
  "$python_bin" demo/stage_c_a800_low_budget_v2.py route \
    --sample-manifest \
      "$baseline_root/pilot/sample_ledger/development_samples.jsonl" \
    --checkpoint "$run_root/controller/controller_v2.pt" \
    --output-dir "$run_root/routes"
fi

"$python_bin" - "$run_root/final_training_snapshot.json" "$run_root" <<'PY'
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

def disk(path):
    value = shutil.disk_usage(path)
    return {"total": value.total, "used": value.used, "free": value.free}

root = Path(sys.argv[2])
files = [path for path in root.rglob("*") if path.is_file()]
gpu_rows = []
for line in (root / "logs" / "gpu_samples.csv").read_text().splitlines():
    fields = [field.strip() for field in line.split(",")]
    if len(fields) >= 3:
        try:
            gpu_rows.append(int(fields[2]))
        except ValueError:
            pass
out = {
    "time_utc": datetime.now(timezone.utc).isoformat(),
    "status": "training-selection-and-routes-complete",
    "ordinary_file_bytes": sum(path.stat().st_size for path in files),
    "ordinary_file_count": len(files),
    "peak_nvidia_smi_memory_mib": max(gpu_rows) if gpu_rows else None,
    "disks": {path: disk(path) for path in (
        "/root", "/root/autodl-tmp", "/root/autodl-fs")},
    "gpu": subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.used,memory.total",
         "--format=csv,noheader,nounits"], text=True).strip(),
}
Path(sys.argv[1]).write_text(json.dumps(out, indent=2) + "\n")
PY

touch "$run_root/TRAINING_AND_ROUTES_COMPLETE"
