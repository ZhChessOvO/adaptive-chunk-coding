#!/usr/bin/env bash
set -Eeuo pipefail

repo=/root/autodl-tmp/adaptive-chunk-coding
fast_root=/root/autodl-tmp/DCVC
env_root="$fast_root/envs/dcvcuf"
persist=/root/autodl-fs/DCVC
data_root="$persist/data/UVG_adaptation"
baseline_root="$persist/runs/a800_pilot_20260917"
v5_root="$persist/runs/a800_low_budget_v5_20260918"
joint_root="$persist/runs/a800_joint_evaluation_20260918"
run_root="$persist/runs/a800_uvg_adaptation_20260919"
teacher_uvg="$run_root/teacher_uvg"
roi_uvg="$run_root/roi_cost_uvg"
combined_root="$run_root/combined_labels"
controller_root="$run_root/controller_residual_adaptation"
routes_root="$run_root/routes_joint37"
log_root="$run_root/logs"
log_file="$log_root/run_uvg_adaptation.log"
started_marker="$run_root/training_started_epoch.txt"
python_bin="$env_root/bin/python"

export TMPDIR="$fast_root/tmp"
export PIP_CACHE_DIR="$fast_root/cache/pip"
export TORCH_EXTENSIONS_DIR="$fast_root/torch_extensions"
export HF_HOME="$persist/cache/huggingface"
export TORCH_HOME="$persist/cache/torch"
export CUDA_HOME="$env_root"
export PATH="$env_root/bin:/usr/local/cuda/bin:$PATH"
cuda_wheel_root="$env_root/lib/python3.12/site-packages/nvidia/cu13"
export LD_LIBRARY_PATH="$cuda_wheel_root/lib:$env_root/targets/x86_64-linux/lib:$env_root/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$repo:/root/autodl-tmp/DCVC/DCVC:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0

mkdir -p "$log_root" "$teacher_uvg" "$roi_uvg" "$combined_root" \
  "$controller_root" "$routes_root"
if [[ ! -f "$started_marker" ]]; then
  date +%s > "$started_marker.tmp"
  mv "$started_marker.tmp" "$started_marker"
fi
experiment_start_epoch=$(<"$started_marker")
invocation_start_epoch=$(date +%s)
cp "$0" "$log_root/executed_run_stage_c_a800_uvg_adaptation.sh"
exec > >(tee -a "$log_file") 2>&1

heartbeat_pid=
gpu_monitor_pid=
cleanup() {
  if [[ -n "$heartbeat_pid" ]]; then kill "$heartbeat_pid" 2>/dev/null || true; fi
  if [[ -n "$gpu_monitor_pid" ]]; then kill "$gpu_monitor_pid" 2>/dev/null || true; fi
  if [[ -n "$heartbeat_pid" ]]; then wait "$heartbeat_pid" 2>/dev/null || true; fi
  if [[ -n "$gpu_monitor_pid" ]]; then wait "$gpu_monitor_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

(
  while true; do
    now_epoch=$(date +%s)
    quality_count=$(find "$teacher_uvg/samples" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l)
    roi_count=$(find "$roi_uvg/samples" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l)
    printf 'HEARTBEAT uvg_adaptation_train utc=%s experiment_elapsed_s=%s invocation_elapsed_s=%s quality=%s/60 roi=%s/60 ' \
      "$(date -u +%FT%TZ)" "$((now_epoch - experiment_start_epoch))" \
      "$((now_epoch - invocation_start_epoch))" "$quality_count" "$roi_count"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits | tr '\n' ';' || true
    printf ' '
    df -h /root /root/autodl-tmp /root/autodl-fs | tail -n +2 | tr '\n' ';'
    printf '\n'
    sleep 60
  done
) >> "$log_root/training_heartbeat.log" 2>&1 &
heartbeat_pid=$!

(
  while true; do
    nvidia-smi \
      --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw \
      --format=csv,noheader,nounits || true
    sleep 5
  done
) >> "$log_root/training_gpu_samples.csv" 2>&1 &
gpu_monitor_pid=$!

echo "START UVG adaptation labels and residual experts utc=$(date -u +%FT%TZ)"
cd "$repo"
if [[ -n "$(git status --short)" ]]; then
  echo "ERROR repository must be clean before frozen adaptation training" >&2
  git status --short
  exit 1
fi
if [[ ! -s "$data_root/dataset.complete" ]]; then
  echo "ERROR UVG adaptation data is not complete" >&2
  exit 1
fi
if [[ "$(wc -l < "$data_root/uvg_adaptation_samples.jsonl")" -ne 60 ]]; then
  echo "ERROR UVG adaptation manifest must contain 60 samples" >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version \
  --format=csv,noheader,nounits
df -h /root /root/autodl-tmp /root/autodl-fs
git rev-parse HEAD

if [[ ! -s "$run_root/frozen_training_source.json" ]]; then
  "$python_bin" - "$run_root/frozen_training_source.json" \
    "$data_root/uvg_adaptation_samples.jsonl" \
    "$baseline_root/pilot/controller/controller.pt" \
    "$v5_root/selection/selection_summary.json" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

target = Path(sys.argv[1])
value = {
    "experiment": "A800 UVG residual-expert adaptation freeze",
    "status": "frozen_before_labels_and_training",
    "frozen_utc": datetime.now(timezone.utc).isoformat(),
    "git_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(),
    "uvg_sample_manifest": str(Path(sys.argv[2]).resolve()),
    "uvg_sample_manifest_sha256": sha(sys.argv[2]),
    "frozen_reds_v1_checkpoint": str(Path(sys.argv[3]).resolve()),
    "frozen_reds_v1_checkpoint_sha256": sha(sys.argv[3]),
    "frozen_v5_selection": str(Path(sys.argv[4]).resolve()),
    "frozen_v5_selection_sha256": sha(sys.argv[4]),
    "adaptation_scope": (
        "add 60 UVG samples and retrain the two residual experts only; keep "
        "the REDS v1 anchor, v5 selection, budgets, codec and restorer fixed"
    ),
    "uvg_training_sequences": [
        "Beauty", "Bosphorus", "HoneyBee", "Jockey", "ShakeNDry"],
    "uvg_holdout_sequences": ["ReadySetGo", "YachtRide"],
}
temporary = target.with_suffix(".json.tmp")
temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, target)
PY
fi

quality_complete=$(
  "$python_bin" - "$teacher_uvg/manifest.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
try:
    d = json.loads(p.read_text())
    print(int(d.get("complete") is True and d.get("completed_sample_count") == 60))
except Exception:
    print(0)
PY
)
if [[ "$quality_complete" -ne 1 ]]; then
  timeout 4h torchrun --standalone --nproc-per-node=1 \
    demo/stage_c_a800_teacher.py \
    --sample-manifest "$data_root/uvg_adaptation_samples.jsonl" \
    --output-dir "$teacher_uvg" \
    --scratch-dir "$fast_root/tmp/a800_uvg_adaptation_teacher" \
    --limit 60 --visual-count 5 --max-wall-seconds 13800 \
    --dit-checkpoint third_party/SeedVR2/ckpts/seedvr2_ema_3b_bf16.safetensors
fi
"$python_bin" - "$teacher_uvg/manifest.json" <<'PY'
import json, sys
from pathlib import Path
d = json.loads(Path(sys.argv[1]).read_text())
assert d["complete"] and d["completed_sample_count"] == 60
PY

roi_complete=$(
  "$python_bin" - "$roi_uvg/manifest.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
try:
    d = json.loads(p.read_text())
    print(int(d.get("complete") is True and d.get("completed_sample_count") == 60))
except Exception:
    print(0)
PY
)
if [[ "$roi_complete" -ne 1 ]]; then
  timeout 4h torchrun --standalone --nproc-per-node=1 \
    demo/stage_c_a800_roi_cost_teacher.py \
    --teacher-manifest "$teacher_uvg/manifest.json" \
    --output-dir "$roi_uvg" --limit 60 --max-wall-seconds 13800 \
    --dit-checkpoint third_party/SeedVR2/ckpts/seedvr2_ema_3b_bf16.safetensors
fi
"$python_bin" - "$roi_uvg/manifest.json" <<'PY'
import json, sys
from pathlib import Path
d = json.loads(Path(sys.argv[1]).read_text())
assert d["complete"] and d["completed_sample_count"] == 60
PY

if [[ ! -s "$combined_root/merge_summary.json" ]]; then
  "$python_bin" demo/stage_c_a800_uvg_adaptation.py merge-labels \
    --reds-quality "$baseline_root/pilot/teacher_train/manifest.json" \
    --reds-roi-cost "$baseline_root/pilot/roi_cost_train/manifest.json" \
    --uvg-quality "$teacher_uvg/manifest.json" \
    --uvg-roi-cost "$roi_uvg/manifest.json" \
    --v5-selection "$v5_root/selection/selection_summary.json" \
    --output-dir "$combined_root"
fi

if [[ ! -s "$controller_root/controller_v5.pt" ]]; then
  "$python_bin" demo/stage_c_a800_low_budget_v5.py train \
    --train-teacher-manifest "$combined_root/combined_quality_manifest.json" \
    --train-roi-cost-manifest "$combined_root/combined_roi_cost_manifest.json" \
    --selection-summary "$combined_root/fixed_v5_selection.json" \
    --v1-checkpoint "$baseline_root/pilot/controller/controller.pt" \
    --output-dir "$controller_root" --epochs 240 --batch-size 512 \
    --learning-rate 0.001 --weight-decay 0.0001 \
    --base-seed 20260917 --correction-seed 20260919
fi

if [[ ! -s "$routes_root/manifest.json" ]]; then
  "$python_bin" demo/stage_c_a800_low_budget_v5.py route \
    --sample-manifest "$joint_root/manifests/joint_samples.jsonl" \
    --base-probe-root "$joint_root/formal/evaluation" \
    --checkpoint "$controller_root/controller_v5.pt" \
    --output-dir "$routes_root"
fi

if [[ ! -s "$run_root/route_only_comparison.json" ]]; then
  "$python_bin" demo/stage_c_a800_uvg_adaptation.py compare-routes \
    --sample-manifest "$joint_root/manifests/joint_samples.jsonl" \
    --baseline-routes "$joint_root/routes/controller/manifest.json" \
    --candidate-routes "$routes_root/manifest.json" \
    --output "$run_root/route_only_comparison.json"
fi

"$python_bin" - "$run_root" "$experiment_start_epoch" <<'PY'
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1]).resolve()
start = int(sys.argv[2])
gpu_samples = root / "logs" / "training_gpu_samples.csv"
peak = 0
if gpu_samples.is_file():
    for line in gpu_samples.read_text(encoding="utf-8").splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) >= 3:
            try:
                peak = max(peak, int(float(fields[2])))
            except ValueError:
                pass
usage = {}
for name in ("/root", "/root/autodl-tmp", "/root/autodl-fs"):
    value = shutil.disk_usage(name)
    usage[name] = {
        "total_bytes": value.total,
        "used_bytes": value.used,
        "free_bytes": value.free,
    }
value = {
    "experiment": "A800 UVG residual-expert adaptation route-only stage",
    "status": "complete",
    "completed_utc": datetime.now(timezone.utc).isoformat(),
    "wall_seconds": int(datetime.now(timezone.utc).timestamp()) - start,
    "peak_nvidia_smi_used_memory_mib": peak,
    "ordinary_file_bytes": sum(
        path.stat().st_size for path in root.rglob("*") if path.is_file()
    ),
    "mounts": usage,
    "single_gpu": True,
    "formal_quality_evaluation_complete": False,
}
target = root / "training_resource_snapshot.json"
temporary = target.with_suffix(".json.tmp")
temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, target)
PY

printf 'complete\n' > "$run_root/adapted_route_only.complete.tmp"
mv "$run_root/adapted_route_only.complete.tmp" \
  "$run_root/adapted_route_only.complete"
echo "COMPLETE UVG adaptation route-only stage utc=$(date -u +%FT%TZ)"
df -h /root /root/autodl-tmp /root/autodl-fs
