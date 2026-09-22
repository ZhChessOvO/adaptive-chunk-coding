#!/usr/bin/env bash
set -Eeuo pipefail

repo=/root/autodl-tmp/adaptive-chunk-coding
fast_root=/root/autodl-tmp/DCVC
persist=/root/autodl-fs/DCVC
python_bin="$fast_root/envs/dcvcuf/bin/python"
run_root=${RUN_ROOT:-$persist/runs/a800_spatial_qp_balanced_v2_20260922}
log_root="$run_root/logs"
runner="$repo/demo/run_stage_c_a800_spatial_qp_finetune.sh"

mkdir -p "$log_root"
rm -f "$run_root/run.complete"
started_marker="$run_root/run_started_epoch.txt"
if [[ ! -s "$started_marker" ]]; then
  date +%s > "$started_marker.tmp"
  mv "$started_marker.tmp" "$started_marker"
fi
experiment_start_epoch=$(<"$started_marker")
cp "$0" "$log_root/executed_run_stage_c_a800_spatial_qp_balanced_v2.sh"
exec > >(tee -a "$log_root/spatial_qp_balanced_v2.log") 2>&1

heartbeat_pid=
cleanup() {
  if [[ -n "$heartbeat_pid" ]]; then kill "$heartbeat_pid" 2>/dev/null || true; fi
  if [[ -n "$heartbeat_pid" ]]; then wait "$heartbeat_pid" 2>/dev/null || true; fi
}
require_mount_headroom() {
  local mount used_percent
  for mount in /root /root/autodl-tmp /root/autodl-fs; do
    used_percent=$(df -P "$mount" | awk 'NR==2 {gsub(/%/, "", $5); print $5}')
    if (( used_percent >= 80 )); then
      echo "refusing to start: $mount is ${used_percent}% full" >&2
      return 1
    fi
  done
}
fail() {
  status=$?
  cleanup
  if (( status != 0 )); then
    printf '%s\n' "$status" > "$run_root/run.failed.tmp"
    mv "$run_root/run.failed.tmp" "$run_root/run.failed"
    echo "FAILED spatial_qp_balanced_v2 status=$status utc=$(date -u +%FT%TZ)"
  fi
  exit "$status"
}
trap fail EXIT INT TERM
(
  while true; do
    printf 'HEARTBEAT spatial_qp_balanced_v2 utc=%s elapsed_s=%s ' \
      "$(date -u +%FT%TZ)" "$(($(date +%s) - experiment_start_epoch))"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits | tr '\n' ';' || true
    for candidate in uvg025 uvg050; do
      progress="$run_root/$candidate/training/progress.json"
      if [[ -s "$progress" ]]; then
        "$python_bin" - "$candidate" "$progress" <<'PY' || true
import json
import sys
value = json.load(open(sys.argv[2], encoding="utf-8"))
print(f" {sys.argv[1]}={value['completed_steps']}/{value['target_steps']}", end="")
PY
      fi
    done
    printf ' '
    df -h /root /root/autodl-tmp /root/autodl-fs | tail -n +2 | tr '\n' ';'
    printf '\n'
    sleep 60
  done
) >> "$log_root/heartbeat.log" 2>&1 &
heartbeat_pid=$!

cd "$repo"
[[ -x "$python_bin" ]]
[[ -x "$runner" ]]
[[ -s checkpoints/cvpr2026_image.pth.tar ]]
[[ -s checkpoints/cvpr2026_video_hts.pth.tar ]]
[[ -d "$persist/data/REDS/train_sharp" ]]
[[ -d "$persist/data/UVG_adaptation/samples" ]]
if [[ "$(nvidia-smi -L | wc -l)" -ne 1 ]]; then
  echo "balanced v2 requires exactly one visible GPU" >&2
  exit 3
fi
require_mount_headroom

"$python_bin" demo/stage_c_spatial_qp_finetune.py self-test
"$python_bin" - "$run_root/protocol.json" <<'PY'
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
current_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], text=True).strip()
value = {
    "experiment": "domain-balanced spatial-QP-aware DCVC-UF fine-tuning v2",
    "status": "fixed-before-training",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "git_commit": current_commit,
    "single_gpu": True,
    "initialization": "same official frozen DCVC-UF checkpoints for both candidates",
    "shared_training": {
        "steps": 1000,
        "frames_per_step": 17,
        "patch": [512, 512],
        "learning_rate": 2e-6,
        "mixed_route_fraction": 0.75,
        "uniform_rehearsal_fraction_per_domain": 0.25,
    },
    "candidates": [
        {"name": "uvg025", "uvg_step_fraction": 0.25},
        {"name": "uvg050", "uvg_step_fraction": 0.50},
    ],
    "uvg_sampling": "balanced across Beauty, Bosphorus, HoneyBee, Jockey, ShakeNDry",
    "scientific_role": "candidate training; quality selection requires the fixed real-stream evaluation",
}
if path.exists():
    existing = json.loads(path.read_text(encoding="utf-8"))
    if existing != value:
        if existing.get("git_commit") != current_commit:
            raise RuntimeError(
                "refusing to resume balanced v2 under a different Git commit")
        value = existing
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
os.replace(temporary, path)
PY

echo "START spatial_qp_balanced_v2 utc=$(date -u +%FT%TZ) root=$run_root"
for specification in uvg025:0.25 uvg050:0.50; do
  candidate=${specification%%:*}
  probability=${specification##*:}
  candidate_root="$run_root/$candidate"
  if [[ -f "$candidate_root/run.complete" ]]; then
    echo "SKIP complete candidate=$candidate"
    continue
  fi
  require_mount_headroom
  echo "START candidate=$candidate uvg_probability=$probability utc=$(date -u +%FT%TZ)"
  RUN_ROOT="$candidate_root" MAX_STEPS=1000 PATCH_SIZE=512 SAVE_EVERY=25 \
    UVG_PROBABILITY="$probability" SAMPLING_MODE=balanced \
    timeout --signal=TERM --kill-after=10m 6h "$runner" train
  [[ -f "$candidate_root/run.complete" ]]
  echo "COMPLETE candidate=$candidate utc=$(date -u +%FT%TZ)"
done

"$python_bin" - "$run_root" <<'PY'
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
candidates = {}
for name in ("uvg025", "uvg050"):
    summary_path = root / name / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    schedule = json.loads(
        (root / name / "training" / "sampling_schedule.json").read_text(
            encoding="utf-8"))
    candidates[name] = {
        "run_summary": str(summary_path.resolve()),
        "training_steps": summary["training_steps"],
        "peak_cuda_allocated_bytes": summary["peak_cuda_allocated_bytes"],
        "validation_stream_bytes": summary["actual_validation_stream_bytes"],
        "fresh_decode_pixel_exact": summary["fresh_decode_pixel_exact"],
        "dataset_step_counts": schedule["dataset_step_counts"],
        "uvg_source_step_counts": schedule["uvg_source_step_counts"],
        "uniform_step_counts": schedule["uniform_step_counts"],
    }
files = [item for item in root.rglob("*") if item.is_file()]
mounts = {}
for mount in ("/root", "/root/autodl-tmp", "/root/autodl-fs"):
    usage = shutil.disk_usage(mount)
    mounts[mount] = {
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    }
value = {
    "experiment": "domain-balanced spatial-QP-aware DCVC-UF fine-tuning v2",
    "status": "training-complete-evaluation-pending",
    "completed_utc": datetime.now(timezone.utc).isoformat(),
    "single_gpu": True,
    "candidates": candidates,
    "ordinary_file_count": len(files),
    "ordinary_file_bytes": sum(item.stat().st_size for item in files),
    "mounts": mounts,
    "next_step": "fixed 3 REDS + 3 UVG uniform-QP real-stream curves",
}
path = root / "run_summary.json"
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
os.replace(temporary, path)
PY

rm -f "$run_root/run.failed"
printf 'complete\n' > "$run_root/run.complete.tmp"
mv "$run_root/run.complete.tmp" "$run_root/run.complete"
cleanup
trap - EXIT INT TERM
du -sh "$run_root"
df -h /root /root/autodl-tmp /root/autodl-fs
echo "COMPLETE spatial_qp_balanced_v2 utc=$(date -u +%FT%TZ) elapsed_s=$(($(date +%s) - experiment_start_epoch))"
