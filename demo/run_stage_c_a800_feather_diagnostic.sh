#!/usr/bin/env bash
set -Eeuo pipefail

repo=/root/autodl-tmp/adaptive-chunk-coding
fast_root=/root/autodl-tmp/DCVC
env_root="$fast_root/envs/dcvcuf"
persist=/root/autodl-fs/DCVC
joint_root="$persist/runs/a800_joint_evaluation_20260918"
run_root="$persist/runs/a800_feather_diagnostic_20260919"
log_file="$run_root/logs/feather_diagnostic.log"
started_marker="$run_root/run_started_epoch.txt"
max_wall_seconds=7200

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

mkdir -p "$run_root/logs"
if [[ ! -s "$started_marker" ]]; then
  date +%s > "$started_marker.tmp"
  mv "$started_marker.tmp" "$started_marker"
fi
experiment_start_epoch=$(<"$started_marker")
invocation_start_epoch=$(date +%s)
cp "$0" "$run_root/logs/executed_run_stage_c_a800_feather_diagnostic.sh"
exec > >(tee -a "$log_file") 2>&1

heartbeat() {
  while true; do
    local now_epoch completed run_bytes used path
    now_epoch=$(date +%s)
    completed=0
    for marker in "$run_root"/samples/*/summary.json; do
      [[ -s "$marker" ]] && completed=$((completed + 1))
    done
    run_bytes=$(du -sb "$run_root" 2>/dev/null | awk '{print $1}')
    echo "HEARTBEAT feather_diagnostic utc=$(date -u +%FT%TZ) experiment_elapsed_s=$((now_epoch - experiment_start_epoch)) invocation_elapsed_s=$((now_epoch - invocation_start_epoch)) samples=$completed/37 run_bytes=${run_bytes:-0}"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits || true
    df -h /root /root/autodl-tmp /root/autodl-fs
    if [[ $((now_epoch - invocation_start_epoch)) -ge "$max_wall_seconds" ]]; then
      echo "ERROR experiment wall limit reached" >&2
      kill -TERM "$main_pid" 2>/dev/null || true
      return
    fi
    for path in /root /root/autodl-tmp /root/autodl-fs; do
      used=$(df -P "$path" | awk 'NR==2 {gsub(/%/, "", $5); print $5}')
      if [[ "$used" -ge 80 ]]; then
        echo "ERROR disk limit reached: path=$path used_percent=$used" >&2
        kill -TERM "$main_pid" 2>/dev/null || true
        return
      fi
    done
    sleep 60
  done
}

gpu_monitor() {
  while true; do
    nvidia-smi \
      --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw \
      --format=csv,noheader,nounits || true
    sleep 5
  done
}

cd "$repo"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR repository must be clean before freezing the diagnostic" >&2
  exit 1
fi
[[ -s "$joint_root/formal/joint_evaluation_summary.json" ]]
[[ -f "$joint_root/formal_evaluation.complete" ]]

git_commit=$(git rev-parse HEAD)
joint_sha=$(sha256sum "$joint_root/formal/joint_evaluation_summary.json" | awk '{print $1}')
checkpoint_sha=$(python - "$joint_root/frozen_controller.json" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["checkpoint_sha256"])
PY
)
python - "$run_root/frozen_source.json" "$git_commit" "$joint_sha" "$checkpoint_sha" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
value = {
    "experiment": "A800 frozen-v5 Generate feather diagnostic",
    "status": "frozen_before_diagnostic",
    "frozen_utc": datetime.now(timezone.utc).isoformat(),
    "git_commit": sys.argv[2],
    "source_joint_summary_sha256": sys.argv[3],
    "controller_checkpoint_sha256": sys.argv[4],
    "feather_pixels": [8, 16, 32],
    "scientific_boundary": {
        "codec_rerun": False,
        "seedvr2_rerun": False,
        "action_map_changed": False,
        "single_gpu_visible": True,
    },
}
if path.is_file():
    existing = json.loads(path.read_text(encoding="utf-8"))
    for key in (
            "git_commit", "source_joint_summary_sha256",
            "controller_checkpoint_sha256", "feather_pixels"):
        if existing.get(key) != value[key]:
            raise RuntimeError(f"frozen diagnostic source changed: {key}")
    raise SystemExit(0)
temporary = path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY

echo "START feather_diagnostic utc=$(date -u +%FT%TZ) git=$git_commit"
gpu_monitor >> "$run_root/logs/gpu_samples.csv" 2>&1 &
gpu_monitor_pid=$!
python demo/stage_c_a800_feather_diagnostic.py \
  --joint-root "$joint_root" --output-root "$run_root" \
  --feather-pixels 8 16 32 --visual-frame 9 &
main_pid=$!
heartbeat &
heartbeat_pid=$!
cleanup() {
  kill "$heartbeat_pid" "$gpu_monitor_pid" 2>/dev/null || true
  wait "$heartbeat_pid" "$gpu_monitor_pid" 2>/dev/null || true
}
trap cleanup EXIT
wait "$main_pid"
[[ -s "$run_root/formal.complete" ]]

python - "$run_root" "$experiment_start_epoch" "$git_commit" <<'PY'
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
started = int(sys.argv[2])
files = [path for path in root.rglob("*") if path.is_file() and not path.is_symlink()]
mounts = {}
for name in ("/root", "/root/autodl-tmp", "/root/autodl-fs"):
    value = os.statvfs(name)
    mounts[name] = {
        "total": value.f_blocks * value.f_frsize,
        "used": (value.f_blocks - value.f_bfree) * value.f_frsize,
        "free": value.f_bavail * value.f_frsize,
    }
gpu = subprocess.check_output([
    "nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
    "--format=csv,noheader,nounits"], text=True).strip()
gpu_samples = root / "logs" / "gpu_samples.csv"
peak_memory_mib = 0
if gpu_samples.is_file():
    for line in gpu_samples.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 3:
            try:
                peak_memory_mib = max(peak_memory_mib, int(fields[2]))
            except ValueError:
                pass
value = {
    "experiment": "A800 frozen-v5 Generate feather diagnostic final snapshot",
    "status": "complete",
    "time_utc": datetime.now(timezone.utc).isoformat(),
    "wall_seconds": int(datetime.now(timezone.utc).timestamp()) - started,
    "git_commit": sys.argv[3],
    "sample_count": len(list((root / "samples").glob("*/summary.json"))),
    "ordinary_file_count_before_snapshot": len(files),
    "ordinary_file_bytes_before_snapshot": sum(path.stat().st_size for path in files),
    "mounts": mounts,
    "gpu_current": gpu,
    "peak_nvidia_smi_used_memory_mib": peak_memory_mib,
    "single_gpu": True,
    "codec_or_seedvr2_rerun": False,
}
path = root / "final_resource_snapshot.json"
temporary = path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY

echo "COMPLETE feather_diagnostic utc=$(date -u +%FT%TZ)"
