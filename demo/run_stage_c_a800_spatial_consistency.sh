#!/usr/bin/env bash
set -Eeuo pipefail

repo=/root/autodl-tmp/adaptive-chunk-coding
fast_root=/root/autodl-tmp/DCVC
env_root="$fast_root/envs/dcvcuf"
persist=/root/autodl-fs/DCVC
joint_root="$persist/runs/a800_joint_evaluation_20260918"
adaptation_root="$persist/runs/a800_uvg_adaptation_20260919"
run_root="$persist/runs/a800_spatial_consistency_20260920"
log_root="$run_root/logs"
log_file="$log_root/spatial_consistency.log"
started_marker="$run_root/run_started_epoch.txt"
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

mkdir -p "$log_root" "$run_root/formal"
if [[ ! -f "$started_marker" ]]; then
  date +%s > "$started_marker.tmp"
  mv "$started_marker.tmp" "$started_marker"
fi
experiment_start_epoch=$(<"$started_marker")
cp "$0" "$log_root/executed_run_stage_c_a800_spatial_consistency.sh"
exec > >(tee -a "$log_file") 2>&1

heartbeat_pid=
cleanup() {
  if [[ -n "$heartbeat_pid" ]]; then kill "$heartbeat_pid" 2>/dev/null || true; fi
  if [[ -n "$heartbeat_pid" ]]; then wait "$heartbeat_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM
(
  while true; do
    printf 'HEARTBEAT spatial_consistency utc=%s elapsed_s=%s ' \
      "$(date -u +%FT%TZ)" "$(($(date +%s) - experiment_start_epoch))"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits | tr '\n' ';' || true
    printf ' '
    df -h /root /root/autodl-tmp /root/autodl-fs | tail -n +2 | tr '\n' ';'
    printf '\n'
    sleep 60
  done
) >> "$log_root/heartbeat.log" 2>&1 &
heartbeat_pid=$!

echo "START spatial_consistency utc=$(date -u +%FT%TZ)"
cd "$repo"
if [[ -n "$(git status --short)" ]]; then
  echo "ERROR repository must be clean before frozen routing" >&2
  git status --short
  exit 1
fi

v5_manifest="$joint_root/routes/controller/manifest.json"
adapted_manifest="$adaptation_root/routes_joint37/manifest.json"
[[ -s "$v5_manifest" ]]
[[ -s "$adapted_manifest" ]]
[[ -s "$adaptation_root/adapted_route_only.complete" ]]

if [[ ! -s "$run_root/frozen_source.json" ]]; then
  "$python_bin" - "$run_root/frozen_source.json" \
    "$v5_manifest" "$adapted_manifest" <<'PY'
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
    "experiment": "A800 v6 exact spatial-consistency freeze",
    "status": "frozen_before_spatial_rerouting",
    "frozen_utc": datetime.now(timezone.utc).isoformat(),
    "git_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(),
    "spatial_lambda": 0.004,
    "lambda_origin": (
        "reuse the previous 0.004 fragment-penalty utility scale as an "
        "explicit cost per Generate/non-Generate neighbor edge"),
    "v5_route_manifest": str(Path(sys.argv[2]).resolve()),
    "v5_route_manifest_sha256": sha(sys.argv[2]),
    "uvg_adapted_route_manifest": str(Path(sys.argv[3]).resolve()),
    "uvg_adapted_route_manifest_sha256": sha(sys.argv[3]),
    "hard_promotion_gate": False,
}
temporary = target.with_suffix(".json.tmp")
temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, target)
PY
fi

"$python_bin" demo/stage_c_a800_spatial_consistency.py self-test

if [[ ! -s "$run_root/formal/v5_spatial_sweep.json" ]]; then
  "$python_bin" demo/stage_c_a800_spatial_consistency.py sweep \
    --input-manifest "$v5_manifest" \
    --output "$run_root/formal/v5_spatial_sweep.json" \
    --expected-sample-count 37
fi
if [[ ! -s "$run_root/routes_v5_spatial/manifest.json" ]]; then
  "$python_bin" demo/stage_c_a800_spatial_consistency.py route \
    --input-manifest "$v5_manifest" \
    --output-dir "$run_root/routes_v5_spatial" \
    --spatial-lambda 0.004 --expected-sample-count 37
fi

if [[ ! -s "$run_root/formal/adapted_spatial_sweep.json" ]]; then
  "$python_bin" demo/stage_c_a800_spatial_consistency.py sweep \
    --input-manifest "$adapted_manifest" \
    --output "$run_root/formal/adapted_spatial_sweep.json" \
    --expected-sample-count 37
fi
if [[ ! -s "$run_root/routes_v6_combined/manifest.json" ]]; then
  "$python_bin" demo/stage_c_a800_spatial_consistency.py route \
    --input-manifest "$adapted_manifest" \
    --output-dir "$run_root/routes_v6_combined" \
    --spatial-lambda 0.004 --expected-sample-count 37
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
v5 = json.loads((root / "formal/v5_spatial_sweep.json").read_text())
adapted = json.loads((root / "formal/adapted_spatial_sweep.json").read_text())
v5_routes = json.loads((root / "routes_v5_spatial/manifest.json").read_text())
combined_routes = json.loads(
    (root / "routes_v6_combined/manifest.json").read_text())
assert v5["lambda_zero_exact_action_regression"] is True
assert adapted["lambda_zero_exact_action_regression"] is True
assert v5_routes["sample_count"] == combined_routes["sample_count"] == 37
assert v5_routes["spatial_lambda"] == combined_routes["spatial_lambda"] == 0.004
files = [path for path in root.rglob("*") if path.is_file()]
mounts = {}
for name in ("/root", "/root/autodl-tmp", "/root/autodl-fs"):
    value = shutil.disk_usage(name)
    mounts[name] = {
        "total_bytes": value.total,
        "used_bytes": value.used,
        "free_bytes": value.free,
    }
summary = {
    "experiment": "A800 v6 exact spatial-consistency route-only stage",
    "status": "complete",
    "completed_utc": datetime.now(timezone.utc).isoformat(),
    "wall_seconds": int(__import__("time").time()) - start,
    "spatial_lambda": 0.004,
    "sample_count": 37,
    "v5_spatial_only": v5["selected_summary"],
    "uvg_adaptation_plus_spatial": adapted["selected_summary"],
    "integrity": {
        "self_test_passed": True,
        "lambda_zero_exact_action_regression_for_both_inputs": True,
        "route_manifests_complete": True,
        "quality_evaluation_complete": False,
    },
    "ordinary_file_count": len(files),
    "ordinary_file_bytes": sum(path.stat().st_size for path in files),
    "mounts": mounts,
    "single_gpu": True,
}
target = root / "route_only_summary.json"
temporary = target.with_suffix(".json.tmp")
temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, target)
PY

printf 'complete\n' > "$run_root/spatial_route_only.complete.tmp"
mv "$run_root/spatial_route_only.complete.tmp" \
  "$run_root/spatial_route_only.complete"
cleanup
trap - EXIT
du -sb "$run_root"
df -h /root /root/autodl-tmp /root/autodl-fs
echo "COMPLETE spatial_consistency utc=$(date -u +%FT%TZ) elapsed_s=$(($(date +%s) - experiment_start_epoch))"
