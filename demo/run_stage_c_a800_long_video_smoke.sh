#!/usr/bin/env bash
set -Eeuo pipefail

repo=/root/autodl-tmp/adaptive-chunk-coding
fast_root=/root/autodl-tmp/DCVC
env_root="$fast_root/envs/dcvcuf"
persist=/root/autodl-fs/DCVC
run_root="$persist/runs/a800_long_video_smoke_20260920"
router_root="$run_root/router_windows"
plan_root="$run_root/plan"
codec_root="$run_root/codec"
roi_plan_root="$run_root/seedvr2_roi_plan"
roi_restore_root="$run_root/seedvr2_roi_restore"
evaluation_root="$run_root/evaluation"
log_root="$run_root/logs"
python_bin="$env_root/bin/python"
torchrun_bin="$env_root/bin/torchrun"
sample_id=dev-s000-f16-x384-y096
source_dir="$persist/data/REDS/val_sharp/000"
route_f00="$persist/runs/a800_spatial_consistency_20260920/routes_v6_combined/dev-s000-f00-x384-y096.json"
controller="$persist/runs/a800_uvg_adaptation_20260919/controller_residual_adaptation/controller_v5.pt"

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

mkdir -p "$log_root" "$router_root/$sample_id/uniform_gate"
invocation_start_epoch=$(date +%s)
started_marker="$run_root/run_started_epoch.txt"
if [[ ! -s "$started_marker" ]]; then
  date +%s > "$started_marker.tmp"
  mv "$started_marker.tmp" "$started_marker"
fi
experiment_start_epoch=$(<"$started_marker")
cp "$0" "$log_root/executed_run_stage_c_a800_long_video_smoke.sh"
exec > >(tee -a "$log_root/long_video_smoke.log") 2>&1

heartbeat_pid=
cleanup() {
  if [[ -n "$heartbeat_pid" ]]; then kill "$heartbeat_pid" 2>/dev/null || true; fi
  if [[ -n "$heartbeat_pid" ]]; then wait "$heartbeat_pid" 2>/dev/null || true; fi
}
fail() {
  status=$?
  cleanup
  if (( status != 0 )); then
    printf '%s\n' "$status" > "$run_root/run.failed.tmp"
    mv "$run_root/run.failed.tmp" "$run_root/run.failed"
    echo "FAILED long_video_smoke status=$status utc=$(date -u +%FT%TZ)"
  fi
  exit "$status"
}
trap fail EXIT INT TERM
(
  while true; do
    printf 'HEARTBEAT long_video_smoke utc=%s elapsed_s=%s ' \
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

echo "START long_video_smoke utc=$(date -u +%FT%TZ)"
cd "$repo"
[[ -s "$route_f00" ]]
[[ -s "$controller" ]]
[[ -d "$source_dir" ]]

"$python_bin" demo/stage_c_long_video_plan.py self-test
"$python_bin" demo/stage_c_long_video_seedvr2.py self-test
"$python_bin" demo/stage_c_a800_spatial_consistency.py self-test

sample_manifest="$router_root/sample_f16.jsonl"
if [[ ! -s "$sample_manifest" ]]; then
  "$python_bin" - "$sample_manifest" "$source_dir" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

target = Path(sys.argv[1])
source = Path(sys.argv[2]).resolve()
paths = sorted(source.glob("*.png"))[16:33]
if len(paths) != 17:
    raise RuntimeError("REDS/000 does not provide frames 16 through 32")
digest = hashlib.sha256()
for path in paths:
    digest.update(path.read_bytes())
record = {
    "sample_id": "dev-s000-f16-x384-y096",
    "dataset": "REDS",
    "split": "validation",
    "data_role": "development",
    "source_role": (
        "REDS validation development window used only for the long-video "
        "mechanism smoke; not independent test evidence"),
    "sequence": "000",
    "source_dir": str(source),
    "frame_start": 16,
    "frame_count": 17,
    "source_files": [str(path) for path in paths],
    "selected_source_sha256": digest.hexdigest(),
    "crop": {"x": 384, "y": 96, "width": 512, "height": 512},
    "seed": 21260920,
}
temporary = target.with_suffix(".jsonl.tmp")
temporary.write_text(json.dumps(record) + "\n", encoding="utf-8")
os.replace(temporary, target)
PY
fi

gate_root="$router_root/$sample_id/uniform_gate"
if [[ ! -s "$gate_root/summary.json" ]]; then
  "$python_bin" demo/stage_c_restorer_gate.py \
    --sequence-name "$sample_id" --source-dir "$source_dir" \
    --frame-start 16 --source-role \
    "REDS validation development window used for long-video mechanism smoke" \
    --output-dir "$gate_root" --frame-count 17 \
    --width 512 --height 512 --crop-x 384 --crop-y 96 \
    --qps 16 --decode-repeats 1 --visual-frame 9
fi

routes_v5="$router_root/routes_v5"
if [[ ! -s "$routes_v5/manifest.json" ]]; then
  "$python_bin" demo/stage_c_a800_low_budget_v5.py route \
    --sample-manifest "$sample_manifest" \
    --base-probe-root "$router_root" \
    --checkpoint "$controller" --output-dir "$routes_v5"
fi

routes_v6="$router_root/routes_v6"
if [[ ! -s "$routes_v6/manifest.json" ]]; then
  "$python_bin" demo/stage_c_a800_spatial_consistency.py route \
    --input-manifest "$routes_v5/manifest.json" \
    --output-dir "$routes_v6" --spatial-lambda 0.004 \
    --expected-sample-count 1
fi
route_f16="$routes_v6/$sample_id.json"
[[ -s "$route_f16" ]]

if [[ ! -s "$plan_root/plan_summary.json" ]]; then
  "$python_bin" demo/stage_c_long_video_plan.py build \
    --source-dir "$source_dir" --source-start 0 --frame-count 33 \
    --sequence-name REDS-000 \
    --source-role \
    "REDS validation development mechanism smoke; not independent test evidence" \
    --crop-x 384 --crop-y 96 --width 512 --height 512 \
    --window-route "0=$route_f00" --window-route "16=$route_f16" \
    --output-dir "$plan_root"
fi

stream="$codec_root/streams/continuous_33f_spatial_qp.dqvc"
if [[ ! -s "$codec_root/encode_summary.json" || ! -s "$stream" ]]; then
  "$python_bin" demo/stage_c_spatial_quality_codec.py encode \
    --gate-summary "$plan_root/long_gate.json" \
    --route-summary "$plan_root/long_route.json" \
    --output-stream "$stream" --output-dir "$codec_root" \
    --frame-count 33 --cell-size 64 \
    --generate-qp 8 --base-qp 16 --enhance-qp 32
fi
if [[ ! -s "$codec_root/decode_summary.json" ]]; then
  "$python_bin" demo/stage_c_spatial_quality_codec.py decode \
    --input-stream "$stream" --output-dir "$codec_root"
fi

if [[ ! -s "$roi_plan_root/manifest.json" ]]; then
  "$python_bin" demo/stage_c_long_video_seedvr2.py prepare \
    --codec-dir "$codec_root" --output-dir "$roi_plan_root" \
    --context-pixels 64 --processing-scale 1.5
fi

"$torchrun_bin" --standalone --nproc-per-node=1 \
  demo/stage_c_long_video_seedvr2.py restore \
  --manifest "$roi_plan_root/manifest.json" \
  --output-root "$roi_restore_root" --seed 21260920 \
  --sample-steps 1 --cfg-scale 1.0 --dit-dtype bfloat16

if [[ ! -s "$evaluation_root/summary.json" ]]; then
  "$python_bin" demo/stage_c_long_video_seedvr2.py evaluate \
    --manifest "$roi_plan_root/manifest.json" \
    --gate-summary "$plan_root/long_gate.json" \
    --codec-dir "$codec_root" --restored-root "$roi_restore_root" \
    --output-dir "$evaluation_root" --feather-pixels 16
fi

"$python_bin" - "$run_root" "$experiment_start_epoch" \
  "$invocation_start_epoch" <<'PY'
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1]).resolve()
started = int(sys.argv[2])
invocation_started = int(sys.argv[3])
evaluation = json.loads((root / "evaluation/summary.json").read_text())
encode = json.loads((root / "codec/encode_summary.json").read_text())
decode = json.loads((root / "codec/decode_summary.json").read_text())
files = [path for path in root.rglob("*") if path.is_file()]
mounts = {}
for name in ("/root", "/root/autodl-tmp", "/root/autodl-fs"):
    value = shutil.disk_usage(name)
    mounts[name] = {
        "total_bytes": value.total,
        "used_bytes": value.used,
        "free_bytes": value.free,
    }
target = root / "run_summary.json"
existing = (
    json.loads(target.read_text(encoding="utf-8"))
    if target.is_file() else {})
now = datetime.now(timezone.utc).isoformat()
elapsed = int(time.time()) - started
first_completed = existing.get("first_completed_utc", now)
first_elapsed = existing.get("first_completion_elapsed_seconds", elapsed)
summary = {
    "experiment": "A800 single-GPU 33-frame long-video mechanism smoke",
    "status": "complete",
    "completed_utc": first_completed,
    "first_completed_utc": first_completed,
    "latest_finalize_utc": now,
    "git_commit_at_execution": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(),
    "wall_seconds_since_first_start": first_elapsed,
    "first_completion_elapsed_seconds": first_elapsed,
    "latest_finalize_invocation_seconds": int(time.time()) - invocation_started,
    "elapsed_since_first_start_at_latest_finalize": elapsed,
    "frames": 33,
    "single_gpu": True,
    "continuous_codec_reference": True,
    "time_varying_action_maps": encode["time_varying_action_maps"],
    "actual_stream_bytes": encode["stream_bytes"],
    "fresh_decode_pixel_exact": evaluation[
        "fresh_decode_regression"]["pixel_exact"],
    "quality": evaluation["quality"],
    "runtime": evaluation["runtime"],
    "peak_cuda_allocated_bytes": evaluation[
        "runtime"]["peak_cuda_allocated_bytes"],
    "ordinary_file_count": len(files),
    "ordinary_file_bytes": sum(path.stat().st_size for path in files),
    "mounts": mounts,
    "scientific_role": (
        "development mechanism smoke; not independent benchmark evidence"),
    "artifacts": {
        "evaluation": str(root / "evaluation/summary.json"),
        "visual": evaluation["visual"],
        "stream": encode["stream"],
        "decode_frames": decode["fresh_decode_dir"],
    },
}
temporary = target.with_suffix(".json.tmp")
temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, target)
PY

if [[ ! -f "$run_root/run.complete" ]]; then
  printf 'complete\n' > "$run_root/run.complete.tmp"
  mv "$run_root/run.complete.tmp" "$run_root/run.complete"
fi
rm -f "$run_root/run.failed"
cleanup
trap - EXIT INT TERM
du -sb "$run_root"
df -h /root /root/autodl-tmp /root/autodl-fs
echo "COMPLETE long_video_smoke utc=$(date -u +%FT%TZ) elapsed_s=$(($(date +%s) - experiment_start_epoch))"
