#!/usr/bin/env bash
set -Eeuo pipefail

repo=/root/autodl-tmp/adaptive-chunk-coding
fast_root=/root/autodl-tmp/DCVC
env_root="$fast_root/envs/dcvcuf"
persist=/root/autodl-fs/DCVC
train_root="$persist/runs/a800_spatial_qp_balanced_v2_20260922"
endpoint_root="$persist/runs/a800_spatial_qp_finetune_eval_20260920"
run_root=${RUN_ROOT:-$persist/runs/a800_spatial_qp_balanced_v2_eval_20260922}
protocol_root="$run_root/protocol"
log_root="$run_root/logs"
python_bin="$env_root/bin/python"
endpoint_protocol="$endpoint_root/protocol/protocol.json"
endpoint_summary="$endpoint_root/summary.json"
balanced_25_i="$train_root/uvg025/training/checkpoints/image_model.pth.tar"
balanced_25_p="$train_root/uvg025/training/checkpoints/video_model_hts.pth.tar"
balanced_50_i="$train_root/uvg050/training/checkpoints/image_model.pth.tar"
balanced_50_p="$train_root/uvg050/training/checkpoints/video_model_hts.pth.tar"
max_wall_seconds=${MAX_WALL_SECONDS:-21600}

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

mkdir -p "$log_root" "$protocol_root"
rm -f "$run_root/run.complete"
started_marker="$run_root/run_started_epoch.txt"
if [[ ! -s "$started_marker" ]]; then
  date +%s > "$started_marker.tmp"
  mv "$started_marker.tmp" "$started_marker"
fi
experiment_start_epoch=$(<"$started_marker")
cp "$0" "$log_root/executed_run_stage_c_a800_spatial_qp_balanced_eval.sh"
exec > >(tee -a "$log_root/spatial_qp_balanced_eval.log") 2>&1
tee_pid=$!

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
      echo "refusing to continue: $mount is ${used_percent}% full" >&2
      return 1
    fi
  done
}
require_time_headroom() {
  local elapsed
  elapsed=$(($(date +%s) - experiment_start_epoch))
  if (( elapsed >= max_wall_seconds )); then
    echo "evaluation exceeded ${max_wall_seconds}s wall-time guard" >&2
    return 1
  fi
}
fail() {
  status=$?
  cleanup
  if (( status != 0 )); then
    rm -f "$run_root/run.complete"
    printf '%s\n' "$status" > "$run_root/run.failed.tmp"
    mv "$run_root/run.failed.tmp" "$run_root/run.failed"
    echo "FAILED spatial_qp_balanced_eval status=$status utc=$(date -u +%FT%TZ)"
  fi
  exit "$status"
}
trap fail EXIT INT TERM
(
  while true; do
    printf 'HEARTBEAT spatial_qp_balanced_eval utc=%s elapsed_s=%s ' \
      "$(date -u +%FT%TZ)" "$(($(date +%s) - experiment_start_epoch))"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits | tr '\n' ';' || true
    progress="$run_root/progress.json"
    if [[ -s "$progress" ]]; then
      "$python_bin" - "$progress" <<'PY' || true
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
print(f" progress={value['completed_tasks']}/{value['total_tasks']}", end="")
PY
    fi
    printf ' '
    df -h /root /root/autodl-tmp /root/autodl-fs | tail -n +2 | tr '\n' ';'
    printf '\n'
    sleep 60
  done
) >> "$log_root/heartbeat.log" 2>&1 &
heartbeat_pid=$!

echo "START spatial_qp_balanced_eval utc=$(date -u +%FT%TZ) root=$run_root"
cd "$repo"
[[ -x "$python_bin" ]]
[[ -f "$train_root/run.complete" ]]
[[ -f "$endpoint_root/run.complete" ]]
[[ -s "$endpoint_protocol" && -s "$endpoint_summary" ]]
[[ -s "$balanced_25_i" && -s "$balanced_25_p" ]]
[[ -s "$balanced_50_i" && -s "$balanced_50_p" ]]
if [[ "$(nvidia-smi -L | wc -l)" -ne 1 ]]; then
  echo "balanced evaluation requires exactly one visible GPU" >&2
  exit 3
fi
require_mount_headroom

"$python_bin" demo/stage_c_spatial_qp_balanced_eval.py self-test
protocol="$protocol_root/protocol.json"
"$python_bin" demo/stage_c_spatial_qp_balanced_eval.py prepare \
  --endpoint-protocol "$endpoint_protocol" \
  --endpoint-summary "$endpoint_summary" --output-dir "$protocol_root" \
  --balanced-25-image "$balanced_25_i" --balanced-25-video "$balanced_25_p" \
  --balanced-50-image "$balanced_50_i" --balanced-50-video "$balanced_50_p"

tasks_tsv="$protocol_root/tasks.tsv"
"$python_bin" demo/stage_c_spatial_qp_balanced_eval.py list-tasks \
  --protocol "$protocol" > "$tasks_tsv.tmp"
mv "$tasks_tsv.tmp" "$tasks_tsv"

task_index=0
task_total=$(wc -l < "$tasks_tsv")
while IFS=$'\t' read -r task_id sample_id family model_role gate route \
    model_i model_p checkpoint_role output_rel; do
  task_index=$((task_index + 1))
  require_time_headroom
  require_mount_headroom
  output="$run_root/$output_rel"
  stream="$output/streams/output.dqvc"
  mkdir -p "$output/streams"
  echo "TASK $task_index/$task_total $task_id"
  if [[ ! -s "$output/encode_summary.json" || ! -s "$stream" ]]; then
    "$python_bin" demo/stage_c_spatial_quality_codec.py encode \
      --gate-summary "$gate" --route-summary "$route" \
      --output-stream "$stream" --output-dir "$output" \
      --frame-count 17 --cell-size 64 \
      --generate-qp 8 --base-qp 16 --enhance-qp 32 \
      --model-path-i "$model_i" --model-path-p "$model_p" \
      --checkpoint-role "$checkpoint_role"
  fi
  if [[ ! -s "$output/decode_summary.json" ]]; then
    "$python_bin" demo/stage_c_spatial_quality_codec.py decode \
      --input-stream "$stream" --output-dir "$output" \
      --model-path-i "$model_i" --model-path-p "$model_p" \
      --checkpoint-role "$checkpoint_role"
  fi
  if [[ ! -s "$output/fresh_decode_regression.json" ]]; then
    "$python_bin" demo/stage_c_a800_compare_frames.py \
      --reference-dir "$output/encoder_reconstruction" \
      --candidate-dir "$output/fresh_decode" --expected-frames 17 \
      --output "$output/fresh_decode_regression.json"
  fi
  "$python_bin" - "$run_root/progress.json" "$task_index" "$task_total" "$task_id" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
path = Path(sys.argv[1])
value = {
    "completed_tasks": int(sys.argv[2]),
    "total_tasks": int(sys.argv[3]),
    "last_completed_task": sys.argv[4],
    "updated_utc": datetime.now(timezone.utc).isoformat(),
}
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
done < "$tasks_tsv"

"$python_bin" demo/stage_c_spatial_qp_balanced_eval.py summarize \
  --protocol "$protocol" --endpoint-summary "$endpoint_summary" \
  --output-dir "$run_root"
cleanup
printf 'complete\n' > "$run_root/run.complete.tmp"
mv "$run_root/run.complete.tmp" "$run_root/run.complete"
rm -f "$run_root/run.failed"

du -sb "$run_root"
df -h /root /root/autodl-tmp /root/autodl-fs
echo "COMPLETE spatial_qp_balanced_eval utc=$(date -u +%FT%TZ) elapsed_s=$(($(date +%s) - experiment_start_epoch))"
exec > "$log_root/finalization.log" 2>&1
wait "$tee_pid"
"$python_bin" demo/stage_c_spatial_qp_balanced_eval.py resource-snapshot \
  --output-dir "$run_root"
trap - EXIT INT TERM
