#!/usr/bin/env bash
set -Eeuo pipefail

repo=/root/autodl-tmp/adaptive-chunk-coding
fast_root=/root/autodl-tmp/DCVC
env_root="$fast_root/envs/dcvcuf"
persist=/root/autodl-fs/DCVC
baseline_root="$persist/runs/a800_pilot_20260917"
v1_followup_root="$persist/runs/a800_followup_20260917"
run_root="$persist/runs/a800_low_budget_v4_20260918"
formal_root="$run_root/formal/development"
routes_root="$run_root/routes"
log_file="$run_root/logs/formal.log"
started_marker="$run_root/formal_started_epoch.txt"
invocation_start_epoch=$(date +%s)

export TMPDIR="$fast_root/tmp"
export PIP_CACHE_DIR="$fast_root/cache/pip"
export TORCH_EXTENSIONS_DIR="$fast_root/torch_extensions"
export HF_HOME="$persist/cache/huggingface"
export TORCH_HOME="$persist/cache/torch"
export PATH="$env_root/bin:/usr/local/cuda/bin:$PATH"
cuda_wheel_root="$env_root/lib/python3.12/site-packages/nvidia/cu13"
export LD_LIBRARY_PATH="$cuda_wheel_root/lib:$env_root/targets/x86_64-linux/lib:$env_root/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$repo:/root/autodl-tmp/DCVC/DCVC:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0

mkdir -p "$run_root/logs" "$formal_root"
if [[ ! -s "$started_marker" ]]; then
  date +%s > "$started_marker.tmp"
  mv "$started_marker.tmp" "$started_marker"
fi
experiment_start_epoch=$(<"$started_marker")
cp "$0" "$run_root/logs/executed_run_stage_c_a800_low_budget_v4_formal.sh"
exec > >(tee -a "$log_file") 2>&1

heartbeat() {
  while true; do
    local now_epoch completed batches
    now_epoch=$(date +%s)
    completed=$(find "$formal_root" -mindepth 2 -maxdepth 2 \
      -name sample.complete 2>/dev/null | wc -l)
    batches=$(find "$formal_root" -name roi_batch_metadata.json \
      2>/dev/null | wc -l)
    echo "HEARTBEAT low_budget_v4_formal utc=$(date -u +%FT%TZ) experiment_elapsed_s=$((now_epoch - experiment_start_epoch)) invocation_elapsed_s=$((now_epoch - invocation_start_epoch)) samples=$completed/6 persistent_batches=$batches/12"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits || true
    df -h /root /root/autodl-tmp /root/autodl-fs
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

heartbeat &
heartbeat_pid=$!
gpu_monitor >> "$run_root/logs/gpu_samples.csv" 2>&1 &
gpu_monitor_pid=$!
cleanup() {
  kill "$heartbeat_pid" "$gpu_monitor_pid" 2>/dev/null || true
  wait "$heartbeat_pid" "$gpu_monitor_pid" 2>/dev/null || true
}
trap cleanup EXIT

valid_marker() {
  local marker=$1
  [[ -s "$marker" ]] || return 1
  if [[ "$marker" == *.json ]]; then
    python - "$marker" <<'PY' >/dev/null
import json
import sys
from pathlib import Path
json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
PY
  fi
}

run_if_missing() {
  local marker=$1
  local label=$2
  shift 2
  if valid_marker "$marker"; then
    echo "SKIP $label marker=$marker"
    return
  fi
  echo "START_STEP $label utc=$(date -u +%FT%TZ)"
  "$@"
  if ! valid_marker "$marker"; then
    echo "ERROR missing marker after $label: $marker" >&2
    return 1
  fi
  echo "COMPLETE_STEP $label utc=$(date -u +%FT%TZ)"
}

warm_seedvr2_file_cache() {
  python - "$repo/third_party/SeedVR2/ckpts/seedvr2_ema_3b_bf16.safetensors" <<'PY'
import json
import sys
import time
from pathlib import Path

checkpoint = Path(sys.argv[1]).resolve()
started = time.perf_counter()
byte_count = 0
with checkpoint.open("rb") as handle:
    while block := handle.read(64 * 1024 * 1024):
        byte_count += len(block)
print(json.dumps({
    "stage": "seedvr2-file-cache-warmup",
    "checkpoint": str(checkpoint),
    "bytes_read": byte_count,
    "wall_seconds_not_charged_to_variant_runtime": time.perf_counter() - started,
}))
PY
}

run_spatial_codec() {
  local sample_id=$1
  local variant=$2
  local gate_summary=$3
  local route_summary=$4
  local variant_root="$formal_root/$sample_id/spatial/$variant"
  local codec_root="$variant_root/codec"
  mkdir -p "$codec_root"
  run_if_missing "$codec_root/encode_summary.json" "$sample_id/$variant encode" \
    timeout 900s python demo/stage_c_spatial_quality_codec.py encode \
      --gate-summary "$gate_summary" \
      --route-summary "$route_summary" \
      --output-stream "$codec_root/stream.dcvc-sq" \
      --output-dir "$codec_root" --frame-count 17
  run_if_missing "$codec_root/decode_summary.json" \
    "$sample_id/$variant fresh-decode" \
    timeout 900s python demo/stage_c_spatial_quality_codec.py decode \
      --input-stream "$codec_root/stream.dcvc-sq" \
      --output-dir "$codec_root"
}

run_generic_evaluation() {
  local sample_id=$1
  local variant=$2
  local gate_summary=$3
  local route_summary=$4
  local variant_root="$formal_root/$sample_id/spatial/$variant"
  run_if_missing "$variant_root/evaluation/evaluation.json" \
    "$sample_id/$variant evaluate" \
    timeout 900s python demo/stage_c_a800_evaluate_variant.py \
      --gate-summary "$gate_summary" \
      --route-summary "$route_summary" \
      --codec-dir "$variant_root/codec" \
      --frames-dir "$variant_root/codec/fresh_decode" \
      --variant-label "$variant" \
      --output-dir "$variant_root/evaluation" --visual-frame 9
}

run_persistent_roi_variant() {
  local sample_id=$1
  local variant=$2
  local gate_summary=$3
  local route_summary=$4
  local seed=$5
  local variant_root="$formal_root/$sample_id/spatial/$variant"
  local codec_root="$variant_root/codec"
  local roi_root="$variant_root/roi"
  run_spatial_codec "$sample_id" "$variant" "$gate_summary" "$route_summary"
  run_if_missing "$roi_root/manifest.json" "$sample_id/$variant ROI prepare" \
    python demo/stage_c_seedvr2_roi.py prepare \
      --route-summary "$route_summary" \
      --input-dir "$codec_root/fresh_decode" \
      --output-dir "$roi_root" --context-pixels 64 --processing-scale 1.0
  run_if_missing "$roi_root/roi_batch_metadata.json" \
    "$sample_id/$variant persistent ROI restore" \
    timeout 2400s torchrun --standalone --nproc-per-node=1 \
      demo/stage_c_seedvr2_roi.py restore \
      --manifest "$roi_root/manifest.json" --output-root "$roi_root" \
      --dit-checkpoint third_party/SeedVR2/ckpts/seedvr2_ema_3b_bf16.safetensors \
      --seed "$seed" --sample-steps 1 --cfg-scale 1.0 --dit-dtype bfloat16
  run_if_missing "$variant_root/evaluation/summary.json" \
    "$sample_id/$variant ROI evaluate" \
    timeout 900s python demo/stage_c_seedvr2_roi.py evaluate \
      --manifest "$roi_root/manifest.json" \
      --gate-summary "$gate_summary" --codec-dir "$codec_root" \
      --restored-root "$roi_root" --output-dir "$variant_root/evaluation" \
      --feather-pixels 8
}

echo "START low_budget_v4_formal utc=$(date -u +%FT%TZ)"
cd "$repo"
[[ -e "$run_root/TRAINING_AND_ROUTES_COMPLETE" ]]
[[ -s "$run_root/selection/selection_summary.json" ]]
[[ -s "$run_root/controller/controller_v4.pt" ]]
[[ -s "$routes_root/manifest.json" ]]
[[ -s "$baseline_root/formal/formal_summary.json" ]]
[[ -s "$v1_followup_root/formal/followup_summary.json" ]]

warm_seedvr2_file_cache

while IFS=$'\t' read -r sample_id seed; do
  sample_root="$formal_root/$sample_id"
  baseline_sample="$baseline_root/formal/development/$sample_id"
  v1_sample="$v1_followup_root/formal/development/$sample_id"
  gate_summary="$baseline_sample/uniform_gate/summary.json"
  route_root="$sample_root/routes"
  mkdir -p "$sample_root"
  if [[ -f "$sample_root/sample.complete" ]]; then
    echo "SKIP v4 formal sample $sample_id"
    continue
  fi

  run_if_missing "$route_root/manifest.json" "$sample_id v4 route controls" \
    python demo/stage_c_a800_route_variants.py \
      --route-summary "$routes_root/$sample_id.json" \
      --source-variant anchored-hybrid-low-v4 --output-dir "$route_root"

  run_persistent_roi_variant "$sample_id" "low-joint" "$gate_summary" \
    "$route_root/learned-joint.json" "$seed"

  run_spatial_codec "$sample_id" "low-no-generate" "$gate_summary" \
    "$route_root/same-route-no-generate.json"
  run_generic_evaluation "$sample_id" "low-no-generate" "$gate_summary" \
    "$route_root/same-route-no-generate.json"

  run_persistent_roi_variant "$sample_id" "low-no-enhance" "$gate_summary" \
    "$route_root/same-route-no-enhance.json" "$seed"

  run_if_missing "$sample_root/visuals/manifest.json" \
    "$sample_id fixed v4 visual" \
    python demo/stage_c_a800_low_budget_v4_visual.py \
      --baseline-sample-root "$baseline_sample" \
      --v1-followup-sample-root "$v1_sample" \
      --v4-sample-root "$sample_root" --visual-frame 9

  touch "$sample_root/sample.complete"
  echo "COMPLETE v4 formal sample $sample_id utc=$(date -u +%FT%TZ)"
done < <(python - "$baseline_root/pilot/sample_ledger/development_samples.jsonl" <<'PY'
import json
import sys
from pathlib import Path
for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    item = json.loads(line)
    print(f"{item['sample_id']}\t{item['seed']}")
PY
)

run_if_missing "$run_root/formal/low_budget_v4_summary.json" \
  "low-budget v4 aggregate" \
  python demo/stage_c_a800_low_budget_v4_summary.py \
    --v1-followup-summary "$v1_followup_root/formal/followup_summary.json" \
    --selection-summary "$run_root/selection/selection_summary.json" \
    --formal-root "$formal_root" --routes-root "$routes_root" \
    --output "$run_root/formal/low_budget_v4_summary.json"

touch "$run_root/formal_evaluation.complete"
experiment_wall_seconds=$(($(date +%s) - experiment_start_epoch))
run_if_missing "$run_root/final_resource_snapshot.json" \
  "low-budget v4 final resource snapshot" \
  python demo/stage_c_a800_low_budget_v4_finalize.py \
    --run-root "$run_root" --wall-seconds "$experiment_wall_seconds" \
    --output "$run_root/final_resource_snapshot.json"

du -sb "$run_root"
df -h /root /root/autodl-tmp /root/autodl-fs
echo "COMPLETE low_budget_v4_formal utc=$(date -u +%FT%TZ) experiment_elapsed_s=$experiment_wall_seconds invocation_elapsed_s=$(($(date +%s) - invocation_start_epoch))"
