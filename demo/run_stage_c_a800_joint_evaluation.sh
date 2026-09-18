#!/usr/bin/env bash
set -Eeuo pipefail

repo=/root/autodl-tmp/adaptive-chunk-coding
fast_root=/root/autodl-tmp/DCVC
env_root="$fast_root/envs/dcvcuf"
persist=/root/autodl-fs/DCVC
baseline_root="$persist/runs/a800_pilot_20260917"
v1_followup_root="$persist/runs/a800_followup_20260917"
v5_root="$persist/runs/a800_low_budget_v5_20260918"
independent_root="$persist/runs/a800_independent_test_20260918"
run_root="$persist/runs/a800_joint_evaluation_20260918"
formal_root="$run_root/formal/evaluation"
routes_root="$run_root/routes"
manifest_root="$run_root/manifests"
manifest="$manifest_root/joint_samples.jsonl"
ledger_summary="$manifest_root/summary.json"
frozen_controller="$run_root/frozen_controller.json"
log_file="$run_root/logs/joint_evaluation.log"
started_marker="$run_root/run_started_epoch.txt"
invocation_start_epoch=$(date +%s)
max_wall_seconds=43200

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

mkdir -p "$run_root/logs" "$formal_root" "$routes_root" "$manifest_root"
if [[ ! -s "$started_marker" ]]; then
  date +%s > "$started_marker.tmp"
  mv "$started_marker.tmp" "$started_marker"
fi
experiment_start_epoch=$(<"$started_marker")
cp "$0" "$run_root/logs/executed_run_stage_c_a800_joint_evaluation.sh"
exec > >(tee -a "$log_file") 2>&1

heartbeat() {
  while true; do
    local now_epoch completed gates batches run_bytes marker
    now_epoch=$(date +%s)
    completed=0
    for marker in "$formal_root"/*/sample.complete; do
      if [[ -e "$marker" ]]; then completed=$((completed + 1)); fi
    done
    gates=0
    for marker in "$formal_root"/*/uniform_gate/summary.json; do
      if [[ -s "$marker" ]]; then gates=$((gates + 1)); fi
    done
    batches=0
    for marker in \
      "$formal_root"/*/spatial/final-joint/roi/roi_batch_metadata.json \
      "$formal_root"/*/spatial/final-no-enhance/roi/roi_batch_metadata.json; do
      if [[ -s "$marker" ]]; then batches=$((batches + 1)); fi
    done
    run_bytes=$(du -sb "$run_root" 2>/dev/null | awk '{print $1}')
    echo "HEARTBEAT joint_evaluation utc=$(date -u +%FT%TZ) experiment_elapsed_s=$((now_epoch - experiment_start_epoch)) invocation_elapsed_s=$((now_epoch - invocation_start_epoch)) gates=$gates/37 samples=$completed/37 persistent_batches=$batches run_bytes=${run_bytes:-0}"
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

check_limits() {
  local elapsed used path
  elapsed=$(($(date +%s) - experiment_start_epoch))
  if [[ "$elapsed" -ge "$max_wall_seconds" ]]; then
    echo "ERROR experiment wall limit reached: $elapsed seconds" >&2
    return 1
  fi
  for path in /root /root/autodl-tmp /root/autodl-fs; do
    used=$(df -P "$path" | awk 'NR==2 {gsub(/%/, "", $5); print $5}')
    if [[ "$used" -ge 80 ]]; then
      echo "ERROR disk limit reached: path=$path used_percent=$used" >&2
      return 1
    fi
  done
}

ensure_gate_link() {
  local target=$1
  local link=$2
  if [[ -L "$link" ]]; then
    [[ "$(readlink -f "$link")" == "$(readlink -f "$target")" ]] || {
      echo "ERROR gate link points elsewhere: $link" >&2
      return 1
    }
    return
  fi
  if [[ -e "$link" ]]; then
    return
  fi
  [[ -s "$target/summary.json" ]]
  ln -s "$target" "$link"
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

run_full_seed() {
  local label=$1
  local input_dir=$2
  local output_dir=$3
  local seed=$4
  run_if_missing "$output_dir/seedvr2_metadata.json" "$label SeedVR2" \
    timeout 1800s torchrun --standalone --nproc-per-node=1 \
      demo/stage_c_seedvr2_bridge.py \
      --input-dir "$input_dir" --output-dir "$output_dir" \
      --dit-checkpoint third_party/SeedVR2/ckpts/seedvr2_ema_3b_bf16.safetensors \
      --seed "$seed" --sample-steps 1 --cfg-scale 1.0 --dit-dtype bfloat16
}

run_generic_evaluation() {
  local sample_id=$1
  local variant=$2
  local gate_summary=$3
  local route_summary=$4
  local frames_dir=$5
  local metadata=${6:-}
  local variant_root="$formal_root/$sample_id/spatial/$variant"
  local command=(
    python demo/stage_c_a800_evaluate_variant.py
    --gate-summary "$gate_summary"
    --route-summary "$route_summary"
    --codec-dir "$variant_root/codec"
    --frames-dir "$frames_dir"
    --variant-label "$variant"
    --output-dir "$variant_root/evaluation"
    --visual-frame 9
  )
  if [[ -n "$metadata" ]]; then
    command+=(--restoration-metadata "$metadata")
  fi
  run_if_missing "$variant_root/evaluation/evaluation.json" \
    "$sample_id/$variant evaluate" timeout 900s "${command[@]}"
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

echo "START joint_evaluation utc=$(date -u +%FT%TZ)"
cd "$repo"
[[ -s "$frozen_controller" ]]
[[ -s "$v5_root/formal/low_budget_v5_summary.json" ]]
[[ -s "$baseline_root/pilot/controller/controller.pt" ]]

mapfile -t controller_fields < <(python - "$frozen_controller" <<'PY'
import json
import sys
from pathlib import Path
value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for name in ("kind", "name", "checkpoint", "route_variant", "git_commit"):
    print(value[name])
PY
)
controller_kind=${controller_fields[0]}
controller_name=${controller_fields[1]}
controller_checkpoint=${controller_fields[2]}
route_variant=${controller_fields[3]}
frozen_git_commit=${controller_fields[4]}
[[ "$controller_kind" == v1 || "$controller_kind" == v5 ]]
[[ -s "$controller_checkpoint" ]]

run_if_missing "$ledger_summary" "frozen REDS plus UVG ledger" \
  python demo/stage_c_a800_joint_evaluation_manifest.py \
    --reds-root "$persist/data/REDS/val_sharp" \
    --uvg-root "$persist/assets/evaluation/UVG" \
    --controller-name "$controller_name" \
    --controller-checkpoint "$controller_checkpoint" \
    --git-commit "$frozen_git_commit" --output-dir "$manifest_root"

# Build or reuse every uniform-QP gate before v5 Base-probe routing.
while IFS=$'\t' read -r sample_id dataset sequence source_dir source_role crop_x crop_y; do
  check_limits
  sample_root="$formal_root/$sample_id"
  gate_root="$sample_root/uniform_gate"
  mkdir -p "$sample_root"
  if [[ "$sample_id" == dev-s* ]]; then
    ensure_gate_link \
      "$baseline_root/formal/development/$sample_id/uniform_gate" "$gate_root"
  elif [[ "$sample_id" == test-s* ]]; then
    ensure_gate_link \
      "$independent_root/formal/test/$sample_id/uniform_gate" "$gate_root"
  else
    run_if_missing "$gate_root/summary.json" "$sample_id uniform gate" \
      timeout 1800s python demo/stage_c_restorer_gate.py \
        --sequence-name "$sample_id" --source-dir "$source_dir" \
        --source-role "$source_role" --output-dir "$gate_root" \
        --frame-count 17 --width 512 --height 512 \
        --crop-x "$crop_x" --crop-y "$crop_y" --qps 8 16 24 32 \
        --decode-repeats 3 --visual-frame 9
  fi
done < <(python - "$manifest" <<'PY'
import json
import sys
from pathlib import Path
for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    item = json.loads(line)
    crop = item["crop"]
    print("\t".join(map(str, (
        item["sample_id"], item["dataset"], item["sequence"],
        item["source_dir"], item["source_role"], crop["x"], crop["y"],
    ))))
PY
)

if [[ "$controller_kind" == v1 ]]; then
  run_if_missing "$routes_root/controller/manifest.json" \
    "frozen v1 routes for all joint samples" \
    python demo/stage_c_a800_controller.py route \
      --sample-manifest "$manifest" --checkpoint "$controller_checkpoint" \
      --output-dir "$routes_root/controller" --method mlp \
      --byte-budget-ratios 0.25 --generate-budget-tiles 4
else
  run_if_missing "$routes_root/controller/manifest.json" \
    "frozen v5 routes for all joint samples" \
    python demo/stage_c_a800_low_budget_v5.py route \
      --sample-manifest "$manifest" --base-probe-root "$formal_root" \
      --checkpoint "$controller_checkpoint" \
      --output-dir "$routes_root/controller"
fi

warm_seedvr2_file_cache

while IFS=$'\t' read -r sample_id seed; do
  check_limits
  sample_root="$formal_root/$sample_id"
  gate_summary="$sample_root/uniform_gate/summary.json"
  controller_route="$routes_root/controller/$sample_id.json"
  route_root="$sample_root/routes"
  mkdir -p "$sample_root" "$route_root"
  if [[ -f "$sample_root/sample.complete" ]]; then
    echo "SKIP joint sample $sample_id"
    continue
  fi

  run_if_missing "$route_root/manifest.json" "$sample_id fixed route controls" \
    python demo/stage_c_a800_route_variants.py \
      --route-summary "$controller_route" \
      --source-variant "$route_variant" --output-dir "$route_root"

  run_if_missing "$sample_root/reuse.json" "$sample_id reuse compatible outputs" \
    python demo/stage_c_a800_joint_reuse.py \
      --controller-kind "$controller_kind" --sample-id "$sample_id" \
      --sample-root "$sample_root" --current-route "$route_root/learned-joint.json" \
      --baseline-root "$baseline_root" --v1-followup-root "$v1_followup_root" \
      --v5-root "$v5_root" --independent-root "$independent_root"

  route="$route_root/enhance-only.json"
  run_spatial_codec "$sample_id" "enhance-only" "$gate_summary" "$route"
  run_generic_evaluation "$sample_id" "enhance-only" "$gate_summary" \
    "$route" "$sample_root/spatial/enhance-only/codec/fresh_decode"

  route="$route_root/all-generate.json"
  run_spatial_codec "$sample_id" "all-generate" "$gate_summary" "$route"
  all_generate_seed="$sample_root/spatial/all-generate/seedvr2"
  run_full_seed "$sample_id all-generate" \
    "$sample_root/spatial/all-generate/codec/fresh_decode" \
    "$all_generate_seed" "$seed"
  run_generic_evaluation "$sample_id" "all-generate" "$gate_summary" \
    "$route" "$all_generate_seed" \
    "$all_generate_seed/seedvr2_metadata.json"

  run_persistent_roi_variant "$sample_id" "final-joint" "$gate_summary" \
    "$route_root/learned-joint.json" "$seed"

  run_spatial_codec "$sample_id" "final-no-generate" "$gate_summary" \
    "$route_root/same-route-no-generate.json"
  run_generic_evaluation "$sample_id" "final-no-generate" "$gate_summary" \
    "$route_root/same-route-no-generate.json" \
    "$sample_root/spatial/final-no-generate/codec/fresh_decode"

  run_persistent_roi_variant "$sample_id" "final-no-enhance" "$gate_summary" \
    "$route_root/same-route-no-enhance.json" "$seed"

  run_if_missing "$sample_root/visuals/manifest.json" \
    "$sample_id fixed joint-evaluation visual" \
    python demo/stage_c_a800_joint_evaluation_visual.py \
      --sample-root "$sample_root" --visual-frame 9

  touch "$sample_root/sample.complete"
  echo "COMPLETE joint sample $sample_id utc=$(date -u +%FT%TZ)"
done < <(python - "$manifest" <<'PY'
import json
import sys
from pathlib import Path
for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    item = json.loads(line)
    print(f"{item['sample_id']}\t{item['seed']}")
PY
)

run_if_missing "$run_root/formal/joint_evaluation_summary.json" \
  "REDS plus UVG joint aggregate" \
  python demo/stage_c_a800_joint_evaluation_summary.py \
    --formal-root "$formal_root" --routes-root "$routes_root" \
    --sample-manifest "$manifest" --ledger-summary "$ledger_summary" \
    --frozen-controller "$frozen_controller" \
    --output "$run_root/formal/joint_evaluation_summary.json"

touch "$run_root/formal_evaluation.complete"
experiment_wall_seconds=$(($(date +%s) - experiment_start_epoch))
cleanup
trap - EXIT
run_if_missing "$run_root/final_resource_snapshot.json" \
  "joint-evaluation final resource snapshot" \
  python demo/stage_c_a800_joint_evaluation_finalize.py \
    --run-root "$run_root" --wall-seconds "$experiment_wall_seconds" \
    --output "$run_root/final_resource_snapshot.json"

du -sb "$run_root"
df -h /root /root/autodl-tmp /root/autodl-fs
echo "COMPLETE joint_evaluation utc=$(date -u +%FT%TZ) experiment_elapsed_s=$experiment_wall_seconds invocation_elapsed_s=$(($(date +%s) - invocation_start_epoch))"
