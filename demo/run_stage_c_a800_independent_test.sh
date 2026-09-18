#!/usr/bin/env bash
set -Eeuo pipefail

repo=/root/autodl-tmp/adaptive-chunk-coding
fast_root=/root/autodl-tmp/DCVC
env_root="$fast_root/envs/dcvcuf"
persist=/root/autodl-fs/DCVC
pilot_root="$persist/runs/a800_pilot_20260917"
run_root="$persist/runs/a800_independent_test_20260918"
formal_root="$run_root/formal/test"
routes_root="$run_root/routes"
manifest_root="$run_root/manifests"
manifest="$manifest_root/test_samples.jsonl"
controller="$pilot_root/pilot/controller/controller.pt"
log_file="$run_root/logs/independent_test.log"
started_marker="$run_root/run_started_epoch.txt"
invocation_start_epoch=$(date +%s)
max_wall_seconds=41400

export TMPDIR="$fast_root/tmp"
export PIP_CACHE_DIR="$fast_root/cache/pip"
export TORCH_EXTENSIONS_DIR="$fast_root/torch_extensions"
export HF_HOME="$persist/cache/huggingface"
export TORCH_HOME="$persist/cache/torch"
export CUDA_HOME="$env_root"
export PATH="$env_root/bin:$PATH"
cuda_wheel_root="$env_root/lib/python3.12/site-packages/nvidia/cu13"
export LD_LIBRARY_PATH="$cuda_wheel_root/lib:$env_root/targets/x86_64-linux/lib:$env_root/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0

mkdir -p "$run_root/logs" "$formal_root" "$routes_root" "$manifest_root"
if [[ ! -s "$started_marker" ]]; then
  temporary="$started_marker.tmp"
  date +%s > "$temporary"
  mv "$temporary" "$started_marker"
fi
experiment_start_epoch=$(<"$started_marker")
cp "$0" "$run_root/logs/executed_run_a800_independent_test.sh"
exec > >(tee -a "$log_file") 2>&1

heartbeat() {
  while true; do
    now_epoch=$(date +%s)
    completed=$(find "$formal_root" -mindepth 2 -maxdepth 2 -name sample.complete 2>/dev/null | wc -l)
    batches=$(find "$formal_root" -name roi_batch_metadata.json 2>/dev/null | wc -l)
    run_bytes=$(du -sb "$run_root" 2>/dev/null | awk '{print $1}')
    echo "HEARTBEAT independent_test utc=$(date -u +%FT%TZ) experiment_elapsed_s=$((now_epoch - experiment_start_epoch)) invocation_elapsed_s=$((now_epoch - invocation_start_epoch)) samples=$completed/12 persistent_batches=$batches/48 run_bytes=${run_bytes:-0}"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
    df -h /root /root/autodl-tmp /root/autodl-fs
    sleep 60
  done
}

heartbeat &
heartbeat_pid=$!
cleanup() {
  kill "$heartbeat_pid" 2>/dev/null || true
  wait "$heartbeat_pid" 2>/dev/null || true
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
path = Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8"))
if path.name == "final_resource_snapshot.json" and value.get("status") != "complete":
    raise SystemExit(1)
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
  local elapsed=$(($(date +%s) - experiment_start_epoch))
  if [[ "$elapsed" -ge "$max_wall_seconds" ]]; then
    echo "ERROR experiment wall limit reached: $elapsed seconds" >&2
    return 1
  fi
  local path used
  for path in /root /root/autodl-tmp /root/autodl-fs; do
    used=$(df -P "$path" | awk 'NR==2 {gsub(/%/, "", $5); print $5}')
    if [[ "$used" -ge 80 ]]; then
      echo "ERROR disk limit reached: path=$path used_percent=$used" >&2
      return 1
    fi
  done
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
  run_if_missing "$codec_root/decode_summary.json" "$sample_id/$variant fresh-decode" \
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

echo "START independent_test utc=$(date -u +%FT%TZ)"
cd "$repo"
[[ -s "$run_root/data_restore.complete.json" ]]
[[ -s "$controller" ]]
[[ -s "$pilot_root/formal/formal_summary.json" ]]
for sequence in $(seq -f '%03g' 24 29); do
  [[ ! -e "$persist/data/REDS/val_sharp/$sequence" ]]
done

run_if_missing "$manifest_root/summary.json" "fixed independent-test ledger" \
  python demo/stage_c_a800_independent_manifest.py \
    --validation-root "$persist/data/REDS/val_sharp" \
    --output-dir "$manifest_root"

run_if_missing "$routes_root/controller/manifest.json" \
  "frozen controller routes before quality evaluation" \
  python demo/stage_c_a800_controller.py route \
    --sample-manifest "$manifest" --checkpoint "$controller" \
    --output-dir "$routes_root/controller" --method mlp \
    --byte-budget-ratios 0.25 0.5 1.0 \
    --generate-budget-tiles 4 8 12

warmup_root="$run_root/warmup/invocation_$invocation_start_epoch"
mkdir -p "$warmup_root"
echo "START_STEP excluded one-frame SeedVR2 cache warmup utc=$(date -u +%FT%TZ)"
timeout 1800s torchrun --standalone --nproc-per-node=1 \
  demo/stage_c_seedvr2_bridge.py \
  --input-dir "$pilot_root/formal/development/dev-s000-f00-x384-y096/uniform_gate/frames/qp8-base" \
  --output-dir "$warmup_root" --max-frames 1 \
  --dit-checkpoint third_party/SeedVR2/ckpts/seedvr2_ema_3b_bf16.safetensors \
  --seed 20260918 --sample-steps 1 --cfg-scale 1.0 --dit-dtype bfloat16
[[ -s "$warmup_root/seedvr2_metadata.json" ]]
echo "COMPLETE_STEP excluded cache warmup utc=$(date -u +%FT%TZ)"

while IFS=$'\t' read -r sample_id sequence source_dir crop_x crop_y seed; do
  check_limits
  sample_root="$formal_root/$sample_id"
  gate_root="$sample_root/uniform_gate"
  gate_summary="$gate_root/summary.json"
  controller_route="$routes_root/controller/$sample_id.json"
  mkdir -p "$sample_root"
  if [[ -f "$sample_root/sample.complete" ]]; then
    echo "SKIP independent sample $sample_id"
    continue
  fi

  run_if_missing "$gate_summary" "$sample_id uniform gate" \
    timeout 1800s python demo/stage_c_restorer_gate.py \
      --sequence-name "$sample_id" --source-dir "$source_dir" \
      --source-role "one-shot independent test; REDS val/012..023; consumed after this evaluation and never used for tuning" \
      --output-dir "$gate_root" --frame-count 17 --width 512 --height 512 \
      --crop-x "$crop_x" --crop-y "$crop_y" --qps 8 16 24 32 \
      --decode-repeats 3 --visual-frame 9

  for qp in 8 16 24 32; do
    for restorer in none basicvsrpp; do
      if [[ "$restorer" == none ]]; then
        benchmark_name="scalar-qp$qp"
      else
        benchmark_name="basicvsrpp-qp$qp"
      fi
      run_if_missing "$gate_root/isolated_fresh_decode/$benchmark_name.json" \
        "$sample_id $benchmark_name isolated fresh-decode" \
        timeout 900s python demo/stage_c_a800_scalar_fresh_decode.py \
          --stream "$gate_root/streams/all_base_qp$qp.dcvc" \
          --frame-count 17 --decode-repeats 1 --restorer "$restorer" \
          --output-frames-dir "$gate_root/isolated_fresh_decode/frames/$benchmark_name" \
          --output "$gate_root/isolated_fresh_decode/$benchmark_name.json"
    done
  done

  for budget in low middle; do
    if [[ "$budget" == low ]]; then
      source_variant=mlp-budget-0
    else
      source_variant=mlp-budget-1
    fi
    run_if_missing "$sample_root/routes/$budget/manifest.json" \
      "$sample_id $budget fixed route controls" \
      python demo/stage_c_a800_route_variants.py \
        --route-summary "$controller_route" \
        --source-variant "$source_variant" \
        --output-dir "$sample_root/routes/$budget"
  done

  route="$sample_root/routes/low/enhance-only.json"
  run_spatial_codec "$sample_id" "enhance-only" "$gate_summary" "$route"
  run_generic_evaluation "$sample_id" "enhance-only" "$gate_summary" \
    "$route" "$sample_root/spatial/enhance-only/codec/fresh_decode"

  route="$sample_root/routes/low/all-generate.json"
  run_spatial_codec "$sample_id" "all-generate" "$gate_summary" "$route"
  all_generate_seed="$sample_root/spatial/all-generate/seedvr2"
  run_full_seed "$sample_id all-generate" \
    "$sample_root/spatial/all-generate/codec/fresh_decode" \
    "$all_generate_seed" "$seed"
  run_generic_evaluation "$sample_id" "all-generate" "$gate_summary" \
    "$route" "$all_generate_seed" \
    "$all_generate_seed/seedvr2_metadata.json"

  for budget in low middle; do
    route_root="$sample_root/routes/$budget"
    run_persistent_roi_variant "$sample_id" "$budget-joint" \
      "$gate_summary" "$route_root/learned-joint.json" "$seed"

    run_spatial_codec "$sample_id" "$budget-no-generate" "$gate_summary" \
      "$route_root/same-route-no-generate.json"
    run_generic_evaluation "$sample_id" "$budget-no-generate" \
      "$gate_summary" "$route_root/same-route-no-generate.json" \
      "$sample_root/spatial/$budget-no-generate/codec/fresh_decode"

    run_persistent_roi_variant "$sample_id" "$budget-no-enhance" \
      "$gate_summary" "$route_root/same-route-no-enhance.json" "$seed"
  done

  run_if_missing "$sample_root/visuals/manifest.json" \
    "$sample_id fixed independent visual" \
    python demo/stage_c_a800_independent_visual.py \
      --sample-root "$sample_root" --visual-frame 9

  touch "$sample_root/sample.complete"
  echo "COMPLETE independent sample $sample_id utc=$(date -u +%FT%TZ)"
done < <(python - "$manifest" <<'PY'
import json
import sys
from pathlib import Path
for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    item = json.loads(line)
    crop = item["crop"]
    print("\t".join(map(str, (
        item["sample_id"], item["sequence"], item["source_dir"],
        crop["x"], crop["y"], item["seed"],
    ))))
PY
)

run_if_missing "$run_root/formal/independent_summary.json" \
  "independent-test aggregate" \
  python demo/stage_c_a800_independent_summary.py \
    --formal-root "$formal_root" --routes-root "$routes_root" \
    --sample-manifest "$manifest" --controller-checkpoint "$controller" \
    --output "$run_root/formal/independent_summary.json"

touch "$run_root/formal_evaluation.complete"
experiment_wall_seconds=$(($(date +%s) - experiment_start_epoch))
# Stop the live heartbeat before taking the final resource snapshot.  The tee
# log still receives the final report, so the snapshot explicitly excludes it.
cleanup
trap - EXIT
run_if_missing "$run_root/final_resource_snapshot.json" \
  "independent-test resource snapshot" \
  python demo/stage_c_a800_independent_finalize.py \
    --run-root "$run_root" --wall-seconds "$experiment_wall_seconds" \
    --output "$run_root/final_resource_snapshot.json"

du -sb "$run_root"
df -h /root /root/autodl-tmp /root/autodl-fs
echo "COMPLETE independent_test utc=$(date -u +%FT%TZ) experiment_elapsed_s=$experiment_wall_seconds invocation_elapsed_s=$(($(date +%s) - invocation_start_epoch))"
