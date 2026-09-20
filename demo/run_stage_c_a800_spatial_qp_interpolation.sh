#!/usr/bin/env bash
set -Eeuo pipefail

repo=/root/autodl-tmp/adaptive-chunk-coding
fast_root=/root/autodl-tmp/DCVC
env_root="$fast_root/envs/dcvcuf"
persist=/root/autodl-fs/DCVC
endpoint_root="$persist/runs/a800_spatial_qp_finetune_eval_20260920"
train_root="$persist/runs/a800_spatial_qp_finetune_v1_20260920"
run_root="$persist/runs/a800_spatial_qp_interpolation_20260920"
log_root="$run_root/logs"
python_bin="$env_root/bin/python"
endpoint_protocol="$endpoint_root/protocol/protocol.json"
endpoint_summary="$endpoint_root/summary.json"
frozen_i="$repo/checkpoints/cvpr2026_image.pth.tar"
frozen_p="$repo/checkpoints/cvpr2026_video_hts.pth.tar"
tuned_i="$train_root/training/checkpoints/image_model.pth.tar"
tuned_p="$train_root/training/checkpoints/video_model_hts.pth.tar"

export TMPDIR="$fast_root/tmp"
export PIP_CACHE_DIR="$fast_root/cache/pip"
export TORCH_EXTENSIONS_DIR="$fast_root/torch_extensions"
export HF_HOME="$persist/cache/huggingface"
export TORCH_HOME="$persist/cache/torch"
export CUDA_HOME="$env_root"
export PATH="$env_root/bin:/usr/local/cuda/bin:$PATH"
cuda_wheel_root="$env_root/lib/python3.12/site-packages/nvidia/cu13"
export LD_LIBRARY_PATH="$cuda_wheel_root/lib:$env_root/targets/x86_64-linux/lib:$env_root/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$repo:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0

mkdir -p "$log_root"
started_marker="$run_root/run_started_epoch.txt"
if [[ ! -s "$started_marker" ]]; then
  date +%s > "$started_marker.tmp"
  mv "$started_marker.tmp" "$started_marker"
fi
experiment_start_epoch=$(<"$started_marker")
cp "$0" "$log_root/executed_run_stage_c_a800_spatial_qp_interpolation.sh"
exec > >(tee -a "$log_root/spatial_qp_interpolation.log") 2>&1
tee_pid=$!

heartbeat_pid=
cleanup() {
  if [[ -n "$heartbeat_pid" ]]; then kill "$heartbeat_pid" 2>/dev/null || true; fi
  if [[ -n "$heartbeat_pid" ]]; then wait "$heartbeat_pid" 2>/dev/null || true; fi
}
fail() {
  status=$?
  cleanup
  if (( status != 0 )); then
    rm -f "$run_root/run.complete"
    printf '%s\n' "$status" > "$run_root/run.failed.tmp"
    mv "$run_root/run.failed.tmp" "$run_root/run.failed"
    echo "FAILED spatial_qp_interpolation status=$status utc=$(date -u +%FT%TZ)"
  fi
  exit "$status"
}
trap fail EXIT INT TERM
(
  while true; do
    printf 'HEARTBEAT spatial_qp_interpolation utc=%s elapsed_s=%s ' \
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

echo "START spatial_qp_interpolation utc=$(date -u +%FT%TZ)"
cd "$repo"
[[ -f "$endpoint_root/run.complete" ]]
[[ -s "$endpoint_protocol" ]]
[[ -s "$endpoint_summary" ]]
[[ -s "$frozen_i" && -s "$frozen_p" && -s "$tuned_i" && -s "$tuned_p" ]]
gpu_count=$(nvidia-smi -L | wc -l)
if [[ "$gpu_count" -ne 1 ]]; then
  echo "expected one visible GPU, found $gpu_count" >&2
  exit 3
fi

"$python_bin" demo/stage_c_spatial_qp_interpolation.py self-test
protocol="$run_root/protocol.json"
if [[ ! -s "$protocol" ]]; then
  "$python_bin" demo/stage_c_spatial_qp_interpolation.py prepare \
    --endpoint-protocol "$endpoint_protocol" \
    --endpoint-summary "$endpoint_summary" --output-dir "$run_root" \
    --frozen-image "$frozen_i" --frozen-video "$frozen_p" \
    --tuned-image "$tuned_i" --tuned-video "$tuned_p" \
    --alphas 0.25,0.50,0.75
fi

tasks_tsv="$run_root/tasks.tsv"
"$python_bin" demo/stage_c_spatial_qp_interpolation.py list-tasks \
  --protocol "$protocol" > "$tasks_tsv.tmp"
mv "$tasks_tsv.tmp" "$tasks_tsv"

task_index=0
task_total=$(wc -l < "$tasks_tsv")
while IFS=$'\t' read -r task_id sample_id family model_role alpha gate route \
    model_i model_p checkpoint_role output_rel; do
  task_index=$((task_index + 1))
  output="$run_root/$output_rel"
  stream="$output/streams/output.dqvc"
  mkdir -p "$output/streams"
  echo "TASK $task_index/$task_total $task_id alpha=$alpha"
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
done < "$tasks_tsv"

"$python_bin" demo/stage_c_spatial_qp_interpolation.py summarize \
  --protocol "$protocol" --endpoint-summary "$endpoint_summary" \
  --output-dir "$run_root"
cleanup
printf 'complete\n' > "$run_root/run.complete.tmp"
mv "$run_root/run.complete.tmp" "$run_root/run.complete"
rm -f "$run_root/run.failed"

du -sb "$run_root"
df -h /root /root/autodl-tmp /root/autodl-fs
echo "COMPLETE spatial_qp_interpolation utc=$(date -u +%FT%TZ) elapsed_s=$(($(date +%s) - experiment_start_epoch))"
exec > "$log_root/finalization.log" 2>&1
wait "$tee_pid"
"$python_bin" demo/stage_c_spatial_qp_finetune_eval.py resource-snapshot \
  --output-dir "$run_root"
trap - EXIT INT TERM
