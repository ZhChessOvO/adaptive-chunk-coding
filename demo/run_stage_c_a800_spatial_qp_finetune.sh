#!/usr/bin/env bash
set -Eeuo pipefail

mode=${1:-smoke}
if [[ "$mode" == smoke ]]; then
  default_run_root=/root/autodl-fs/DCVC/runs/a800_spatial_qp_finetune_smoke_20260920
  default_steps=2
  default_patch=256
  default_save_every=1
elif [[ "$mode" == train ]]; then
  default_run_root=/root/autodl-fs/DCVC/runs/a800_spatial_qp_finetune_v1_20260920
  default_steps=1000
  default_patch=512
  default_save_every=25
else
  echo "usage: $0 [smoke|train]" >&2
  exit 2
fi

repo=/root/autodl-tmp/adaptive-chunk-coding
fast_root=/root/autodl-tmp/DCVC
env_root="$fast_root/envs/dcvcuf"
persist=/root/autodl-fs/DCVC
run_root=${RUN_ROOT:-$default_run_root}
max_steps=${MAX_STEPS:-$default_steps}
patch_size=${PATCH_SIZE:-$default_patch}
save_every=${SAVE_EVERY:-$default_save_every}
training_root="$run_root/training"
validation_root="$run_root/validation_step_${max_steps}"
codec_root="$validation_root/codec"
log_root="$run_root/logs"
python_bin="$env_root/bin/python"
model_i="$repo/checkpoints/cvpr2026_image.pth.tar"
model_p="$repo/checkpoints/cvpr2026_video_hts.pth.tar"
trained_i="$training_root/checkpoints/image_model.pth.tar"
trained_p="$training_root/checkpoints/video_model_hts.pth.tar"
gate="$persist/runs/a800_long_video_smoke_20260920/plan/long_gate.json"
route="$persist/runs/a800_spatial_consistency_20260920/routes_v6_combined/dev-s000-f00-x384-y096.json"

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

mkdir -p "$log_root" "$codec_root"
rm -f "$run_root/run.complete"
started_marker="$run_root/run_started_epoch.txt"
if [[ ! -s "$started_marker" ]]; then
  date +%s > "$started_marker.tmp"
  mv "$started_marker.tmp" "$started_marker"
fi
experiment_start_epoch=$(<"$started_marker")
cp "$0" "$log_root/executed_run_stage_c_a800_spatial_qp_finetune.sh"
exec > >(tee -a "$log_root/spatial_qp_finetune_${mode}.log") 2>&1

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
    echo "FAILED spatial_qp_finetune mode=$mode status=$status utc=$(date -u +%FT%TZ)"
  fi
  exit "$status"
}
trap fail EXIT INT TERM
(
  while true; do
    printf 'HEARTBEAT spatial_qp_finetune mode=%s utc=%s elapsed_s=%s ' \
      "$mode" "$(date -u +%FT%TZ)" "$(($(date +%s) - experiment_start_epoch))"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits | tr '\n' ';' || true
    printf ' '
    df -h /root /root/autodl-tmp /root/autodl-fs | tail -n +2 | tr '\n' ';'
    printf '\n'
    sleep 60
  done
) >> "$log_root/heartbeat.log" 2>&1 &
heartbeat_pid=$!

echo "START spatial_qp_finetune mode=$mode utc=$(date -u +%FT%TZ) steps=$max_steps patch=$patch_size"
cd "$repo"
[[ -x "$python_bin" ]]
[[ -s "$model_i" ]]
[[ -s "$model_p" ]]
[[ -d "$persist/data/REDS/train_sharp" ]]
[[ -d "$persist/data/UVG_adaptation/samples" ]]
[[ -s "$gate" ]]
[[ -s "$route" ]]
gpu_count=$(nvidia-smi -L | wc -l)
if [[ "$gpu_count" -ne 1 ]]; then
  echo "expected one visible GPU, found $gpu_count" >&2
  exit 3
fi

"$python_bin" demo/stage_c_spatial_qp_finetune.py self-test
"$python_bin" demo/stage_c_spatial_qp_finetune.py train \
  --output-dir "$training_root" \
  --model-path-i "$model_i" --model-path-p "$model_p" \
  --reds-root "$persist/data/REDS/train_sharp" \
  --uvg-root "$persist/data/UVG_adaptation/samples" \
  --max-steps "$max_steps" --patch-size "$patch_size" \
  --learning-rate 2e-6 --weight-decay 1e-4 --max-grad-norm 0.2 \
  --uvg-probability 0.107 --uniform-probability 0.25 \
  --seed 21260920 --save-every "$save_every" --log-every 1 \
  --amp-dtype bfloat16
[[ -f "$training_root/training.complete" ]]
[[ -s "$trained_i" ]]
[[ -s "$trained_p" ]]

stream="$codec_root/streams/finetuned_spatial_qp.dqvc"
if [[ ! -s "$codec_root/encode_summary.json" || ! -s "$stream" ]]; then
  "$python_bin" demo/stage_c_spatial_quality_codec.py encode \
    --gate-summary "$gate" --route-summary "$route" \
    --output-stream "$stream" --output-dir "$codec_root" \
    --frame-count 17 --cell-size 64 \
    --generate-qp 8 --base-qp 16 --enhance-qp 32 \
    --model-path-i "$trained_i" --model-path-p "$trained_p" \
    --checkpoint-role spatial-qp-finetuned
fi
if [[ ! -s "$codec_root/decode_summary.json" ]]; then
  "$python_bin" demo/stage_c_spatial_quality_codec.py decode \
    --input-stream "$stream" --output-dir "$codec_root" \
    --model-path-i "$trained_i" --model-path-p "$trained_p" \
    --checkpoint-role spatial-qp-finetuned
fi

regression="$validation_root/fresh_decode_regression.json"
if [[ ! -s "$regression" ]]; then
  "$python_bin" demo/stage_c_a800_compare_frames.py \
    --reference-dir "$codec_root/encoder_reconstruction" \
    --candidate-dir "$codec_root/fresh_decode" \
    --expected-frames 17 --output "$regression"
fi

"$python_bin" demo/stage_c_spatial_qp_finetune.py finalize \
  --run-root "$run_root" --expected-steps "$max_steps" \
  --codec-dir "$codec_root" --regression "$regression" --mode "$mode"

rm -f "$run_root/run.failed"
cleanup
trap - EXIT INT TERM
du -sh "$run_root"
df -h /root /root/autodl-tmp /root/autodl-fs
echo "COMPLETE spatial_qp_finetune mode=$mode utc=$(date -u +%FT%TZ) elapsed_s=$(($(date +%s) - experiment_start_epoch))"
