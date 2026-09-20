#!/usr/bin/env bash
set -Eeuo pipefail

mode=${1:-}
case "$mode" in
  smoke)
    run_root=/root/autodl-fs/DCVC/runs/a800_seedvr2_lora_smoke_20260920
    cache_limit_args=(--limit-per-dataset 1)
    max_steps=2
    patch_size=128
    save_every=1
    ;;
  train)
    run_root=/root/autodl-fs/DCVC/runs/a800_seedvr2_lora_v1_20260920
    cache_limit_args=()
    max_steps=1000
    patch_size=256
    save_every=25
    ;;
  *)
    echo "usage: $0 smoke|train" >&2
    exit 2
    ;;
esac

repo=/root/autodl-tmp/adaptive-chunk-coding
fast_root=/root/autodl-tmp/DCVC
env_root="$fast_root/envs/dcvcuf"
persist=/root/autodl-fs/DCVC
interp_root="$persist/runs/a800_spatial_qp_interpolation_20260920"
endpoint_root="$persist/runs/a800_spatial_qp_finetune_eval_20260920"
codec_train_root="$persist/runs/a800_spatial_qp_finetune_v1_20260920"
sample_manifest="$persist/runs/a800_uvg_adaptation_20260919/combined_labels/combined_quality_manifest.json"
cache_root="$run_root/latent_cache"
training_root="$run_root/training"
scratch_root="$fast_root/tmp/seedvr2_lora_${mode}"
log_root="$run_root/logs"
python_bin="$env_root/bin/python"
torchrun_bin="$env_root/bin/torchrun"
dit="$repo/third_party/SeedVR2/ckpts/seedvr2_ema_3b_bf16.safetensors"
vae="$repo/third_party/SeedVR2/ckpts/ema_vae.pth"
positive="$repo/third_party/SeedVR2/pos_emb.pt"
negative="$repo/third_party/SeedVR2/neg_emb.pt"

export TMPDIR="$fast_root/tmp"
export PIP_CACHE_DIR="$fast_root/cache/pip"
export TORCH_EXTENSIONS_DIR="$fast_root/torch_extensions"
export HF_HOME="$persist/cache/huggingface"
export TORCH_HOME="$persist/cache/torch"
export CUDA_HOME="$env_root"
export PATH="$env_root/bin:/usr/local/cuda/bin:$PATH"
cuda_wheel_root="$env_root/lib/python3.12/site-packages/nvidia/cu13"
export LD_LIBRARY_PATH="$cuda_wheel_root/lib:$env_root/targets/x86_64-linux/lib:$env_root/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$repo:$repo/third_party/SeedVR2:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0

mkdir -p "$log_root" "$scratch_root"
started_marker="$run_root/run_started_epoch.txt"
if [[ ! -s "$started_marker" ]]; then
  date +%s > "$started_marker.tmp"
  mv "$started_marker.tmp" "$started_marker"
fi
experiment_start_epoch=$(<"$started_marker")
cp "$0" "$log_root/executed_run_stage_c_a800_seedvr2_lora.sh"
exec > >(tee -a "$log_root/seedvr2_lora_${mode}.log") 2>&1
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
    echo "FAILED seedvr2_lora mode=$mode status=$status utc=$(date -u +%FT%TZ)"
  fi
  exit "$status"
}
trap fail EXIT INT TERM
(
  while true; do
    printf 'HEARTBEAT seedvr2_lora mode=%s utc=%s elapsed_s=%s ' \
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

echo "START seedvr2_lora mode=$mode utc=$(date -u +%FT%TZ)"
cd "$repo"
[[ -f "$interp_root/run.complete" ]]
[[ -s "$interp_root/summary.json" ]]
[[ -s "$sample_manifest" && -s "$dit" && -s "$vae" && -s "$positive" && -s "$negative" ]]
gpu_count=$(nvidia-smi -L | wc -l)
if [[ "$gpu_count" -ne 1 ]]; then
  echo "expected one visible GPU, found $gpu_count" >&2
  exit 3
fi

selection="$run_root/codec_selection.json"
"$python_bin" - "$interp_root/summary.json" "$selection" "$repo" \
  "$codec_train_root" "$interp_root" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
summary_path, output_path, repo, train_root, interp_root = map(Path, sys.argv[1:])
summary=json.loads(summary_path.read_text())
tag=summary['selected_tag']; alpha=float(summary['selected_alpha'])
if tag == 'alpha000':
    image=repo/'checkpoints/cvpr2026_image.pth.tar'
    video=repo/'checkpoints/cvpr2026_video_hts.pth.tar'
elif tag == 'alpha100':
    image=train_root/'training/checkpoints/image_model.pth.tar'
    video=train_root/'training/checkpoints/video_model_hts.pth.tar'
else:
    image=interp_root/'checkpoints'/tag/'image_model.pth.tar'
    video=interp_root/'checkpoints'/tag/'video_hts_model.pth.tar'
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()
value={
    'selected_tag':tag,'selected_alpha':alpha,
    'selection_summary':str(summary_path.resolve()),
    'image':str(image.resolve()),'image_sha256':sha(image),
    'video':str(video.resolve()),'video_sha256':sha(video),
    'role':f'spatial-QP selected alpha={alpha:.2f}',
}
output_path.parent.mkdir(parents=True,exist_ok=True)
tmp=output_path.with_suffix('.json.tmp')
tmp.write_text(json.dumps(value,indent=2)+'\n')
os.replace(tmp,output_path)
print(json.dumps(value,indent=2))
PY

readarray -t codec_values < <("$python_bin" - "$selection" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
print(x['image']); print(x['video']); print(x['role'])
PY
)
model_i=${codec_values[0]}
model_p=${codec_values[1]}
checkpoint_role=${codec_values[2]}

"$python_bin" demo/stage_c_seedvr2_lora_finetune.py self-test
"$torchrun_bin" --standalone --nproc-per-node=1 \
  demo/stage_c_seedvr2_lora_finetune.py cache \
  --sample-manifest "$sample_manifest" --output-dir "$cache_root" \
  --scratch-dir "$scratch_root" --model-path-i "$model_i" \
  --model-path-p "$model_p" --checkpoint-role "$checkpoint_role" \
  --vae-checkpoint "$vae" "${cache_limit_args[@]}"

"$torchrun_bin" --standalone --nproc-per-node=1 \
  demo/stage_c_seedvr2_lora_finetune.py train \
  --cache-manifest "$cache_root/manifest.json" --output-dir "$training_root" \
  --dit-checkpoint "$dit" --vae-checkpoint "$vae" \
  --positive-embedding "$positive" --negative-embedding "$negative" \
  --max-steps "$max_steps" --patch-size "$patch_size" \
  --uvg-probability 0.25 --learning-rate 1e-4 --rank 8 \
  --lora-alpha 8 --last-n-blocks 8 --save-every "$save_every"

cleanup
printf 'complete\n' > "$run_root/run.complete.tmp"
mv "$run_root/run.complete.tmp" "$run_root/run.complete"
rm -f "$run_root/run.failed"
du -sb "$run_root"
df -h /root /root/autodl-tmp /root/autodl-fs
echo "COMPLETE seedvr2_lora mode=$mode utc=$(date -u +%FT%TZ) elapsed_s=$(($(date +%s) - experiment_start_epoch))"
exec > "$log_root/finalization.log" 2>&1
wait "$tee_pid"
"$python_bin" demo/stage_c_spatial_qp_finetune_eval.py resource-snapshot \
  --output-dir "$run_root"
trap - EXIT INT TERM
