#!/usr/bin/env bash
set -Eeuo pipefail

repo=/root/autodl-tmp/adaptive-chunk-coding
fast_root=/root/autodl-tmp/DCVC
env_root="$fast_root/envs/dcvcuf"
persist=/root/autodl-fs/DCVC
base_root="$persist/runs/a800_long_video_smoke_20260920"
lora_root="$persist/runs/a800_seedvr2_lora_v1_20260920"
strength_root="$persist/runs/a800_seedvr2_lora_strength_20260921"
run_root="$persist/runs/a800_seedvr2_lora_roi_long_20260921"
log_root="$run_root/logs"
restore_root="$run_root/lora050_roi_restore"
evaluation_root="$run_root/lora050_evaluation"
formal_root="$run_root/formal"
plan="$run_root/experiment_plan.json"
started_marker="$run_root/run_started_epoch.txt"
python_bin="$env_root/bin/python"
torchrun_bin="$env_root/bin/torchrun"
dit="$repo/third_party/SeedVR2/ckpts/seedvr2_ema_3b_bf16.safetensors"
vae="$repo/third_party/SeedVR2/ckpts/ema_vae.pth"
positive="$repo/third_party/SeedVR2/pos_emb.pt"
negative="$repo/third_party/SeedVR2/neg_emb.pt"
adapter="$lora_root/training/seedvr2_codec_lora.pt"
strength_summary="$strength_root/formal/summary.json"
manifest="$base_root/seedvr2_roi_plan/manifest.json"
gate="$base_root/plan/long_gate.json"
codec_root="$base_root/codec"

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

mkdir -p "$log_root" "$restore_root" "$evaluation_root" "$formal_root"
if [[ ! -s "$started_marker" ]]; then
  date +%s > "$started_marker.tmp"
  mv "$started_marker.tmp" "$started_marker"
fi
experiment_start_epoch=$(<"$started_marker")
invocation_start_epoch=$(date +%s)
cp "$0" "$log_root/executed_run_stage_c_a800_seedvr2_lora_roi_long.sh"
exec > >(tee -a "$log_root/seedvr2_lora_roi_long.log") 2>&1

heartbeat_pid=
gpu_monitor_pid=
cleanup() {
  if [[ -n "$heartbeat_pid" ]]; then kill "$heartbeat_pid" 2>/dev/null || true; fi
  if [[ -n "$gpu_monitor_pid" ]]; then kill "$gpu_monitor_pid" 2>/dev/null || true; fi
  if [[ -n "$heartbeat_pid" ]]; then wait "$heartbeat_pid" 2>/dev/null || true; fi
  if [[ -n "$gpu_monitor_pid" ]]; then wait "$gpu_monitor_pid" 2>/dev/null || true; fi
}
fail() {
  status=$?
  cleanup
  if (( status != 0 )); then
    rm -f "$run_root/run.complete"
    printf '%s\n' "$status" > "$run_root/run.failed.tmp"
    mv "$run_root/run.failed.tmp" "$run_root/run.failed"
    echo "FAILED seedvr2_lora_roi_long status=$status utc=$(date -u +%FT%TZ)"
  fi
  exit "$status"
}
trap fail EXIT INT TERM

(
  while true; do
    component_count=$(find "$restore_root" -mindepth 3 -maxdepth 3 \
      -name seedvr2_metadata.json -type f | wc -l)
    evaluation_done=0
    formal_done=0
    [[ -s "$evaluation_root/summary.json" ]] && evaluation_done=1
    [[ -s "$formal_root/summary.complete" ]] && formal_done=1
    printf 'HEARTBEAT seedvr2_lora_roi_long utc=%s elapsed_s=%s components=%s/3 evaluation=%s formal=%s ' \
      "$(date -u +%FT%TZ)" "$(($(date +%s) - experiment_start_epoch))" \
      "$component_count" "$evaluation_done" "$formal_done"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits | tr '\n' ';' || true
    printf ' '
    df -h /root /root/autodl-tmp /root/autodl-fs | tail -n +2 | tr '\n' ';'
    printf '\n'
    sleep 60
  done
) >> "$log_root/heartbeat.log" 2>&1 &
heartbeat_pid=$!

(
  while true; do
    nvidia-smi \
      --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw \
      --format=csv,noheader,nounits || true
    sleep 5
  done
) >> "$log_root/gpu_samples.csv" 2>&1 &
gpu_monitor_pid=$!

run_if_missing() {
  local marker=$1
  local label=$2
  shift 2
  if [[ -s "$marker" ]]; then
    echo "SKIP $label marker=$marker"
    return
  fi
  echo "START_STEP $label utc=$(date -u +%FT%TZ)"
  "$@"
  [[ -s "$marker" ]]
  echo "COMPLETE_STEP $label utc=$(date -u +%FT%TZ)"
}

warm_seedvr2_file_cache() {
  "$python_bin" - "$dit" <<'PY'
import json, sys, time
from pathlib import Path
path = Path(sys.argv[1])
started = time.perf_counter()
byte_count = 0
with path.open("rb") as handle:
    while block := handle.read(64 * 1024 * 1024):
        byte_count += len(block)
print(json.dumps({
    "stage": "seedvr2-file-cache-warmup",
    "bytes_read": byte_count,
    "seconds": time.perf_counter() - started,
}))
PY
}

make_comparison_video() {
  local temporary="$formal_root/lora050_long_video_comparison.mp4.tmp"
  ffmpeg -hide_banner -loglevel warning -y -framerate 8 \
    -i "$formal_root/comparison_frames/frame_%03d.png" \
    -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -movflags +faststart \
    -f mp4 "$temporary"
  mv "$temporary" "$formal_root/lora050_long_video_comparison.mp4"
}

echo "START seedvr2_lora_roi_long utc=$(date -u +%FT%TZ)"
rm -f "$run_root/run.failed"
cd "$repo"
if [[ -n "$(git status --short)" ]]; then
  echo "ERROR repository must be clean before the frozen ROI/long-video check" >&2
  git status --short
  exit 1
fi
for path in "$base_root/run.complete" "$strength_root/run.complete" \
  "$lora_root/run.complete" "$strength_summary" "$manifest" "$gate" \
  "$codec_root/decode_summary.json" "$adapter" "$dit" "$vae" \
  "$positive" "$negative"; do
  [[ -s "$path" ]]
done
gpu_count=$(nvidia-smi -L | wc -l)
if [[ "$gpu_count" -ne 1 ]]; then
  echo "expected one visible GPU, found $gpu_count" >&2
  exit 3
fi
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version \
  --format=csv,noheader,nounits
df -h /root /root/autodl-tmp /root/autodl-fs

"$python_bin" demo/stage_c_seedvr2_lora_utils.py self-test
"$python_bin" demo/stage_c_long_video_seedvr2.py self-test
"$python_bin" demo/stage_c_seedvr2_lora_roi_long_eval.py self-test

run_if_missing "$plan" "freeze LoRA-0.50 ROI/long-video plan" \
  "$python_bin" demo/stage_c_seedvr2_lora_roi_long_eval.py plan \
    --base-run "$base_root" --strength-summary "$strength_summary" \
    --lora-checkpoint "$adapter" --strength 0.50 --output "$plan"

warm_seedvr2_file_cache
run_if_missing "$restore_root/long_roi_batch_metadata.json" \
  "LoRA-0.50 overlapping ROI restoration" \
  timeout 1800s "$torchrun_bin" --standalone --nproc-per-node=1 \
    demo/stage_c_long_video_seedvr2.py restore \
    --manifest "$manifest" --output-root "$restore_root" --seed 21260920 \
    --upstream-root "$repo/third_party/SeedVR2" \
    --dit-checkpoint "$dit" --vae-checkpoint "$vae" \
    --positive-embedding "$positive" --negative-embedding "$negative" \
    --lora-checkpoint "$adapter" --lora-strength 0.50 \
    --sample-steps 1 --cfg-scale 1.0 --dit-dtype bfloat16

run_if_missing "$evaluation_root/summary.json" "LoRA-0.50 long-video evaluation" \
  timeout 900s "$python_bin" demo/stage_c_long_video_seedvr2.py evaluate \
    --manifest "$manifest" --gate-summary "$gate" \
    --codec-dir "$codec_root" --restored-root "$restore_root" \
    --output-dir "$evaluation_root" --feather-pixels 16

run_if_missing "$formal_root/summary.complete" "frozen-vs-LoRA ROI summary" \
  timeout 900s "$python_bin" demo/stage_c_seedvr2_lora_roi_long_eval.py summarize \
    --plan "$plan" --lora-evaluation "$evaluation_root/summary.json" \
    --lora-restored-root "$restore_root" --output-dir "$formal_root"

run_if_missing "$formal_root/lora050_long_video_comparison.mp4" \
  "lossless-source visual comparison MP4" make_comparison_video

cleanup

"$python_bin" - "$run_root" "$experiment_start_epoch" <<'PY'
import json, shutil, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1]).resolve()
started = int(sys.argv[2])
summary = json.load(open(root / "formal/summary.json"))
gpu_indices, used = set(), []
gpu_log = root / "logs/gpu_samples.csv"
if gpu_log.is_file():
    for line in gpu_log.read_text().splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 3:
            try:
                gpu_indices.add(int(fields[1]))
                used.append(int(float(fields[2])))
            except ValueError:
                pass
assert not gpu_indices or gpu_indices == {0}
assert summary["verification"]["outside_generate_pixels_unchanged"] is True
assert (root / "formal/lora050_long_video_comparison.mp4").stat().st_size > 0
files = [path for path in root.rglob("*") if path.is_file()]
mounts = {}
for name in ("/root", "/root/autodl-tmp", "/root/autodl-fs"):
    usage = shutil.disk_usage(name)
    mounts[name] = {
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    }
value = {
    "experiment": "SeedVR2 LoRA-0.50 ROI/long-video final resource snapshot",
    "status": "complete",
    "completed_utc": datetime.now(timezone.utc).isoformat(),
    "wall_seconds": int(time.time()) - started,
    "frames": summary["frames"],
    "roi_component_count": 3,
    "peak_nvidia_smi_used_memory_mib": max(used) if used else None,
    "single_gpu_indices_observed": sorted(gpu_indices),
    "ordinary_file_count_before_snapshot": len(files),
    "ordinary_file_bytes_before_snapshot": sum(path.stat().st_size for path in files),
    "du_bytes_before_snapshot": int(subprocess.check_output(
        ["du", "-sb", str(root)], text=True).split()[0]),
    "mounts": mounts,
    "verification": summary["verification"],
}
target = root / "final_resource_snapshot.json"
temporary = target.with_suffix(".json.tmp")
temporary.write_text(json.dumps(value, indent=2) + "\n")
temporary.replace(target)
print(json.dumps(value, indent=2))
PY

du -sb "$run_root"
df -h /root /root/autodl-tmp /root/autodl-fs
echo "COMPLETE seedvr2_lora_roi_long utc=$(date -u +%FT%TZ) experiment_elapsed_s=$(($(date +%s) - experiment_start_epoch)) invocation_elapsed_s=$(($(date +%s) - invocation_start_epoch))"
printf 'complete\n' > "$run_root/run.complete.tmp"
mv "$run_root/run.complete.tmp" "$run_root/run.complete"
rm -f "$run_root/run.failed"
trap - EXIT
