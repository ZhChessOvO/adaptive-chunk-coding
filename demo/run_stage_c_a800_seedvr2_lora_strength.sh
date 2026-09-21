#!/usr/bin/env bash
set -Eeuo pipefail

repo=/root/autodl-tmp/adaptive-chunk-coding
fast_root=/root/autodl-tmp/DCVC
env_root="$fast_root/envs/dcvcuf"
persist=/root/autodl-fs/DCVC
base_root="$persist/runs/a800_seedvr2_lora_eval_20260921"
lora_root="$persist/runs/a800_seedvr2_lora_v1_20260920"
run_root="$persist/runs/a800_seedvr2_lora_strength_20260921"
log_root="$run_root/logs"
output_root="$run_root/outputs"
summary_root="$run_root/formal"
sweep_plan="$run_root/strength_sweep_plan.json"
started_marker="$run_root/run_started_epoch.txt"
python_bin="$env_root/bin/python"
torchrun_bin="$env_root/bin/torchrun"
dit="$repo/third_party/SeedVR2/ckpts/seedvr2_ema_3b_bf16.safetensors"
vae="$repo/third_party/SeedVR2/ckpts/ema_vae.pth"
positive="$repo/third_party/SeedVR2/pos_emb.pt"
negative="$repo/third_party/SeedVR2/neg_emb.pt"
adapter="$lora_root/training/seedvr2_codec_lora.pt"
base_plan="$base_root/evaluation_plan.json"

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

mkdir -p "$log_root" "$summary_root" \
  "$output_root/lora-025" "$output_root/lora-050" "$output_root/lora-075"
if [[ ! -s "$started_marker" ]]; then
  date +%s > "$started_marker.tmp"
  mv "$started_marker.tmp" "$started_marker"
fi
experiment_start_epoch=$(<"$started_marker")
invocation_start_epoch=$(date +%s)
cp "$0" "$log_root/executed_run_stage_c_a800_seedvr2_lora_strength.sh"
exec > >(tee -a "$log_root/seedvr2_lora_strength.log") 2>&1

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
    echo "FAILED seedvr2_lora_strength status=$status utc=$(date -u +%FT%TZ)"
  fi
  exit "$status"
}
trap fail EXIT INT TERM

(
  while true; do
    count_025=$(find "$output_root/lora-025" -mindepth 2 -maxdepth 2 -name metadata.json -type f | wc -l)
    count_050=$(find "$output_root/lora-050" -mindepth 2 -maxdepth 2 -name metadata.json -type f | wc -l)
    count_075=$(find "$output_root/lora-075" -mindepth 2 -maxdepth 2 -name metadata.json -type f | wc -l)
    printf 'HEARTBEAT seedvr2_lora_strength utc=%s elapsed_s=%s s025=%s/37 s050=%s/37 s075=%s/37 ' \
      "$(date -u +%FT%TZ)" "$(($(date +%s) - experiment_start_epoch))" \
      "$count_025" "$count_050" "$count_075"
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

echo "START seedvr2_lora_strength utc=$(date -u +%FT%TZ)"
rm -f "$run_root/run.failed"
cd "$repo"
if [[ -n "$(git status --short)" ]]; then
  echo "ERROR repository must be clean before the frozen strength sweep" >&2
  git status --short
  exit 1
fi
for path in "$base_root/run.complete" "$base_root/formal/summary.json" \
  "$base_plan" "$lora_root/run.complete" "$dit" "$vae" "$positive" \
  "$negative" "$adapter"; do
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

"$python_bin" demo/stage_c_seedvr2_lora_strength_sweep.py self-test
run_if_missing "$sweep_plan" "freeze 37-sample LoRA-strength plan" \
  "$python_bin" demo/stage_c_seedvr2_lora_strength_sweep.py plan \
    --base-plan "$base_plan" --base-eval-root "$base_root" \
    --lora-checkpoint "$adapter" --output "$sweep_plan"

warm_seedvr2_file_cache
run_if_missing "$output_root/restore.complete" "LoRA strength restores" \
  timeout 3600s "$torchrun_bin" --standalone --nproc-per-node=1 \
    demo/stage_c_seedvr2_lora_strength_sweep.py restore \
    --sweep-plan "$sweep_plan" --output-root "$output_root" \
    --upstream-root "$repo/third_party/SeedVR2" \
    --dit-checkpoint "$dit" --vae-checkpoint "$vae" \
    --positive-embedding "$positive" --negative-embedding "$negative" \
    --lora-checkpoint "$adapter"

run_if_missing "$summary_root/summary.complete" "LoRA strength metrics" \
  timeout 1800s "$python_bin" demo/stage_c_seedvr2_lora_strength_sweep.py summarize \
    --sweep-plan "$sweep_plan" --output-root "$output_root" \
    --output-dir "$summary_root" --lpips-batch-size 8

cleanup

"$python_bin" - "$run_root" "$experiment_start_epoch" <<'PY'
import json, os, shutil, subprocess, sys, time
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
                gpu_indices.add(int(fields[1])); used.append(int(float(fields[2])))
            except ValueError:
                pass
assert not gpu_indices or gpu_indices == {0}
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
    "experiment": "SeedVR2 LoRA-strength final resource snapshot",
    "status": "complete",
    "completed_utc": datetime.now(timezone.utc).isoformat(),
    "wall_seconds": int(time.time()) - started,
    "sample_count": summary["sample_count"],
    "new_output_count": summary["verification"]["new_output_count"],
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
os.replace(temporary, target)
print(json.dumps(value, indent=2))
PY

du -sb "$run_root"
df -h /root /root/autodl-tmp /root/autodl-fs
echo "COMPLETE seedvr2_lora_strength utc=$(date -u +%FT%TZ) experiment_elapsed_s=$(($(date +%s) - experiment_start_epoch)) invocation_elapsed_s=$(($(date +%s) - invocation_start_epoch))"
printf 'complete\n' > "$run_root/run.complete.tmp"
mv "$run_root/run.complete.tmp" "$run_root/run.complete"
rm -f "$run_root/run.failed"
trap - EXIT
