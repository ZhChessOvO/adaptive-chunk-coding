#!/usr/bin/env bash
set -Eeuo pipefail

repo=/root/autodl-tmp/adaptive-chunk-coding
fast_root=/root/autodl-tmp/DCVC
env_root="$fast_root/envs/dcvcuf"
persist=/root/autodl-fs/DCVC
joint_root="$persist/runs/a800_joint_evaluation_20260918"
lora_root="$persist/runs/a800_seedvr2_lora_v1_20260920"
run_root="$persist/runs/a800_seedvr2_lora_eval_20260921"
log_root="$run_root/logs"
output_root="$run_root/outputs"
summary_root="$run_root/formal"
plan="$run_root/evaluation_plan.json"
started_marker="$run_root/run_started_epoch.txt"
python_bin="$env_root/bin/python"
torchrun_bin="$env_root/bin/torchrun"
dit="$repo/third_party/SeedVR2/ckpts/seedvr2_ema_3b_bf16.safetensors"
vae="$repo/third_party/SeedVR2/ckpts/ema_vae.pth"
positive="$repo/third_party/SeedVR2/pos_emb.pt"
negative="$repo/third_party/SeedVR2/neg_emb.pt"
adapter="$lora_root/training/seedvr2_codec_lora.pt"
sample_manifest="$joint_root/manifests/joint_samples.jsonl"

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

mkdir -p "$log_root" "$output_root/frozen" "$output_root/lora" "$summary_root"
if [[ ! -s "$started_marker" ]]; then
  date +%s > "$started_marker.tmp"
  mv "$started_marker.tmp" "$started_marker"
fi
experiment_start_epoch=$(<"$started_marker")
invocation_start_epoch=$(date +%s)
cp "$0" "$log_root/executed_run_stage_c_a800_seedvr2_lora_eval.sh"
exec > >(tee -a "$log_root/seedvr2_lora_eval.log") 2>&1

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
    echo "FAILED seedvr2_lora_eval status=$status utc=$(date -u +%FT%TZ)"
  fi
  exit "$status"
}
trap fail EXIT INT TERM

(
  while true; do
    frozen=$(find "$output_root/frozen" -mindepth 2 -maxdepth 2 -name metadata.json -type f 2>/dev/null | wc -l)
    lora=$(find "$output_root/lora" -mindepth 2 -maxdepth 2 -name metadata.json -type f 2>/dev/null | wc -l)
    printf 'HEARTBEAT seedvr2_lora_eval utc=%s elapsed_s=%s frozen=%s/37 lora=%s/37 ' \
      "$(date -u +%FT%TZ)" "$(($(date +%s) - experiment_start_epoch))" \
      "$frozen" "$lora"
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

echo "START seedvr2_lora_eval utc=$(date -u +%FT%TZ)"
rm -f "$run_root/run.failed"
cd "$repo"
if [[ -n "$(git status --short)" ]]; then
  echo "ERROR repository must be clean before the frozen evaluation" >&2
  git status --short
  exit 1
fi
[[ -f "$joint_root/formal_evaluation.complete" ]]
for path in "$lora_root/run.complete" "$sample_manifest" "$dit" "$vae" \
  "$positive" "$negative" "$adapter"; do
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

"$python_bin" demo/stage_c_seedvr2_lora_eval.py self-test
run_if_missing "$plan" "freeze 37-sample paired plan" \
  "$python_bin" demo/stage_c_seedvr2_lora_eval.py plan \
    --joint-root "$joint_root" --sample-manifest "$sample_manifest" \
    --dit-checkpoint "$dit" --vae-checkpoint "$vae" \
    --positive-embedding "$positive" --negative-embedding "$negative" \
    --lora-checkpoint "$adapter" --output "$plan"

warm_seedvr2_file_cache
run_if_missing "$output_root/frozen/variant.complete" "frozen rerun" \
  timeout 7200s "$torchrun_bin" --standalone --nproc-per-node=1 \
    demo/stage_c_seedvr2_lora_eval.py restore \
    --plan "$plan" --output-root "$output_root" --variant frozen \
    --upstream-root "$repo/third_party/SeedVR2" \
    --dit-checkpoint "$dit" --vae-checkpoint "$vae" \
    --positive-embedding "$positive" --negative-embedding "$negative"

run_if_missing "$output_root/lora/variant.complete" "LoRA rerun" \
  timeout 7200s "$torchrun_bin" --standalone --nproc-per-node=1 \
    demo/stage_c_seedvr2_lora_eval.py restore \
    --plan "$plan" --output-root "$output_root" --variant lora \
    --upstream-root "$repo/third_party/SeedVR2" \
    --dit-checkpoint "$dit" --vae-checkpoint "$vae" \
    --positive-embedding "$positive" --negative-embedding "$negative" \
    --lora-checkpoint "$adapter"

run_if_missing "$summary_root/summary.complete" "paired metric summary" \
  timeout 3600s "$python_bin" demo/stage_c_seedvr2_lora_eval.py summarize \
    --plan "$plan" --output-root "$output_root" \
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
    "experiment": "SeedVR2 frozen-vs-LoRA final resource snapshot",
    "status": "complete",
    "completed_utc": datetime.now(timezone.utc).isoformat(),
    "wall_seconds": int(time.time()) - started,
    "sample_count": summary["sample_count"],
    "peak_nvidia_smi_used_memory_mib": max(used) if used else None,
    "single_gpu_indices_observed": sorted(gpu_indices),
    "ordinary_file_count": len(files),
    "ordinary_file_bytes": sum(path.stat().st_size for path in files),
    "du_bytes": int(subprocess.check_output(
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
echo "COMPLETE seedvr2_lora_eval utc=$(date -u +%FT%TZ) experiment_elapsed_s=$(($(date +%s) - experiment_start_epoch)) invocation_elapsed_s=$(($(date +%s) - invocation_start_epoch))"
printf 'complete\n' > "$run_root/run.complete.tmp"
mv "$run_root/run.complete.tmp" "$run_root/run.complete"
rm -f "$run_root/run.failed"
trap - EXIT
