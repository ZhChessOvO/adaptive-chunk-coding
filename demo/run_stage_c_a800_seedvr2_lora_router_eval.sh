#!/usr/bin/env bash
set -Eeuo pipefail

repo=/root/autodl-tmp/adaptive-chunk-coding
fast_root=/root/autodl-tmp/DCVC
env_root="$fast_root/envs/dcvcuf"
persist=/root/autodl-fs/DCVC
teacher_root="$persist/runs/a800_seedvr2_lora_teacher_20260921"
lora_root="$persist/runs/a800_seedvr2_lora_v1_20260920"
joint_root="$persist/runs/a800_joint_evaluation_20260918"
old_spatial_root="$persist/runs/a800_spatial_consistency_20260920"
old_eval_root="$persist/runs/a800_v6_evaluation_20260920"
run_root="$persist/runs/a800_seedvr2_lora_router_eval_20260921"
formal_root="$run_root/formal/evaluation"
log_root="$run_root/logs"
plan="$run_root/evaluation_plan.json"
summary="$run_root/formal/lora_router_2x2_summary.json"
started_marker="$run_root/run_started_epoch.txt"
python_bin="$env_root/bin/python"
torchrun_bin="$env_root/bin/torchrun"
sample_manifest="$joint_root/manifests/joint_samples.jsonl"
old_routes="$old_spatial_root/routes_v6_combined/manifest.json"
new_routes="$teacher_root/routes_retrained_spatial/manifest.json"
old_summary="$old_eval_root/formal/v6_evaluation_summary.json"
joint_formal_root="$joint_root/formal/evaluation"
adapter="$lora_root/training/seedvr2_codec_lora.pt"
dit="$repo/third_party/SeedVR2/ckpts/seedvr2_ema_3b_bf16.safetensors"
vae="$repo/third_party/SeedVR2/ckpts/ema_vae.pth"
positive="$repo/third_party/SeedVR2/pos_emb.pt"
negative="$repo/third_party/SeedVR2/neg_emb.pt"
max_wall_seconds=43000

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

mkdir -p "$log_root" "$formal_root" "$run_root/formal"
if [[ ! -s "$started_marker" ]]; then
  date +%s > "$started_marker.tmp"
  mv "$started_marker.tmp" "$started_marker"
fi
experiment_start_epoch=$(<"$started_marker")
cp "$0" "$log_root/executed_run_stage_c_a800_seedvr2_lora_router_eval.sh"
exec > >(tee -a "$log_root/lora_router_eval.log") 2>&1

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
    echo "FAILED lora_router_eval status=$status utc=$(date -u +%FT%TZ)"
  fi
  exit "$status"
}
trap fail EXIT INT TERM

(
  while true; do
    completed=$(find "$formal_root" -type f -name backend.complete 2>/dev/null | wc -l)
    expected=0
    if [[ -s "$plan" ]]; then
      expected=$("$python_bin" - "$plan" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["physical_backend_task_count"])
PY
)
    fi
    printf 'HEARTBEAT lora_router_eval utc=%s elapsed_s=%s tasks=%s/%s ' \
      "$(date -u +%FT%TZ)" "$(($(date +%s) - experiment_start_epoch))" \
      "$completed" "$expected"
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

valid_json() {
  local path=$1
  [[ -s "$path" ]] || return 1
  "$python_bin" - "$path" <<'PY' >/dev/null
import json, sys
json.load(open(sys.argv[1]))
PY
}

run_if_missing() {
  local marker=$1
  local label=$2
  shift 2
  if [[ "$marker" == *.json ]]; then
    if valid_json "$marker"; then
      echo "SKIP $label marker=$marker"
      return
    fi
  elif [[ -s "$marker" ]]; then
    echo "SKIP $label marker=$marker"
    return
  fi
  echo "START_STEP $label utc=$(date -u +%FT%TZ)"
  "$@"
  if [[ "$marker" == *.json ]]; then
    valid_json "$marker"
  else
    [[ -s "$marker" ]]
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

warm_seedvr2_file_cache() {
  "$python_bin" - "$dit" <<'PY'
import json, sys, time
from pathlib import Path
path = Path(sys.argv[1])
started = time.perf_counter()
size = 0
with path.open("rb") as handle:
    while block := handle.read(64 * 1024 * 1024):
        size += len(block)
print(json.dumps({"stage": "seedvr2-file-cache-warmup", "bytes_read": size,
                  "seconds": time.perf_counter() - started}))
PY
}

run_task() {
  local sample_id=$1
  local route_key=$2
  local backend=$3
  local route=$4
  local seed=$5
  local codec_mode=$6
  local source_codec=$7
  local route_root="$formal_root/$sample_id/$route_key"
  local codec="$route_root/codec"
  local roi="$route_root/roi"
  local backend_root="$route_root/$backend"
  local restore="$backend_root/restore"
  local evaluation="$backend_root/evaluation"
  local gate="$joint_formal_root/$sample_id/uniform_gate/summary.json"
  mkdir -p "$route_root" "$roi" "$restore" "$evaluation"
  if [[ -s "$backend_root/backend.complete" ]]; then
    echo "SKIP complete backend $sample_id/$route_key/$backend"
    return
  fi
  check_limits
  if [[ "$codec_mode" == new-real-stream ]]; then
    mkdir -p "$codec"
    run_if_missing "$codec/encode_summary.json" "$sample_id/$route_key encode" \
      timeout 900s "$python_bin" demo/stage_c_spatial_quality_codec.py encode \
        --gate-summary "$gate" --route-summary "$route" \
        --output-stream "$codec/stream.dcvc-sq" \
        --output-dir "$codec" --frame-count 17
    run_if_missing "$codec/decode_summary.json" "$sample_id/$route_key fresh-decode" \
      timeout 900s "$python_bin" demo/stage_c_spatial_quality_codec.py decode \
        --input-stream "$codec/stream.dcvc-sq" --output-dir "$codec"
  else
    codec="$source_codec"
    [[ -s "$codec/encode_summary.json" && -s "$codec/decode_summary.json" ]]
  fi
  run_if_missing "$roi/manifest.json" "$sample_id/$route_key ROI prepare" \
    "$python_bin" demo/stage_c_seedvr2_roi.py prepare \
      --route-summary "$route" --input-dir "$codec/fresh_decode" \
      --output-dir "$roi" --context-pixels 64 --processing-scale 1.0
  if [[ "$backend" == lora050 ]]; then
    run_if_missing "$restore/roi_batch_metadata.json" \
      "$sample_id/$route_key/$backend restore" \
      timeout 2400s "$torchrun_bin" --standalone --nproc-per-node=1 \
        demo/stage_c_seedvr2_roi.py restore \
        --manifest "$roi/manifest.json" --output-root "$restore" \
        --dit-checkpoint "$dit" --vae-checkpoint "$vae" \
        --positive-embedding "$positive" --negative-embedding "$negative" \
        --lora-checkpoint "$adapter" --lora-strength 0.50 \
        --seed "$seed" --sample-steps 1 --cfg-scale 1.0 --dit-dtype bfloat16
  else
    run_if_missing "$restore/roi_batch_metadata.json" \
      "$sample_id/$route_key/$backend restore" \
      timeout 2400s "$torchrun_bin" --standalone --nproc-per-node=1 \
        demo/stage_c_seedvr2_roi.py restore \
        --manifest "$roi/manifest.json" --output-root "$restore" \
        --dit-checkpoint "$dit" --vae-checkpoint "$vae" \
        --positive-embedding "$positive" --negative-embedding "$negative" \
        --seed "$seed" --sample-steps 1 --cfg-scale 1.0 --dit-dtype bfloat16
  fi
  run_if_missing "$evaluation/summary.json" "$sample_id/$route_key/$backend evaluate" \
    timeout 900s "$python_bin" demo/stage_c_seedvr2_roi.py evaluate \
      --manifest "$roi/manifest.json" --gate-summary "$gate" \
      --codec-dir "$codec" --restored-root "$restore" \
      --output-dir "$evaluation" --feather-pixels 16
  "$python_bin" - "$backend_root" "$codec" "$backend" "$adapter" <<'PY'
import hashlib, json, math, os, sys
from pathlib import Path
root, codec, backend, adapter = map(Path, sys.argv[1:])
batch = json.load(open(root / "restore/roi_batch_metadata.json"))
evaluation = json.load(open(root / "evaluation/summary.json"))
encode = json.load(open(codec / "encode_summary.json"))
decode = json.load(open(codec / "decode_summary.json"))
stream = Path(encode["stream"])
assert batch["complete"]
assert evaluation["fresh_decode_regression"]["pixel_exact"] is True
assert stream.is_file() and stream.stat().st_size == encode["stream_bytes"]
assert decode["stream_bytes"] == encode["stream_bytes"]
if backend.name == "lora050":
    digest = hashlib.sha256(adapter.read_bytes()).hexdigest()
    assert batch["lora_checkpoint_sha256"] == digest
    assert math.isclose(batch["lora_strength"], 0.5, abs_tol=1e-12)
else:
    assert batch["lora_checkpoint"] is None
target = root / "backend.complete"
temporary = root / "backend.complete.tmp"
temporary.write_text("complete\n")
os.replace(temporary, target)
PY
  echo "COMPLETE backend $sample_id/$route_key/$backend utc=$(date -u +%FT%TZ)"
}

echo "START lora_router_eval utc=$(date -u +%FT%TZ)"
rm -f "$run_root/run.failed"
cd "$repo"
if [[ -n "$(git status --short)" ]]; then
  echo "ERROR repository must be clean before frozen 2x2 evaluation" >&2
  git status --short
  exit 1
fi
for path in "$teacher_root/run.complete" "$new_routes" "$sample_manifest" \
  "$old_routes" "$old_summary" "$adapter" "$dit" "$vae" "$positive" \
  "$negative"; do
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

"$python_bin" demo/stage_c_seedvr2_lora_router_evaluation.py self-test
run_if_missing "$plan" "freeze LoRA x router 2x2 plan" \
  "$python_bin" demo/stage_c_seedvr2_lora_router_evaluation.py plan \
    --sample-manifest "$sample_manifest" --old-routes "$old_routes" \
    --new-routes "$new_routes" --old-v6-summary "$old_summary" \
    --lora-checkpoint "$adapter" --lora-strength 0.50 --output "$plan"

warm_seedvr2_file_cache
while IFS=$'\t' read -r sample_id route_key backend route seed codec_mode codec_dir; do
  run_task "$sample_id" "$route_key" "$backend" "$route" "$seed" \
    "$codec_mode" "$codec_dir"
done < <("$python_bin" - "$plan" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1]))
for task in plan["tasks"]:
    print("\t".join(map(str, (
        task["sample_id"], task["route_key"], task["backend"], task["route"],
        task["seed"], task["codec_mode"], task["codec_dir"] or "-"))))
PY
)

run_if_missing "$summary" "aggregate LoRA x router 2x2 evaluation" \
  timeout 1800s "$python_bin" \
    demo/stage_c_seedvr2_lora_router_evaluation.py summarize \
    --run-root "$run_root" --plan "$plan" \
    --joint-formal-root "$joint_formal_root" --output "$summary"

cleanup
trap - EXIT

"$python_bin" - "$run_root" "$experiment_start_epoch" <<'PY'
import json, os, shutil, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
root = Path(sys.argv[1]).resolve()
started = int(sys.argv[2])
summary = json.load(open(root / "formal/lora_router_2x2_summary.json"))
plan = json.load(open(root / "evaluation_plan.json"))
indices, used = set(), []
gpu_log = root / "logs/gpu_samples.csv"
if gpu_log.is_file():
    for line in gpu_log.read_text().splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 3:
            try:
                indices.add(int(fields[1])); used.append(int(float(fields[2])))
            except ValueError:
                pass
assert not indices or indices == {0}
files = [path for path in root.rglob("*") if path.is_file()]
mounts = {}
for name in ("/root", "/root/autodl-tmp", "/root/autodl-fs"):
    value = shutil.disk_usage(name)
    mounts[name] = {"total_bytes": value.total, "used_bytes": value.used,
                    "free_bytes": value.free}
value = {
    "experiment": "SeedVR2 LoRA x retrained router formal 2x2 snapshot",
    "status": "complete",
    "completed_utc": datetime.now(timezone.utc).isoformat(),
    "wall_seconds": int(time.time()) - started,
    "sample_count": summary["sample_count"],
    "physical_backend_tasks": plan["physical_backend_task_count"],
    "changed_route_sample_count": plan["changed_route_sample_count"],
    "peak_nvidia_smi_used_memory_mib": max(used) if used else None,
    "single_gpu_indices_observed": sorted(indices),
    "ordinary_file_count": len(files),
    "ordinary_file_bytes": sum(path.stat().st_size for path in files),
    "du_bytes": int(subprocess.check_output(
        ["du", "-sb", str(root)], text=True).split()[0]),
    "mounts": mounts,
    "verification": summary["verification"],
    "combined_effects": summary["comparisons"]["combined"]["effects"],
    "uvg_effects": summary["comparisons"]["UVG"]["effects"],
}
target = root / "run_summary.json"
temporary = target.with_suffix(".json.tmp")
temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
os.replace(temporary, target)
PY

printf 'complete\n' > "$run_root/run.complete.tmp"
mv "$run_root/run.complete.tmp" "$run_root/run.complete"
du -sb "$run_root"
df -h /root /root/autodl-tmp /root/autodl-fs
echo "COMPLETE lora_router_eval utc=$(date -u +%FT%TZ) elapsed_s=$(($(date +%s) - experiment_start_epoch))"
