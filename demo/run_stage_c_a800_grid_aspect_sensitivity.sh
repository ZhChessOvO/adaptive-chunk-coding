#!/usr/bin/env bash
set -Eeuo pipefail

repo=/root/autodl-tmp/adaptive-chunk-coding
fast_root=/root/autodl-tmp/DCVC
env_root="$fast_root/envs/dcvcuf"
persist=/root/autodl-fs/DCVC
run_root="$persist/runs/a800_grid_aspect_sensitivity_20260921"
joint_root="$persist/runs/a800_joint_evaluation_20260918"
spatial_root="$persist/runs/a800_spatial_consistency_20260920"
reference_root="$persist/runs/a800_seedvr2_lora_router_eval_20260921"
formal_root="$run_root/formal/evaluation"
log_root="$run_root/logs"
plan="$run_root/evaluation_plan.json"
route_manifest="$run_root/routes/manifest.json"
summary="$run_root/formal/grid_aspect_sensitivity_summary.json"
started_marker="$run_root/run_started_epoch.txt"
python_bin="$env_root/bin/python"
torchrun_bin="$env_root/bin/torchrun"
sample_manifest="$joint_root/manifests/joint_samples.jsonl"
joint_formal_root="$joint_root/formal/evaluation"
reference_routes="$spatial_root/routes_v6_combined/manifest.json"
reference_summary="$reference_root/formal/lora_router_2x2_summary.json"
controller="$fast_root/models/controllers/controller_v5_uvg_adapted.pt"
image_model="$fast_root/models/dcvcuf/cvpr2026_image.pth.tar"
video_model="$fast_root/models/dcvcuf/cvpr2026_video_hts.pth.tar"
adapter="$fast_root/models/seedvr2/seedvr2_codec_lora_v1_1000step.pt"
dit="$fast_root/models/seedvr2/seedvr2_ema_3b_bf16.safetensors"
vae="$fast_root/models/seedvr2/ema_vae.pth"
positive="$fast_root/models/seedvr2/pos_emb.pt"
negative="$fast_root/models/seedvr2/neg_emb.pt"
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

mkdir -p "$log_root" "$formal_root" "$run_root/routes" "$run_root/probes"
if [[ "${1:-}" != "--run-snapshot" ]]; then
  snapshot="$log_root/executed_run_stage_c_a800_grid_aspect_sensitivity_$(date -u +%Y%m%dT%H%M%SZ)_$$.sh"
  cp "$0" "$snapshot.tmp"
  mv "$snapshot.tmp" "$snapshot"
  exec bash "$snapshot" --run-snapshot
fi
shift
if [[ ! -s "$started_marker" ]]; then
  date +%s > "$started_marker.tmp"
  mv "$started_marker.tmp" "$started_marker"
fi
experiment_start_epoch=$(<"$started_marker")
exec > >(tee -a "$log_root/grid_aspect_sensitivity.log") 2>&1

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
    echo "FAILED grid_aspect_sensitivity status=$status utc=$(date -u +%FT%TZ)"
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
    printf 'HEARTBEAT grid_aspect utc=%s elapsed_s=%s tasks=%s/%s ' \
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
  local variant=$2
  local gate=$3
  local route=$4
  local seed=$5
  local task_root="$formal_root/$sample_id/$variant"
  local codec="$task_root/codec"
  local roi="$task_root/roi"
  local backend="$task_root/lora050"
  local restore="$backend/restore"
  local evaluation="$backend/evaluation"
  mkdir -p "$codec" "$roi" "$restore" "$evaluation"
  if [[ -s "$backend/backend.complete" ]]; then
    echo "SKIP complete backend $sample_id/$variant/lora050"
    return
  fi
  check_limits
  run_if_missing "$codec/encode_summary.json" "$sample_id/$variant encode" \
    timeout 1200s "$python_bin" demo/stage_c_spatial_quality_codec.py encode \
      --gate-summary "$gate" --route-summary "$route" \
      --output-stream "$codec/stream.dcvc-sq" --output-dir "$codec" \
      --frame-count 17 --cell-size 64 \
      --model-path-i "$image_model" --model-path-p "$video_model"
  run_if_missing "$codec/decode_summary.json" "$sample_id/$variant fresh-decode" \
    timeout 1200s "$python_bin" demo/stage_c_spatial_quality_codec.py decode \
      --input-stream "$codec/stream.dcvc-sq" --output-dir "$codec" \
      --model-path-i "$image_model" --model-path-p "$video_model"
  run_if_missing "$roi/manifest.json" "$sample_id/$variant ROI prepare" \
    "$python_bin" demo/stage_c_seedvr2_roi.py prepare \
      --route-summary "$route" --input-dir "$codec/fresh_decode" \
      --output-dir "$roi" --context-pixels 64 --processing-scale 1.0
  run_if_missing "$restore/roi_batch_metadata.json" \
    "$sample_id/$variant/lora050 restore" \
    timeout 3000s "$torchrun_bin" --standalone --nproc-per-node=1 \
      demo/stage_c_seedvr2_roi.py restore \
      --manifest "$roi/manifest.json" --output-root "$restore" \
      --dit-checkpoint "$dit" --vae-checkpoint "$vae" \
      --positive-embedding "$positive" --negative-embedding "$negative" \
      --lora-checkpoint "$adapter" --lora-strength 0.50 \
      --seed "$seed" --sample-steps 1 --cfg-scale 1.0 --dit-dtype bfloat16
  run_if_missing "$evaluation/summary.json" "$sample_id/$variant evaluate" \
    timeout 1200s "$python_bin" demo/stage_c_seedvr2_roi.py evaluate \
      --manifest "$roi/manifest.json" --gate-summary "$gate" \
      --codec-dir "$codec" --restored-root "$restore" \
      --output-dir "$evaluation" --feather-pixels 16
  "$python_bin" - "$backend" "$codec" "$adapter" <<'PY'
import hashlib, json, math, os, sys
from pathlib import Path
root, codec, adapter = map(Path, sys.argv[1:])
batch = json.load(open(root / "restore/roi_batch_metadata.json"))
evaluation = json.load(open(root / "evaluation/summary.json"))
encode = json.load(open(codec / "encode_summary.json"))
decode = json.load(open(codec / "decode_summary.json"))
stream = Path(encode["stream"])
digest = hashlib.sha256(adapter.read_bytes()).hexdigest()
assert batch["complete"]
assert batch["lora_checkpoint_sha256"] == digest
assert math.isclose(batch["lora_strength"], 0.5, abs_tol=1e-12)
assert evaluation["fresh_decode_regression"]["pixel_exact"] is True
assert stream.is_file() and stream.stat().st_size == encode["stream_bytes"]
assert decode["stream_bytes"] == encode["stream_bytes"]
target = root / "backend.complete"
temporary = root / "backend.complete.tmp"
temporary.write_text("complete\n")
os.replace(temporary, target)
PY
  echo "COMPLETE backend $sample_id/$variant/lora050 utc=$(date -u +%FT%TZ)"
}

echo "START grid_aspect_sensitivity utc=$(date -u +%FT%TZ)"
rm -f "$run_root/run.failed"
cd "$repo"
if [[ -n "$(git status --short)" ]]; then
  echo "ERROR repository must be clean before frozen grid/aspect sensitivity" >&2
  git status --short
  exit 1
fi
for path in "$sample_manifest" "$reference_routes" "$reference_summary" \
  "$controller" "$image_model" "$video_model" "$adapter" "$dit" "$vae" \
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

"$python_bin" demo/stage_c_a800_spatial_consistency.py self-test
"$python_bin" demo/stage_c_grid_aspect_sensitivity.py self-test
run_if_missing "$plan" "freeze grid/aspect sensitivity plan" \
  "$python_bin" demo/stage_c_grid_aspect_sensitivity.py plan \
    --joint-manifest "$sample_manifest" --joint-formal-root "$joint_formal_root" \
    --reference-routes "$reference_routes" \
    --reference-summary "$reference_summary" --checkpoint "$controller" \
    --route-output-root "$run_root/routes" \
    --gate-output-root "$run_root/probes" --output "$plan"

"$python_bin" demo/stage_c_grid_aspect_sensitivity.py probe \
  --plan "$plan" --model-path-i "$image_model" \
  --model-path-p "$video_model" --cuda-idx 0

run_if_missing "$route_manifest" "generalized routes and exact 4x4 regression" \
  timeout 3600s "$python_bin" demo/stage_c_grid_aspect_sensitivity.py route \
    --plan "$plan" --checkpoint "$controller" \
    --output "$route_manifest" --lpips-batch-size 4 --cuda-idx 0

warm_seedvr2_file_cache
while IFS=$'\t' read -r sample_id variant gate route seed; do
  run_task "$sample_id" "$variant" "$gate" "$route" "$seed"
done < <("$python_bin" - "$plan" <<'PY'
import json, sys
for task in json.load(open(sys.argv[1]))["tasks"]:
    print("\t".join(map(str, (
        task["sample_id"], task["variant"], task["gate_summary"],
        task["route"], task["seed"]))))
PY
)

run_if_missing "$summary" "aggregate grid/aspect sensitivity" \
  timeout 1800s "$python_bin" demo/stage_c_grid_aspect_sensitivity.py summarize \
    --plan "$plan" --route-manifest "$route_manifest" \
    --run-root "$run_root" --reference-summary "$reference_summary" \
    --output "$summary"

cleanup
trap - EXIT

"$python_bin" - "$run_root" "$experiment_start_epoch" <<'PY'
import json, os, shutil, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
root = Path(sys.argv[1]).resolve()
started = int(sys.argv[2])
summary = json.load(open(root / "formal/grid_aspect_sensitivity_summary.json"))
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
    "experiment": "selected-system spatial grid and aspect sensitivity",
    "status": "complete",
    "completed_utc": datetime.now(timezone.utc).isoformat(),
    "wall_seconds": int(time.time()) - started,
    "logical_evaluations": plan["logical_evaluation_count"],
    "physical_backend_tasks": plan["physical_backend_task_count"],
    "peak_nvidia_smi_used_memory_mib": max(used) if used else None,
    "single_gpu_indices_observed": sorted(indices),
    "ordinary_file_count": len(files),
    "ordinary_file_bytes": sum(path.stat().st_size for path in files),
    "du_bytes": int(subprocess.check_output(
        ["du", "-sb", str(root)], text=True).split()[0]),
    "mounts": mounts,
    "grid_aggregate": summary["grid_aggregate"],
    "verification": summary["verification"],
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
echo "COMPLETE grid_aspect_sensitivity utc=$(date -u +%FT%TZ) elapsed_s=$(($(date +%s) - experiment_start_epoch))"
