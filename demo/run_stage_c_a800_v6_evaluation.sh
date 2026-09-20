#!/usr/bin/env bash
set -Eeuo pipefail

repo=/root/autodl-tmp/adaptive-chunk-coding
fast_root=/root/autodl-tmp/DCVC
env_root="$fast_root/envs/dcvcuf"
persist=/root/autodl-fs/DCVC
joint_root="$persist/runs/a800_joint_evaluation_20260918"
adaptation_root="$persist/runs/a800_uvg_adaptation_20260919"
spatial_root="$persist/runs/a800_spatial_consistency_20260920"
feather_root="$persist/runs/a800_feather_diagnostic_20260919"
run_root="$persist/runs/a800_v6_evaluation_20260920"
formal_root="$run_root/formal/evaluation"
log_root="$run_root/logs"
log_file="$log_root/v6_evaluation.log"
gpu_log="$log_root/gpu_samples.csv"
heartbeat_log="$log_root/heartbeat.log"
plan="$run_root/evaluation_plan.json"
frozen_source="$run_root/frozen_source.json"
summary="$run_root/formal/v6_evaluation_summary.json"
started_marker="$run_root/run_started_epoch.txt"
python_bin="$env_root/bin/python"
torchrun_bin="$env_root/bin/torchrun"
max_wall_seconds=43200

sample_manifest="$joint_root/manifests/joint_samples.jsonl"
baseline_routes="$joint_root/routes/controller/manifest.json"
adaptation_routes="$adaptation_root/routes_joint37/manifest.json"
spatial_routes="$spatial_root/routes_v5_spatial/manifest.json"
combined_routes="$spatial_root/routes_v6_combined/manifest.json"
joint_summary="$joint_root/formal/joint_evaluation_summary.json"
feather_summary="$feather_root/formal/feather_diagnostic_summary.json"
joint_formal_root="$joint_root/formal/evaluation"

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

mkdir -p "$log_root" "$formal_root" "$run_root/formal"
if [[ ! -f "$started_marker" ]]; then
  date +%s > "$started_marker.tmp"
  mv "$started_marker.tmp" "$started_marker"
fi
experiment_start_epoch=$(<"$started_marker")
invocation_start_epoch=$(date +%s)
cp "$0" "$log_root/executed_run_stage_c_a800_v6_evaluation.sh"
exec > >(tee -a "$log_file") 2>&1

heartbeat_pid=
gpu_monitor_pid=
cleanup() {
  if [[ -n "$heartbeat_pid" ]]; then kill "$heartbeat_pid" 2>/dev/null || true; fi
  if [[ -n "$gpu_monitor_pid" ]]; then kill "$gpu_monitor_pid" 2>/dev/null || true; fi
  if [[ -n "$heartbeat_pid" ]]; then wait "$heartbeat_pid" 2>/dev/null || true; fi
  if [[ -n "$gpu_monitor_pid" ]]; then wait "$gpu_monitor_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

(
  while true; do
    now_epoch=$(date +%s)
    completed=$(find "$formal_root" -type f -name variant.complete 2>/dev/null | wc -l)
    expected=58
    if [[ -s "$plan" ]]; then
      expected=$("$python_bin" - "$plan" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["new_real_evaluation_task_count"])
PY
)
    fi
    printf 'HEARTBEAT v6_evaluation utc=%s experiment_elapsed_s=%s invocation_elapsed_s=%s tasks=%s/%s ' \
      "$(date -u +%FT%TZ)" "$((now_epoch - experiment_start_epoch))" \
      "$((now_epoch - invocation_start_epoch))" "$completed" "$expected"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits | tr '\n' ';' || true
    printf ' '
    df -h /root /root/autodl-tmp /root/autodl-fs | tail -n +2 | tr '\n' ';'
    printf '\n'
    sleep 60
  done
) >> "$heartbeat_log" 2>&1 &
heartbeat_pid=$!

(
  while true; do
    nvidia-smi \
      --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw \
      --format=csv,noheader,nounits || true
    sleep 5
  done
) >> "$gpu_log" 2>&1 &
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
  "$python_bin" - "$repo/third_party/SeedVR2/ckpts/seedvr2_ema_3b_bf16.safetensors" <<'PY'
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

run_variant() {
  local sample_id=$1
  local variant=$2
  local route=$3
  local seed=$4
  local gate="$joint_formal_root/$sample_id/uniform_gate/summary.json"
  local root="$formal_root/$sample_id/$variant"
  local codec="$root/codec"
  local roi="$root/roi"
  local evaluation="$root/evaluation"
  mkdir -p "$codec" "$roi" "$evaluation"
  if [[ -s "$root/variant.complete" ]]; then
    echo "SKIP completed canonical variant $sample_id/$variant"
    return
  fi
  check_limits
  run_if_missing "$codec/encode_summary.json" "$sample_id/$variant encode" \
    timeout 900s "$python_bin" demo/stage_c_spatial_quality_codec.py encode \
      --gate-summary "$gate" --route-summary "$route" \
      --output-stream "$codec/stream.dcvc-sq" \
      --output-dir "$codec" --frame-count 17
  run_if_missing "$codec/decode_summary.json" "$sample_id/$variant fresh-decode" \
    timeout 900s "$python_bin" demo/stage_c_spatial_quality_codec.py decode \
      --input-stream "$codec/stream.dcvc-sq" --output-dir "$codec"
  run_if_missing "$roi/manifest.json" "$sample_id/$variant ROI prepare" \
    "$python_bin" demo/stage_c_seedvr2_roi.py prepare \
      --route-summary "$route" --input-dir "$codec/fresh_decode" \
      --output-dir "$roi" --context-pixels 64 --processing-scale 1.0
  run_if_missing "$roi/roi_batch_metadata.json" "$sample_id/$variant ROI restore" \
    timeout 2400s "$torchrun_bin" --standalone --nproc-per-node=1 \
      demo/stage_c_seedvr2_roi.py restore \
      --manifest "$roi/manifest.json" --output-root "$roi" \
      --dit-checkpoint third_party/SeedVR2/ckpts/seedvr2_ema_3b_bf16.safetensors \
      --seed "$seed" --sample-steps 1 --cfg-scale 1.0 --dit-dtype bfloat16
  run_if_missing "$evaluation/summary.json" "$sample_id/$variant evaluate" \
    timeout 900s "$python_bin" demo/stage_c_seedvr2_roi.py evaluate \
      --manifest "$roi/manifest.json" --gate-summary "$gate" \
      --codec-dir "$codec" --restored-root "$roi" \
      --output-dir "$evaluation" --feather-pixels 16
  "$python_bin" - "$root" "$route" <<'PY'
import json, os, sys
from pathlib import Path
root = Path(sys.argv[1])
route = json.load(open(sys.argv[2]))
selected = route["variants"][route["selected_variant"]]["actions"]
encode = json.load(open(root / "codec/encode_summary.json"))
decode = json.load(open(root / "codec/decode_summary.json"))
roi = json.load(open(root / "roi/manifest.json"))
evaluation = json.load(open(root / "evaluation/summary.json"))
stream = Path(encode["stream"])
assert stream.is_file() and stream.stat().st_size == encode["stream_bytes"]
assert encode["stream_bytes"] == decode["stream_bytes"]
assert evaluation["fresh_decode_regression"]["pixel_exact"] is True
flat = [value for row in roi["actions"] for value in row]
assert flat == selected
target = root / "variant.complete"
temporary = root / "variant.complete.tmp"
temporary.write_text("complete\n")
os.replace(temporary, target)
PY
  echo "COMPLETE canonical variant $sample_id/$variant utc=$(date -u +%FT%TZ)"
}

echo "START v6_evaluation utc=$(date -u +%FT%TZ)"
cd "$repo"
if [[ -n "$(git status --short)" ]]; then
  echo "ERROR repository must be clean before frozen quality evaluation" >&2
  git status --short
  exit 1
fi
for path in "$sample_manifest" "$baseline_routes" "$adaptation_routes" \
  "$spatial_routes" "$combined_routes" "$joint_summary" "$feather_summary"; do
  [[ -s "$path" ]]
done
[[ -s "$spatial_root/spatial_route_only.complete" ]]
[[ -s "$adaptation_root/adapted_route_only.complete" ]]
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version \
  --format=csv,noheader,nounits
df -h /root /root/autodl-tmp /root/autodl-fs

run_if_missing "$plan" "freeze minimal exact-action evaluation plan" \
  "$python_bin" demo/stage_c_a800_v6_evaluation_plan.py \
    --sample-manifest "$sample_manifest" \
    --baseline-routes "$baseline_routes" \
    --adaptation-routes "$adaptation_routes" \
    --spatial-routes "$spatial_routes" \
    --combined-routes "$combined_routes" --output "$plan"

if [[ ! -s "$frozen_source" ]]; then
  "$python_bin" - "$frozen_source" "$plan" "$joint_summary" \
    "$feather_summary" <<'PY'
import hashlib, json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
target = Path(sys.argv[1])
plan = json.load(open(sys.argv[2]))
value = {
    "experiment": "A800 v6 adaptation x spatial-consistency formal freeze",
    "status": "frozen-before-quality-evaluation",
    "frozen_utc": datetime.now(timezone.utc).isoformat(),
    "git_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(),
    "plan": str(Path(sys.argv[2]).resolve()),
    "plan_sha256": sha(sys.argv[2]),
    "new_real_evaluation_task_count": plan["new_real_evaluation_task_count"],
    "logical_target_evaluation_count": plan["logical_target_evaluations"],
    "joint_summary": str(Path(sys.argv[3]).resolve()),
    "joint_summary_sha256": sha(sys.argv[3]),
    "feather_summary": str(Path(sys.argv[4]).resolve()),
    "feather_summary_sha256": sha(sys.argv[4]),
    "feather_pixels": 16,
    "spatial_lambda": 0.004,
    "hard_promotion_gate": False,
    "models_and_codec_frozen": True,
}
temporary = target.with_suffix(".json.tmp")
temporary.write_text(json.dumps(value, indent=2) + "\n")
os.replace(temporary, target)
PY
fi

warm_seedvr2_file_cache

while IFS=$'\t' read -r sample_id variant route seed; do
  run_variant "$sample_id" "$variant" "$route" "$seed"
done < <("$python_bin" - "$plan" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))
for task in value["tasks"]:
    print("\t".join(map(str, (
        task["sample_id"], task["variant"], task["route"], task["seed"]))))
PY
)

run_if_missing "$summary" "v6 real-evaluation aggregate" \
  "$python_bin" demo/stage_c_a800_v6_evaluation_summary.py \
    --run-root "$run_root" --plan "$plan" \
    --joint-summary "$joint_summary" --feather-summary "$feather_summary" \
    --joint-formal-root "$joint_formal_root" --output "$summary"

printf 'complete\n' > "$run_root/formal_evaluation.complete.tmp"
mv "$run_root/formal_evaluation.complete.tmp" "$run_root/formal_evaluation.complete"
cleanup
trap - EXIT

"$python_bin" - "$run_root" "$experiment_start_epoch" <<'PY'
import json, os, shutil, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
root = Path(sys.argv[1]).resolve()
start = int(sys.argv[2])
summary = json.load(open(root / "formal/v6_evaluation_summary.json"))
plan = json.load(open(root / "evaluation_plan.json"))
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
markers = list(root.glob("formal/evaluation/*/*/variant.complete"))
assert len(markers) == plan["new_real_evaluation_task_count"]
files = [path for path in root.rglob("*") if path.is_file()]
mounts = {}
for name in ("/root", "/root/autodl-tmp", "/root/autodl-fs"):
    value = shutil.disk_usage(name)
    mounts[name] = {
        "total_bytes": value.total,
        "used_bytes": value.used,
        "free_bytes": value.free,
    }
value = {
    "experiment": "A800 v6 formal evaluation final snapshot",
    "status": "complete",
    "completed_utc": datetime.now(timezone.utc).isoformat(),
    "wall_seconds": int(__import__("time").time()) - start,
    "sample_count": summary["sample_count"],
    "new_real_evaluation_tasks": len(markers),
    "logical_target_evaluations": plan["logical_target_evaluations"],
    "peak_nvidia_smi_used_memory_mib": max(used) if used else None,
    "single_gpu_indices_observed": sorted(gpu_indices),
    "ordinary_file_count": len(files),
    "ordinary_file_bytes": sum(path.stat().st_size for path in files),
    "du_bytes": int(subprocess.check_output(
        ["du", "-sb", str(root)], text=True).split()[0]),
    "mounts": mounts,
    "verification": summary["verification"],
    "combined_effects": summary["comparisons"]["combined"][
        "factorial_effects"],
    "uvg_effects": summary["comparisons"]["UVG"]["factorial_effects"],
}
target = root / "final_resource_snapshot.json"
temporary = target.with_suffix(".json.tmp")
temporary.write_text(json.dumps(value, indent=2) + "\n")
os.replace(temporary, target)
print(json.dumps(value, indent=2))
PY

du -sb "$run_root"
df -h /root /root/autodl-tmp /root/autodl-fs
echo "COMPLETE v6_evaluation utc=$(date -u +%FT%TZ) experiment_elapsed_s=$(($(date +%s) - experiment_start_epoch)) invocation_elapsed_s=$(($(date +%s) - invocation_start_epoch))"
