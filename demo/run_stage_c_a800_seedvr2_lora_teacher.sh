#!/usr/bin/env bash
set -Eeuo pipefail

repo=/root/autodl-tmp/adaptive-chunk-coding
fast_root=/root/autodl-tmp/DCVC
env_root="$fast_root/envs/dcvcuf"
persist=/root/autodl-fs/DCVC
source_root="$persist/runs/a800_uvg_adaptation_20260919"
lora_root="$persist/runs/a800_seedvr2_lora_v1_20260920"
joint_root="$persist/runs/a800_joint_evaluation_20260918"
pilot_root="$persist/runs/a800_pilot_20260917"
old_spatial_root="$persist/runs/a800_spatial_consistency_20260920"
run_root="$persist/runs/a800_seedvr2_lora_teacher_20260921"
log_root="$run_root/logs"
teacher_root="$run_root/teacher_lora050"
formal_root="$run_root/formal"
controller_root="$run_root/controller"
routes_root="$run_root/routes_retrained"
spatial_routes_root="$run_root/routes_retrained_spatial"
plan="$run_root/experiment_plan.json"
selection="$run_root/fixed_router_selection.json"
started_marker="$run_root/run_started_epoch.txt"
python_bin="$env_root/bin/python"
torchrun_bin="$env_root/bin/torchrun"
quality_source="$source_root/combined_labels/combined_quality_manifest.json"
roi_source="$source_root/combined_labels/combined_roi_cost_manifest.json"
adapter="$lora_root/training/seedvr2_codec_lora.pt"
source_selection="$source_root/combined_labels/fixed_v5_selection.json"
v1_checkpoint="$pilot_root/pilot/controller/controller.pt"
joint_samples="$joint_root/manifests/joint_samples.jsonl"
base_probe_root="$joint_root/formal/evaluation"
old_spatial_routes="$old_spatial_root/routes_v6_combined/manifest.json"
dit="$repo/third_party/SeedVR2/ckpts/seedvr2_ema_3b_bf16.safetensors"
vae="$repo/third_party/SeedVR2/ckpts/ema_vae.pth"
positive="$repo/third_party/SeedVR2/pos_emb.pt"
negative="$repo/third_party/SeedVR2/neg_emb.pt"
codec_image="$repo/checkpoints/cvpr2026_image.pth.tar"
codec_video="$repo/checkpoints/cvpr2026_video_hts.pth.tar"

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

mkdir -p "$log_root" "$teacher_root" "$formal_root" "$controller_root"
if [[ ! -s "$started_marker" ]]; then
  date +%s > "$started_marker.tmp"
  mv "$started_marker.tmp" "$started_marker"
fi
experiment_start_epoch=$(<"$started_marker")
cp "$0" "$log_root/executed_run_stage_c_a800_seedvr2_lora_teacher.sh"
exec > >(tee -a "$log_root/seedvr2_lora_teacher.log") 2>&1

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
    echo "FAILED seedvr2_lora_teacher status=$status utc=$(date -u +%FT%TZ)"
  fi
  exit "$status"
}
trap fail EXIT INT TERM

(
  while true; do
    teacher_count=$(find "$teacher_root/samples" -maxdepth 1 -name '*.json' -type f 2>/dev/null | wc -l)
    router_done=0
    spatial_done=0
    [[ -s "$controller_root/controller_v5.pt" ]] && router_done=1
    [[ -s "$spatial_routes_root/manifest.json" ]] && spatial_done=1
    printf 'HEARTBEAT seedvr2_lora_teacher utc=%s elapsed_s=%s teacher=%s/560 router=%s spatial=%s ' \
      "$(date -u +%FT%TZ)" "$(($(date +%s) - experiment_start_epoch))" \
      "$teacher_count" "$router_done" "$spatial_done"
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

echo "START seedvr2_lora_teacher utc=$(date -u +%FT%TZ)"
rm -f "$run_root/run.failed"
cd "$repo"
if [[ -n "$(git status --short)" ]]; then
  echo "ERROR repository must be clean before the frozen teacher rebuild" >&2
  git status --short
  exit 1
fi
for path in "$quality_source" "$roi_source" "$adapter" \
  "$source_selection" "$v1_checkpoint" "$joint_samples" "$old_spatial_routes" \
  "$dit" "$vae" "$positive" "$negative" "$codec_image" "$codec_video"; do
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

"$python_bin" demo/stage_c_seedvr2_lora_teacher.py self-test
if [[ ! -s "$plan" ]]; then
  "$python_bin" demo/stage_c_seedvr2_lora_teacher.py plan \
    --source-quality-manifest "$quality_source" \
    --roi-cost-manifest "$roi_source" \
    --lora-checkpoint "$adapter" --lora-strength 0.5 \
    --model-path-i "$codec_image" --model-path-p "$codec_video" \
    --dit-checkpoint "$dit" --vae-checkpoint "$vae" \
    --positive-embedding "$positive" --negative-embedding "$negative" \
    --output "$plan"
fi

warm_seedvr2_file_cache

# First write one REDS and one UVG item into the formal output tree.  Both are
# also frozen-strength cache replays, so any condition mismatch stops here.
smoke_complete=$(
  "$python_bin" - "$teacher_root/manifest.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
try:
    d = json.loads(p.read_text())
    print(int(d.get("complete") is True and d.get("completed_sample_count") >= 2))
except Exception:
    print(0)
PY
)
if [[ "$smoke_complete" -ne 1 ]]; then
  "$torchrun_bin" --standalone --nproc-per-node=1 \
    demo/stage_c_seedvr2_lora_teacher.py relabel \
    --plan "$plan" --output-dir "$teacher_root" \
    --limit-per-dataset 1 --frozen-metric-tolerance 0.00001
fi

# Resume the same output tree to all 560 samples.  Per-sample JSON is atomic,
# so a host restart only repeats the current unfinished item.
"$torchrun_bin" --standalone --nproc-per-node=1 \
  demo/stage_c_seedvr2_lora_teacher.py relabel \
  --plan "$plan" --output-dir "$teacher_root" \
  --frozen-metric-tolerance 0.00001 --max-wall-seconds 43000

"$python_bin" - "$teacher_root/manifest.json" <<'PY'
import json, sys
from pathlib import Path
d = json.loads(Path(sys.argv[1]).read_text())
assert d["complete"] and d["completed_sample_count"] == 560
assert d["dataset_sample_counts"] == {"REDS": 500, "UVG": 60}
PY

"$python_bin" demo/stage_c_seedvr2_lora_teacher.py summarize \
  --plan "$plan" --teacher-manifest "$teacher_root/manifest.json" \
  --output-dir "$formal_root"

if [[ ! -s "$selection" ]]; then
  "$python_bin" demo/stage_c_seedvr2_lora_teacher.py make-router-selection \
    --source-selection "$source_selection" --plan "$plan" --output "$selection"
fi
if [[ ! -s "$controller_root/controller_v5.pt" ]]; then
  "$python_bin" demo/stage_c_a800_low_budget_v5.py train \
    --train-teacher-manifest "$teacher_root/manifest.json" \
    --train-roi-cost-manifest "$roi_source" \
    --selection-summary "$selection" --v1-checkpoint "$v1_checkpoint" \
    --output-dir "$controller_root" --epochs 240 --batch-size 512 \
    --learning-rate 0.001 --weight-decay 0.0001 \
    --base-seed 20260917 --correction-seed 20260919
fi
if [[ ! -s "$routes_root/manifest.json" ]]; then
  "$python_bin" demo/stage_c_a800_low_budget_v5.py route \
    --sample-manifest "$joint_samples" --base-probe-root "$base_probe_root" \
    --checkpoint "$controller_root/controller_v5.pt" --output-dir "$routes_root"
fi

"$python_bin" demo/stage_c_a800_spatial_consistency.py self-test
if [[ ! -s "$formal_root/spatial_sweep.json" ]]; then
  "$python_bin" demo/stage_c_a800_spatial_consistency.py sweep \
    --input-manifest "$routes_root/manifest.json" \
    --output "$formal_root/spatial_sweep.json" --expected-sample-count 37
fi
if [[ ! -s "$spatial_routes_root/manifest.json" ]]; then
  "$python_bin" demo/stage_c_a800_spatial_consistency.py route \
    --input-manifest "$routes_root/manifest.json" \
    --output-dir "$spatial_routes_root" --spatial-lambda 0.004 \
    --expected-sample-count 37
fi
if [[ ! -s "$formal_root/route_comparison_vs_previous_v6.json" ]]; then
  "$python_bin" demo/stage_c_a800_uvg_adaptation.py compare-routes \
    --sample-manifest "$joint_samples" --baseline-routes "$old_spatial_routes" \
    --candidate-routes "$spatial_routes_root/manifest.json" \
    --output "$formal_root/route_comparison_vs_previous_v6.json"
fi

"$python_bin" - "$run_root" "$experiment_start_epoch" <<'PY'
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1]).resolve()
start = int(sys.argv[2])
teacher = json.loads((root / "formal/summary.json").read_text())
route = json.loads(
    (root / "formal/route_comparison_vs_previous_v6.json").read_text())
gpu_samples = root / "logs/gpu_samples.csv"
peak = 0
gpu_indices = set()
if gpu_samples.is_file():
    for line in gpu_samples.read_text(encoding="utf-8").splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) >= 3:
            try:
                gpu_indices.add(int(fields[1]))
                peak = max(peak, int(float(fields[2])))
            except ValueError:
                pass
mounts = {}
for name in ("/root", "/root/autodl-tmp", "/root/autodl-fs"):
    value = shutil.disk_usage(name)
    mounts[name] = {
        "total_bytes": value.total,
        "used_bytes": value.used,
        "free_bytes": value.free,
    }
files = [path for path in root.rglob("*") if path.is_file()]
summary = {
    "experiment": "SeedVR2 LoRA teacher rebuild and router retraining",
    "status": "complete",
    "completed_utc": datetime.now(timezone.utc).isoformat(),
    "wall_seconds": int(__import__("time").time()) - start,
    "teacher_summary": teacher,
    "route_comparison": route,
    "resources": {
        "visible_gpu_indices_observed": sorted(gpu_indices),
        "peak_nvidia_smi_used_memory_mib": peak,
        "ordinary_file_count": len(files),
        "ordinary_file_bytes": sum(path.stat().st_size for path in files),
        "du_bytes": int(__import__("subprocess").check_output(
            ["du", "-sb", str(root)], text=True).split()[0]),
        "mounts": mounts,
    },
    "next_step": (
        "run the fixed 37-sample real-stream evaluation with the retrained "
        "spatial routes and SeedVR2 LoRA strength 0.50"
    ),
    "single_gpu": True,
}
target = root / "run_summary.json"
temporary = target.with_suffix(".json.tmp")
temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
os.replace(temporary, target)
PY

printf 'complete\n' > "$run_root/run.complete.tmp"
mv "$run_root/run.complete.tmp" "$run_root/run.complete"
cleanup
trap - EXIT
du -sb "$run_root"
df -h /root /root/autodl-tmp /root/autodl-fs
echo "COMPLETE seedvr2_lora_teacher utc=$(date -u +%FT%TZ) elapsed_s=$(($(date +%s) - experiment_start_epoch))"
