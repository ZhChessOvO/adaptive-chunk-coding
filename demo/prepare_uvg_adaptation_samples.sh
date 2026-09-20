#!/usr/bin/env bash
set -Eeuo pipefail

# Restore a small, deliberately documented UVG adaptation set.  The five
# sequences below are training-side for v6; ReadySetGo and YachtRide are not
# downloaded or touched by this script and remain v6 hold-out sequences.

persist_root=/root/autodl-fs/DCVC
download_root="$persist_root/downloads/uvg-adaptation"
output_root="$persist_root/data/UVG_adaptation"
run_root="$persist_root/runs/a800_uvg_adaptation_20260919"
log_root="$run_root/logs"
python_bin=/root/autodl-tmp/DCVC/envs/dcvcuf/bin/python
bsdtar_bin=/root/miniconda3/bin/bsdtar

names=(Beauty Bosphorus HoneyBee Jockey ShakeNDry)
archive_sizes=(925430047 680772328 906770507 770631599 460046003)
frame_counts=(600 600 600 600 300)
crop_names=(left center right)
crop_xs=(128 704 1280)
crop_y=284
frame_width=1920
frame_height=1080
frame_bytes=$((frame_width * frame_height * 3 / 2))
window_frames=17
download_chunk_bytes=$((1 * 1024 * 1024))
download_parallelism=20
allow_download=${UVG_ALLOW_DOWNLOAD:-1}
download_pids=()
start_epoch=$(date +%s)

mkdir -p "$download_root" "$output_root/samples" "$log_root"
exec > >(tee -a "$log_root/prepare_uvg_adaptation.log") 2>&1

download_one_chunk() {
  local url=$1
  local output=$2
  local start_offset=$3
  local end_offset=$4
  local expected_chunk_size=$((end_offset - start_offset + 1))
  local temporary="${output}.tmp"
  local piece="${temporary}.piece"
  local headers="${temporary}.headers"
  local actual_chunk_size=0
  local piece_size request_start remaining_size http_code curl_status stale
  local empty_failures=0

  if [[ -e "$output" ]]; then
    actual_chunk_size=$(stat -c %s "$output")
    if [[ "$actual_chunk_size" -eq "$expected_chunk_size" ]]; then
      return
    fi
    unlink "$output"
  fi
  if [[ -e "$temporary" ]]; then
    actual_chunk_size=$(stat -c %s "$temporary")
  else
    truncate -s 0 "$temporary"
  fi
  if (( actual_chunk_size > expected_chunk_size )); then
    unlink "$temporary"
    truncate -s 0 "$temporary"
    actual_chunk_size=0
  fi

  while (( actual_chunk_size < expected_chunk_size )); do
    request_start=$((start_offset + actual_chunk_size))
    remaining_size=$((expected_chunk_size - actual_chunk_size))
    for stale in "$piece" "$headers"; do
      if [[ -e "$stale" ]]; then unlink "$stale"; fi
    done
    if http_code=$(env \
      -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
      -u ALL_PROXY -u all_proxy -u FTP_PROXY -u ftp_proxy \
      -u HF_ENDPOINT -u HUGGINGFACE_HUB_CACHE \
      curl --noproxy '*' --silent --show-error -L --fail \
        --connect-timeout 30 --speed-time 60 --speed-limit 1024 \
        --range "${request_start}-${end_offset}" \
        --dump-header "$headers" -o "$piece" \
        --write-out '%{http_code}' "$url"); then
      curl_status=0
    else
      curl_status=$?
    fi

    piece_size=0
    if [[ -e "$piece" ]]; then piece_size=$(stat -c %s "$piece"); fi
    if [[ "$http_code" == 206 ]] \
      && [[ "$piece_size" -gt 0 ]] \
      && [[ "$piece_size" -le "$remaining_size" ]] \
      && rg -q -i \
        "^content-range: bytes ${request_start}-${end_offset}/" "$headers"; then
      dd if="$piece" of="$temporary" oflag=append conv=notrunc status=none
      unlink "$piece"
      unlink "$headers"
      actual_chunk_size=$(stat -c %s "$temporary")
      empty_failures=0
    else
      for stale in "$piece" "$headers"; do
        if [[ -e "$stale" ]]; then unlink "$stale"; fi
      done
      empty_failures=$((empty_failures + 1))
      echo "DOWNLOAD retry range=${request_start}-${end_offset} http=${http_code:-000} curl_status=$curl_status received_bytes=$piece_size failures=$empty_failures" >&2
      if (( empty_failures >= 100 )); then
        echo "too many empty or invalid range responses" >&2
        return 1
      fi
      sleep 2
    fi
  done

  mv "$temporary" "$output"
}

download_worker() {
  local url=$1
  local chunk_dir=$2
  local expected_size=$3
  local index=$4
  local start_offset end_offset chunk_path

  while true; do
    start_offset=$((index * download_chunk_bytes))
    if (( start_offset >= expected_size )); then return; fi
    end_offset=$((start_offset + download_chunk_bytes - 1))
    if (( end_offset >= expected_size )); then end_offset=$((expected_size - 1)); fi
    printf -v chunk_path '%s/%06d.part' "$chunk_dir" "$index"
    download_one_chunk "$url" "$chunk_path" "$start_offset" "$end_offset"
    index=$((index + download_parallelism))
  done
}

cleanup_completed_chunk_dir() {
  local chunk_dir=$1
  local artifact
  if [[ ! -d "$chunk_dir" ]]; then return; fi
  shopt -s nullglob
  for artifact in \
    "$chunk_dir"/*.part \
    "$chunk_dir"/*.part.tmp \
    "$chunk_dir"/*.part.tmp.headers \
    "$chunk_dir"/*.part.tmp.piece; do
    if [[ -e "$artifact" ]]; then unlink "$artifact"; fi
  done
  shopt -u nullglob
  rmdir "$chunk_dir"
}

download_in_chunks() {
  local url=$1
  local partial=$2
  local expected_size=$3
  local chunk_dir="${partial}.chunks"
  local chunk_path start_offset=0 end_offset expected_chunk_size actual_chunk_size
  local index=0 failed=0 pid worker

  if [[ -e "$partial" ]]; then
    actual_chunk_size=$(stat -c %s "$partial")
    if [[ "$actual_chunk_size" -eq "$expected_size" ]]; then
      cleanup_completed_chunk_dir "$chunk_dir"
      return
    fi
  fi
  mkdir -p "$chunk_dir"
  for ((worker = 0; worker < download_parallelism; worker++)); do
    download_worker "$url" "$chunk_dir" "$expected_size" "$worker" &
    download_pids+=("$!")
  done
  for pid in "${download_pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  download_pids=()
  if (( failed != 0 )); then
    echo "one or more download chunks failed; completed chunks remain for resume" >&2
    return 1
  fi

  if [[ -e "$partial" ]]; then unlink "$partial"; fi
  while (( start_offset < expected_size )); do
    end_offset=$((start_offset + download_chunk_bytes - 1))
    if (( end_offset >= expected_size )); then end_offset=$((expected_size - 1)); fi
    expected_chunk_size=$((end_offset - start_offset + 1))
    printf -v chunk_path '%s/%06d.part' "$chunk_dir" "$index"
    actual_chunk_size=$(stat -c %s "$chunk_path")
    if [[ "$actual_chunk_size" -ne "$expected_chunk_size" ]]; then
      echo "saved chunk size mismatch: $chunk_path" >&2
      return 1
    fi
    dd if="$chunk_path" of="$partial" oflag=append conv=notrunc status=none
    start_offset=$((end_offset + 1))
    index=$((index + 1))
  done
  actual_chunk_size=$(stat -c %s "$partial")
  if [[ "$actual_chunk_size" -ne "$expected_size" ]]; then
    echo "assembled archive size mismatch: $actual_chunk_size != $expected_size" >&2
    return 1
  fi
  cleanup_completed_chunk_dir "$chunk_dir"
}

sequence_is_complete() {
  "$python_bin" - "$output_root/samples" "$1" <<'PY'
import json
import sys
from pathlib import Path
from PIL import Image

root = Path(sys.argv[1])
name = sys.argv[2].lower()
manifests = sorted(root.glob(f"uvg-{name}-*/source_manifest.json"))
if len(manifests) != 12:
    raise SystemExit(1)
for manifest_path in manifests:
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = [Path(path) for path in value["source_files"]]
    if len(paths) != 17 or any(not path.is_file() for path in paths):
        raise SystemExit(1)
    for path in paths:
        with Image.open(path) as image:
            if image.size != (512, 512):
                raise SystemExit(1)
PY
}

heartbeat_pid=
cleanup() {
  local pid
  for pid in "${download_pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${download_pids[@]}"; do wait "$pid" 2>/dev/null || true; done
  if [[ -n "$heartbeat_pid" ]]; then kill "$heartbeat_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

(
  while true; do
    complete_sequences=$(find "$output_root" -maxdepth 1 -type f -name '*.complete' | wc -l)
    complete_samples=$(find "$output_root/samples" -mindepth 2 -maxdepth 2 -type f -name source_manifest.json | wc -l)
    saved_chunks=$(find "$download_root" -mindepth 2 -maxdepth 2 -type f -name '*.part' | wc -l)
    printf 'HEARTBEAT uvg_adaptation utc=%s elapsed_s=%s sequences=%s/5 samples=%s/60 chunks=%s ' \
      "$(date -u +%FT%TZ)" "$(($(date +%s) - start_epoch))" \
      "$complete_sequences" "$complete_samples" "$saved_chunks"
    df -h /root /root/autodl-tmp /root/autodl-fs | tail -n +2 | tr '\n' ';'
    printf '\n'
    sleep 60
  done
) >> "$log_root/heartbeat.log" 2>&1 &
heartbeat_pid=$!

cp "$0" "$log_root/executed_prepare_uvg_adaptation_samples.sh"
echo "START UVG adaptation preparation utc=$(date -u +%FT%TZ)"
echo "TRAIN sequences: ${names[*]}"
echo "HOLDOUT sequences not touched: ReadySetGo YachtRide"
df -h /root /root/autodl-tmp /root/autodl-fs

for index in "${!names[@]}"; do
  name=${names[$index]}
  expected_archive_size=${archive_sizes[$index]}
  frame_count=${frame_counts[$index]}
  marker="$output_root/${name}.complete"

  if [[ -f "$marker" ]] && sequence_is_complete "$name"; then
    echo "SKIP complete sequence $name"
    continue
  fi
  if [[ -e "$marker" ]]; then unlink "$marker"; fi

  if [[ "$frame_count" -eq 600 ]]; then
    starts=(32 160 288 416)
  else
    starts=(32 96 160 224)
  fi
  url="https://ultravideo.fi/video/${name}_1920x1080_120fps_420_8bit_YUV_RAW.7z"
  archive="$download_root/${name}.7z"
  uploaded_archive="$download_root/${name}_1920x1080_120fps_420_8bit_YUV_RAW.7z"
  partial="$archive.part"
  raw_root="$download_root/raw-$name"

  echo "START sequence=$name utc=$(date -u +%FT%TZ)"
  if [[ ! -e "$archive" ]] && [[ -e "$uploaded_archive" ]]; then
    actual_uploaded_size=$(stat -c %s "$uploaded_archive")
    if [[ "$actual_uploaded_size" -ne "$expected_archive_size" ]]; then
      echo "uploaded archive size mismatch: $uploaded_archive ($actual_uploaded_size != $expected_archive_size)" >&2
      exit 1
    fi
    echo "DOWNLOAD adopt user-uploaded archive $uploaded_archive"
    mv "$uploaded_archive" "$archive"
  fi
  if [[ -e "$archive" ]] \
    && [[ "$(stat -c %s "$archive")" -eq "$expected_archive_size" ]]; then
    echo "DOWNLOAD reuse assembled archive $archive"
    # A manually supplied complete archive takes precedence over resumable
    # chunks left by an interrupted cloud download.  Remove those stale
    # chunks only after the complete archive size has been verified.
    cleanup_completed_chunk_dir "${partial}.chunks"
  else
    if [[ "$allow_download" != "1" ]]; then
      echo "offline mode: complete archive unavailable for $name" >&2
      exit 1
    fi
    if [[ -e "$archive" ]]; then unlink "$archive"; fi
    download_in_chunks "$url" "$partial" "$expected_archive_size"
    mv "$partial" "$archive"
  fi
  archive_sha256=$(sha256sum "$archive" | awk '{print $1}')

  mkdir -p "$raw_root"
  "$bsdtar_bin" -xf "$archive" -C "$raw_root"
  mapfile -t raw_files < <(find "$raw_root" -type f -name '*.yuv' | sort)
  if [[ "${#raw_files[@]}" -ne 1 ]]; then
    echo "expected one YUV in $archive, found ${#raw_files[@]}" >&2
    exit 1
  fi
  raw_file=${raw_files[0]}
  expected_raw_size=$((frame_bytes * frame_count))
  actual_raw_size=$(stat -c %s "$raw_file")
  if [[ "$actual_raw_size" -ne "$expected_raw_size" ]]; then
    echo "raw YUV size mismatch for $name: $actual_raw_size != $expected_raw_size" >&2
    exit 1
  fi

  for frame_start in "${starts[@]}"; do
    if (( frame_start + window_frames > frame_count )); then
      echo "invalid frame window for $name: $frame_start" >&2
      exit 1
    fi
    for crop_index in "${!crop_names[@]}"; do
      crop_name=${crop_names[$crop_index]}
      crop_x=${crop_xs[$crop_index]}
      lower_name=${name,,}
      printf -v sample_id 'uvg-%s-f%03d-%s' "$lower_name" "$frame_start" "$crop_name"
      sample_root="$output_root/samples/$sample_id"
      manifest="$sample_root/source_manifest.json"
      mkdir -p "$sample_root"

      if "$python_bin" - "$manifest" >/dev/null 2>&1 <<'PY'
import json
import sys
from pathlib import Path
from PIL import Image

path = Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8"))
frames = [Path(item) for item in value["source_files"]]
assert len(frames) == 17
for frame in frames:
    with Image.open(frame) as image:
        assert image.size == (512, 512)
PY
      then
        echo "SAMPLE skip_complete $sample_id"
        continue
      fi

      shopt -s nullglob
      for stale in "$sample_root"/*.png "$sample_root"/*.json "$sample_root"/*.tmp; do
        if [[ -e "$stale" ]]; then unlink "$stale"; fi
      done
      shopt -u nullglob
      skip_bytes=$((frame_start * frame_bytes))
      ffmpeg -hide_banner -loglevel error -y \
        -f rawvideo -pixel_format yuv420p -video_size 1920x1080 \
        -framerate 120 -skip_initial_bytes "$skip_bytes" -i "$raw_file" \
        -vf "crop=512:512:${crop_x}:${crop_y}" -frames:v "$window_frames" \
        -start_number 0 "$sample_root/%08d.png"

      "$python_bin" - "$sample_root" "$manifest" "$sample_id" "$name" \
        "$url" "$expected_archive_size" "$archive_sha256" "$frame_count" \
        "$frame_start" "$crop_name" "$crop_x" "$crop_y" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image

sample_root = Path(sys.argv[1]).resolve()
manifest = Path(sys.argv[2])
paths = sorted(sample_root.glob("*.png"))
if len(paths) != 17:
    raise RuntimeError(f"expected 17 PNG files, found {len(paths)}")
digest = hashlib.sha256()
for path in paths:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        if image.size != (512, 512):
            raise RuntimeError(f"{path}: unexpected size {image.size}")
    digest.update(path.read_bytes())
value = {
    "sample_id": sys.argv[3],
    "dataset": "UVG",
    "split": "v6_adaptation_train",
    "data_role": "v6 cross-domain adaptation training; not independent evidence",
    "source_role": (
        "UVG training-side sequence for v6; official sequence family was "
        "already observed during v5 evaluation"
    ),
    "sequence": sys.argv[4],
    "source_dir": str(sample_root),
    "source_files": [str(path) for path in paths],
    "frame_start": int(sys.argv[9]),
    "frame_count": 17,
    "crop": {"x": 0, "y": 0, "width": 512, "height": 512},
    "original_crop": {
        "name": sys.argv[10], "x": int(sys.argv[11]),
        "y": int(sys.argv[12]), "width": 512, "height": 512,
    },
    "selected_source_sha256": digest.hexdigest(),
    "official_source_url": sys.argv[5],
    "archive_bytes": int(sys.argv[6]),
    "downloaded_archive_sha256": sys.argv[7],
    "original_format": {
        "width": 1920, "height": 1080, "fps": 120,
        "pixel_format": "yuv420p8", "frame_count": int(sys.argv[8]),
    },
    "created_utc": datetime.now(timezone.utc).isoformat(),
}
temporary = manifest.with_suffix(".json.tmp")
temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, manifest)
PY
      echo "SAMPLE complete $sample_id"
    done
  done

  if ! sequence_is_complete "$name"; then
    echo "sequence validation failed: $name" >&2
    exit 1
  fi
  touch "$marker"

  unlink "$raw_file"
  while IFS= read -r -d '' residual; do unlink "$residual"; done \
    < <(find "$raw_root" -type f -print0)
  find "$raw_root" -depth -type d -empty -exec rmdir {} \;
  unlink "$archive"
  echo "COMPLETE sequence=$name utc=$(date -u +%FT%TZ)"
done

"$python_bin" - "$output_root" "$run_root" <<'PY'
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image

root = Path(sys.argv[1]).resolve()
run_root = Path(sys.argv[2]).resolve()
expected_sequences = {"Beauty", "Bosphorus", "HoneyBee", "Jockey", "ShakeNDry"}
holdout_sequences = {"ReadySetGo", "YachtRide"}
manifests = sorted((root / "samples").glob("*/source_manifest.json"))
if len(manifests) != 60:
    raise RuntimeError(f"expected 60 samples, found {len(manifests)}")
records = []
for path in manifests:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value["sequence"] not in expected_sequences:
        raise RuntimeError(f"unexpected training sequence: {value['sequence']}")
    if value["sequence"] in holdout_sequences:
        raise RuntimeError("hold-out sequence entered adaptation data")
    frames = [Path(frame) for frame in value["source_files"]]
    if len(frames) != 17:
        raise RuntimeError(f"wrong frame count: {value['sample_id']}")
    for frame in frames:
        with Image.open(frame) as image:
            if image.size != (512, 512):
                raise RuntimeError(f"invalid frame: {frame}")
    records.append(value)
ids = [record["sample_id"] for record in records]
if len(ids) != len(set(ids)):
    raise RuntimeError("duplicate sample IDs")
counts = Counter(record["sequence"] for record in records)
if counts != Counter({name: 12 for name in expected_sequences}):
    raise RuntimeError(f"unexpected per-sequence counts: {counts}")

jsonl_path = root / "uvg_adaptation_samples.jsonl"
temporary = jsonl_path.with_suffix(".jsonl.tmp")
temporary.write_text(
    "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
    encoding="utf-8",
)
os.replace(temporary, jsonl_path)
manifest_sha = hashlib.sha256(jsonl_path.read_bytes()).hexdigest()
summary = {
    "dataset": "UVG v6 cross-domain adaptation subset",
    "status": "complete",
    "sample_count": len(records),
    "frames_per_sample": 17,
    "training_sequences": sorted(expected_sequences),
    "v6_holdout_sequences_not_in_manifest": sorted(holdout_sequences),
    "samples_per_training_sequence": dict(sorted(counts.items())),
    "temporal_windows_per_sequence": 4,
    "spatial_crops_per_window": 3,
    "manifest": str(jsonl_path),
    "manifest_sha256": manifest_sha,
    "ordinary_file_bytes": sum(
        path.stat().st_size for path in root.rglob("*") if path.is_file()
    ),
    "scientific_boundary": {
        "all_seven_uvg_first_windows_were_seen_in_v5_evaluation": True,
        "holdout_means_not_used_for_v6_training_not_never_seen": True,
        "dcvc_uf_frozen": True,
        "seedvr2_frozen": True,
        "spatial_qp_codec_frozen": True,
    },
    "created_utc": datetime.now(timezone.utc).isoformat(),
}
summary_path = root / "summary.json"
temporary = summary_path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, summary_path)
complete = root / "dataset.complete"
temporary = complete.with_suffix(".complete.tmp")
temporary.write_text(json.dumps({"complete": True, "samples": 60}) + "\n")
os.replace(temporary, complete)

run_root.mkdir(parents=True, exist_ok=True)
usage = shutil.disk_usage(root)
snapshot = {
    "experiment": "UVG v6 adaptation data preparation",
    "elapsed_seconds": None,
    "output_root": str(root),
    "ordinary_file_bytes": summary["ordinary_file_bytes"],
    "filesystem": {
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    },
    "completed_utc": datetime.now(timezone.utc).isoformat(),
}
snapshot_path = run_root / "data_resource_snapshot.json"
temporary = snapshot_path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, snapshot_path)
print(json.dumps(summary, indent=2))
PY

echo "COMPLETE UVG adaptation preparation utc=$(date -u +%FT%TZ) elapsed_s=$(($(date +%s) - start_epoch))"
du -sh "$output_root"
df -h /root /root/autodl-tmp /root/autodl-fs
