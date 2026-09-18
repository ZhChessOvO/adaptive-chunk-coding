#!/usr/bin/env bash
set -Eeuo pipefail

persist_root=/root/autodl-fs/DCVC
download_root="$persist_root/downloads/uvg"
output_root="$persist_root/assets/evaluation/UVG"
log_root="$persist_root/runs/a800_low_budget_v5_20260918/logs"
python_bin=/root/autodl-tmp/DCVC/envs/dcvcuf/bin/python
bsdtar_bin=/root/miniconda3/bin/bsdtar

mkdir -p "$download_root" "$output_root" "$log_root"
exec > >(tee -a "$log_root/restore_uvg.log") 2>&1

names=(Beauty Bosphorus HoneyBee Jockey ReadySetGo ShakeNDry YachtRide)
archive_sizes=(925430047 680772328 906770507 770631599 832143797 460046003 724220168)
frame_counts=(600 600 600 600 600 300 600)
download_chunk_bytes=$((1 * 1024 * 1024))
download_parallelism=12
download_pids=()

download_one_chunk() {
  local url=$1
  local output=$2
  local start_offset=$3
  local end_offset=$4
  local expected_chunk_size=$((end_offset - start_offset + 1))
  local temporary="${output}.tmp"
  local actual_chunk_size

  if [[ -e "$output" ]]; then
    actual_chunk_size=$(stat -c %s "$output")
    if [[ "$actual_chunk_size" -eq "$expected_chunk_size" ]]; then
      echo "DOWNLOAD skip_complete range=${start_offset}-${end_offset}"
      return
    fi
    unlink "$output"
  fi
  if [[ -e "$temporary" ]]; then
    unlink "$temporary"
  fi

  echo "DOWNLOAD start range=${start_offset}-${end_offset} expected_bytes=$expected_chunk_size"
  env \
    -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    -u ALL_PROXY -u all_proxy -u FTP_PROXY -u ftp_proxy \
    -u HF_ENDPOINT -u HUGGINGFACE_HUB_CACHE \
    curl --noproxy '*' --silent --show-error -L --fail \
      --retry 20 --retry-all-errors --connect-timeout 30 \
      --speed-time 120 --speed-limit 1024 \
      --range "${start_offset}-${end_offset}" -o "$temporary" \
      --write-out \
      "DOWNLOAD transfer range=${start_offset}-${end_offset} http=%{http_code} bytes=%{size_download} speed_Bps=%{speed_download}\n" \
      "$url"

  actual_chunk_size=$(stat -c %s "$temporary")
  if [[ "$actual_chunk_size" -ne "$expected_chunk_size" ]]; then
    echo "chunk size mismatch for range ${start_offset}-${end_offset}: $actual_chunk_size != $expected_chunk_size" >&2
    return 1
  fi
  mv "$temporary" "$output"
  echo "DOWNLOAD complete range=${start_offset}-${end_offset} bytes=$actual_chunk_size"
}

download_worker() {
  local url=$1
  local chunk_dir=$2
  local expected_size=$3
  local index=$4
  local start_offset
  local end_offset
  local chunk_path

  while true; do
    start_offset=$((index * download_chunk_bytes))
    if (( start_offset >= expected_size )); then
      return
    fi
    end_offset=$((start_offset + download_chunk_bytes - 1))
    if (( end_offset >= expected_size )); then
      end_offset=$((expected_size - 1))
    fi
    printf -v chunk_path '%s/%06d.part' "$chunk_dir" "$index"
    download_one_chunk \
      "$url" "$chunk_path" "$start_offset" "$end_offset"
    index=$((index + download_parallelism))
  done
}

download_in_chunks() {
  local url=$1
  local partial=$2
  local expected_size=$3
  local chunk_dir="${partial}.chunks"
  local chunk_path
  local start_offset=0
  local end_offset
  local expected_chunk_size
  local actual_chunk_size
  local index=0
  local failed=0
  local pid
  local worker

  if [[ -e "$partial" ]]; then
    actual_chunk_size=$(stat -c %s "$partial")
    if [[ "$actual_chunk_size" -eq "$expected_size" ]]; then
      echo "DOWNLOAD assembled_file_complete bytes=$actual_chunk_size"
      return
    fi
  fi
  mkdir -p "$chunk_dir"

  for ((worker = 0; worker < download_parallelism; worker++)); do
    download_worker "$url" "$chunk_dir" "$expected_size" "$worker" &
    download_pids+=("$!")
  done

  for pid in "${download_pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  download_pids=()
  if (( failed != 0 )); then
    echo "one or more download chunks failed; complete chunks remain for resume" >&2
    return 1
  fi

  if [[ -e "$partial" ]]; then
    unlink "$partial"
  fi
  start_offset=0
  index=0
  while (( start_offset < expected_size )); do
    end_offset=$((start_offset + download_chunk_bytes - 1))
    if (( end_offset >= expected_size )); then
      end_offset=$((expected_size - 1))
    fi
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
    echo "assembled download size mismatch: $actual_chunk_size != $expected_size" >&2
    return 1
  fi
  for chunk_path in "$chunk_dir"/*.part; do
    unlink "$chunk_path"
  done
  rmdir "$chunk_dir"
  echo "DOWNLOAD assembled bytes=$actual_chunk_size"
}

heartbeat_pid=
cleanup() {
  local pid
  for pid in "${download_pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${download_pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  if [[ -n "$heartbeat_pid" ]]; then kill "$heartbeat_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM
(
  while true; do
    complete_sequences=$(find "$output_root" -mindepth 2 -maxdepth 2 \
      -name source_manifest.json | wc -l)
    saved_chunks=$(find "$download_root" -mindepth 2 -maxdepth 2 \
      -type f -name '*.part' | wc -l)
    printf 'HEARTBEAT restore_uvg utc=%s sequences=%s/7 saved_yacht_chunks=%s/691 ' \
      "$(date -u +%FT%TZ)" "$complete_sequences" "$saved_chunks"
    df -h /root /root/autodl-tmp /root/autodl-fs | tail -n +2 | tr '\n' ';'
    printf '\n'
    sleep 60
  done
) >> "$log_root/restore_uvg_heartbeat.log" 2>&1 &
heartbeat_pid=$!

for index in "${!names[@]}"; do
  name=${names[$index]}
  expected_archive_size=${archive_sizes[$index]}
  frame_count=${frame_counts[$index]}
  url="https://ultravideo.fi/video/${name}_1920x1080_120fps_420_8bit_YUV_RAW.7z"
  archive="$download_root/${name}.7z"
  partial="$archive.part"
  sample_root="$output_root/$name"
  manifest="$sample_root/source_manifest.json"
  raw_root="$download_root/raw-$name"

  if [[ -s "$manifest" ]]; then
    count=$(find "$sample_root" -maxdepth 1 -type f -name '*.png' | wc -l)
    if [[ "$count" -eq 17 ]]; then
      echo "SKIP UVG $name already complete"
      continue
    fi
  fi

  echo "START UVG $name utc=$(date -u +%FT%TZ)"
  download_in_chunks "$url" "$partial" "$expected_archive_size"
  actual_archive_size=$(stat -c %s "$partial")
  if [[ "$actual_archive_size" -ne "$expected_archive_size" ]]; then
    echo "archive size mismatch for $name: $actual_archive_size != $expected_archive_size" >&2
    exit 1
  fi
  mv "$partial" "$archive"
  archive_sha256=$(sha256sum "$archive" | awk '{print $1}')

  mkdir -p "$raw_root" "$sample_root"
  "$bsdtar_bin" -xf "$archive" -C "$raw_root"
  mapfile -t raw_files < <(find "$raw_root" -type f -name '*.yuv' | sort)
  if [[ "${#raw_files[@]}" -ne 1 ]]; then
    echo "expected one YUV in $archive, found ${#raw_files[@]}" >&2
    exit 1
  fi
  raw_file=${raw_files[0]}
  expected_raw_size=$((1920 * 1080 * 3 / 2 * frame_count))
  actual_raw_size=$(stat -c %s "$raw_file")
  if [[ "$actual_raw_size" -ne "$expected_raw_size" ]]; then
    echo "raw YUV size mismatch for $name: $actual_raw_size != $expected_raw_size" >&2
    exit 1
  fi

  ffmpeg -hide_banner -loglevel error -y \
    -f rawvideo -pixel_format yuv420p -video_size 1920x1080 -framerate 120 \
    -i "$raw_file" -vf crop=512:512:704:284 -frames:v 17 \
    -start_number 0 "$sample_root/%08d.png"

  "$python_bin" - "$sample_root" "$manifest" "$name" "$url" \
    "$expected_archive_size" "$archive_sha256" "$frame_count" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image

sample_root = Path(sys.argv[1])
manifest = Path(sys.argv[2])
name = sys.argv[3]
paths = sorted(sample_root.glob("*.png"))
if len(paths) != 17:
    raise RuntimeError(f"{name}: expected 17 PNG files, found {len(paths)}")
for path in paths:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        if image.size != (512, 512):
            raise RuntimeError(f"{path}: unexpected size {image.size}")
value = {
    "dataset": "UVG",
    "official_sequence": name,
    "official_source_url": sys.argv[4],
    "archive_bytes": int(sys.argv[5]),
    "downloaded_archive_sha256": sys.argv[6],
    "original_format": {
        "width": 1920,
        "height": 1080,
        "fps": 120,
        "pixel_format": "yuv420p8",
        "frame_count": int(sys.argv[7]),
    },
    "evaluation_sample": {
        "frame_start": 0,
        "frame_count": 17,
        "crop": {"x": 704, "y": 284, "width": 512, "height": 512},
        "png_files": [str(path.resolve()) for path in paths],
    },
    "source_role": (
        "cross-distribution paper evaluation; UVG family and some aliases "
        "were used historically, so exact novelty is reported per sequence"
    ),
    "created_utc": datetime.now(timezone.utc).isoformat(),
}
temporary = manifest.with_suffix(".json.tmp")
temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, manifest)
print(json.dumps({"sequence": name, "png_count": len(paths)}, indent=2))
PY

  unlink "$raw_file"
  find "$raw_root" -depth -type d -empty -delete
  unlink "$archive"
  echo "COMPLETE UVG $name utc=$(date -u +%FT%TZ)"
done

echo "UVG restore complete"
du -sh "$output_root"
df -h /root /root/autodl-tmp /root/autodl-fs
