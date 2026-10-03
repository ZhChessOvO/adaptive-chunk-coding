"""CPU-only, read-only byte decomposition of completed RTVC v1 experiments.

This is accounting, not a new codec experiment. It reads no video pixels,
weights, or quality metrics. Source artifacts are never written. Run the full
audit inside tmux; per-point atomic reports permit interruption and re-entry.
No mask-removal savings are invented: the existing RTVC format already sends
no separate E/G/protection mask, while packet addressing and shared controls
remain charged. Native UF bytes include its existing native framing.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import time
import zlib

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Lightweight format parsing only. Neither dependency imports torch or CUDA.
from demo import compact_enhancement_format as compact

RUNS = Path('/root/autodl-fs/DCVC/runs')
DEFAULT_ROOT = RUNS / 'routervc_20261003'
DEFAULT_OUTPUT = RUNS / 'routervc_revision_20261003/byte_audit'
FORMAT = 'routervc_actual_byte_audit_v1'
# Frozen RTVC v1 syntax: an independent reader avoids importing model loaders.
RTVC_HEADER = struct.Struct('<4sBII')
RTVC_CONTROL = struct.Struct('<QBddd4H')
HASH_NAMES = ('router', 'policy', 'dit', 'vae', 'positive', 'negative', 'lora', 'profile')
CONTROL_NAMES = ('seed', 'max_g', 'boundary_lambda', 'strength', 'blend',
                 'window', 'stride', 'context', 'feather')
BYTE_KEYS = ('native_uf_bytes', 'rtvc_shared_header_bytes', 'acse_header_bytes',
             'e_packet_header_bytes', 'e_payload_bytes', 'incomplete_tail_bytes')
MASK_KEYS = ('explicit_e_mask_bytes', 'explicit_g_mask_bytes', 'explicit_protection_mask_bytes')
CODE_FILES = ('routervc_byte_audit.py', 'routervc_format.py', 'scalable_format.py',
              'compact_enhancement_format.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def encoded_json(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def atomic_write(path, data):
    """Atomic output only; never replace a historical experiment file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(handle, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def immutable_write(path, data, *, verify_only=False):
    path = Path(path)
    if path.exists():
        require(path.read_bytes() == data, f'audit output changed: {path}')
    else:
        require(not verify_only, f'missing audit output: {path}')
        atomic_write(path, data)


def inspect_stream(wire, *, allow_incomplete_tail=False):
    """Validate outer closed schema and inner checksums, then count every byte."""
    require(len(wire) >= RTVC_HEADER.size, 'truncated RTVC header')
    magic, version, length, crc = RTVC_HEADER.unpack_from(wire)
    expected = RTVC_CONTROL.size + 32 * len(HASH_NAMES)
    require((magic, version, length) == (b'RTVC', 1, expected), 'unsupported RTVC schema')
    start = RTVC_HEADER.size + length
    require(start <= len(wire), 'truncated RTVC control')
    body = wire[RTVC_HEADER.size:start]
    require(zlib.crc32(body) == crc, 'RTVC checksum mismatch')
    config = dict(zip(CONTROL_NAMES, RTVC_CONTROL.unpack_from(body)))
    require(config['seed'] < 2**63 and config['max_g'] <= 16, 'invalid RTVC budget/seed')
    require(all(math.isfinite(config[k]) and 0 <= config[k] <= 1
                for k in ('boundary_lambda', 'strength', 'blend')), 'invalid RTVC controls')
    require(config['strength'] == 1 and tuple(config[k] for k in
            ('window', 'stride', 'context', 'feather')) == (17, 8, 64, 16),
            'unsupported RTVC generation profile')
    for i, name in enumerate(HASH_NAMES):
        offset = RTVC_CONTROL.size + 32 * i
        config[name] = body[offset:offset + 32].hex()
    inner = wire[start:]
    parsed = compact.parse(inner, allow_incomplete_tail=allow_incomplete_tail)
    packets = []
    position = parsed.base_end
    for packet in parsed.packets:
        framing = compact.PACKET.size
        metadata = compact.LOCAL.size
        require(len(packet.wire) == framing + metadata + len(packet.payload),
                'packet layout does not reconcile')
        require(packet.end_offset == position + len(packet.wire), 'noncontiguous packets')
        pm = packet.meta
        x, y, w, h = pm['roi']
        require(pm['start'] + pm['count'] <= parsed.meta['frame_count'] and
                x + w <= parsed.meta['width'] and y + h <= parsed.meta['height'],
                'packet coordinates exceed the video')
        packets.append(dict(packet_id=pm['packet_id'], start=pm['start'], count=pm['count'],
                            roi=pm['roi'], qstep=pm['qstep'], wire_offset=start + position,
                            framing_bytes=framing, metadata_bytes=metadata,
                            payload_bytes=len(packet.payload), wire_bytes=len(packet.wire),
                            wire_sha256=digest(packet.wire), payload_sha256=digest(packet.payload)))
        position = packet.end_offset
    counts = dict(native_uf_bytes=len(parsed.base), rtvc_shared_header_bytes=start,
                  acse_header_bytes=parsed.base_end - len(parsed.base),
                  e_packet_header_bytes=sum(p['framing_bytes'] + p['metadata_bytes'] for p in packets),
                  e_payload_bytes=sum(p['payload_bytes'] for p in packets),
                  incomplete_tail_bytes=parsed.incomplete_tail_bytes)
    require(sum(counts.values()) == len(wire), 'byte categories do not reconcile')
    pixels = parsed.meta['frame_count'] * parsed.meta['height'] * parsed.meta['width']
    return dict(format=FORMAT, bytes=counts, total_bytes=len(wire), bpp=8 * len(wire) / pixels,
                separate_mask_bytes={k: 0 for k in MASK_KEYS}, wire_sha256=digest(wire),
                native_uf_sha256=digest(parsed.base), rtvc_header_sha256=digest(wire[:start]),
                acse_header_sha256=digest(inner[:parsed.base_end - len(parsed.base)]),
                incomplete_tail_sha256=digest(inner[parsed.consumed_bytes:]),
                shape=[parsed.meta['frame_count'], parsed.meta['height'], parsed.meta['width']],
                config=config, packets=packets, packet_count=len(packets),
                schema_evidence=dict(rtvc_version=1, acse_version=2,
                    rtvc_control_fields=list(CONTROL_NAMES), rtvc_hash_fields=list(HASH_NAMES),
                    separate_region_mask_fields=[], packet_metadata_fields=sorted(compact.LOCAL_KEYS),
                    e_coordinates_are_charged=True, generator_policy_is_receiver_derived=True,
                    no_claim_of_new_rate_savings=True, all_shared_controls_and_headers_charged=True,
                    scope='No separate mask field; E location/time information is in charged packet headers.'))


def inspect_point(root, job, *, allow_incomplete_tail=False):
    root = Path(root).resolve()
    folder = Path(job['folder']).resolve()
    require(folder.is_relative_to(root) and folder != root, 'job folder escapes source experiment')
    paths = {name: folder / name for name in ('stream.rtvc', 'job.json', 'fresh/decode.json', 'result.json')}
    require(all(path.resolve().is_relative_to(root) for path in paths.values()), 'artifact escapes source root')
    require(read_json(paths['job.json']) == job, 'job manifest differs from per-point job')
    before = paths['stream.rtvc'].stat()
    wire = paths['stream.rtvc'].read_bytes()
    after = paths['stream.rtvc'].stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            'stream changed while reading')
    record = inspect_stream(wire, allow_incomplete_tail=allow_incomplete_tail)
    require(before.st_size == len(wire) == job['bytes'], 'stat/job/wire byte mismatch')
    require(record['wire_sha256'] == job['stream_sha256'], 'job stream SHA256 mismatch')
    decoded = read_json(paths['fresh/decode.json'])
    results = read_json(paths['result.json'])
    counts = record['bytes']
    expected_decode = dict(base_bytes=counts['native_uf_bytes'],
        container_header_bytes=counts['acse_header_bytes'],
        packet_bytes=counts['e_packet_header_bytes'] + counts['e_payload_bytes'],
        incomplete_tail_bytes=counts['incomplete_tail_bytes'],
        generation_control_bytes=counts['rtvc_shared_header_bytes'], total_bytes=len(wire),
        stream_sha256=record['wire_sha256'], explicit_G_map_bytes=0, config=record['config'])
    require(all(decoded.get(k) == v for k, v in expected_decode.items()), 'decoder accounting differs from wire')
    require(results['decode'] == decoded, 'saved result/decoder mismatch')
    hashes = {name: file_hash(path) for name, path in paths.items()}
    require(hashes['stream.rtvc'] == record['wire_sha256'], 'stream changed during audit')
    require(all(results['artifacts'][name] == hashes[name] for name in ('stream.rtvc', 'fresh/decode.json')),
            'result artifact hash mismatch')
    record.update(sample_id=job['sample_id'], dataset=job['dataset'], method=job['method'],
                  ratio=job['ratio'], max_g=job['max_g'], source_folder=str(folder),
                  stat_bytes=before.st_size, source_artifacts_sha256=hashes)
    return record


def aggregate(records):
    groups = defaultdict(list)
    for row in records:
        groups['All'].append(row)
        groups['dataset:' + row['dataset']].append(row)
        groups['method:' + row['method']].append(row)
    output = {}
    for name, selected in sorted(groups.items()):
        totals = {k: sum(row['bytes'][k] for row in selected) for k in BYTE_KEYS}
        total = sum(row['total_bytes'] for row in selected)
        require(sum(totals.values()) == total, 'aggregate bytes do not reconcile')
        output[name] = dict(points=len(selected), total_bytes=total, byte_totals=totals,
                            byte_means={k: v / len(selected) for k, v in totals.items()},
                            fraction_of_wire={k: v / total for k, v in totals.items()},
                            separate_mask_byte_totals={k: 0 for k in MASK_KEYS},
                            packets=sum(row['packet_count'] for row in selected))
    return output


def resource_snapshot():
    return dict(cpu_only=True, gpu_memory_allocated_by_this_audit_bytes=0,
                disks={str(p): dict(zip(('total', 'used', 'free'), shutil.disk_usage(p)))
                       for p in ('/root', '/root/autodl-tmp', '/root/autodl-fs') if Path(p).exists()})


def audit(root=DEFAULT_ROOT, output=DEFAULT_OUTPUT, *, expected_count=120,
          allow_incomplete_tail=False, verify_only=False, stop_after=0, max_seconds=600.):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(not (output.is_relative_to(root) or root.is_relative_to(output)),
            'audit output must be separate from the historical experiment')
    require(type(expected_count) is int and expected_count > 0, 'invalid expected point count')
    require(type(stop_after) is int and stop_after >= 0, 'invalid stop_after')
    require(type(max_seconds) in (int, float) and math.isfinite(max_seconds) and max_seconds > 0,
            'invalid time limit')
    require(not verify_only or output.is_dir(), 'missing audit output directory')
    output.mkdir(parents=True, exist_ok=True)
    # One writer; the lock lives only in the NEW audit output.
    with (output / '.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _audit_locked(root, output, expected_count, allow_incomplete_tail,
                             verify_only, stop_after, max_seconds)


def _audit_locked(root, output, expected_count, allow_incomplete_tail, verify_only, stop_after, max_seconds):
    began = time.monotonic()
    source_files = ('complete.json', 'protocol.json', 'jobs.json', 'summary.json')
    source_hashes = {name: file_hash(root / name) for name in source_files}
    complete, protocol = read_json(root / 'complete.json'), read_json(root / 'protocol.json')
    jobs = read_json(root / 'jobs.json')['jobs']
    require(complete['complete'] is True and complete['points'] == expected_count == len(jobs),
            'incomplete or unexpected experiment point count')
    require(complete['summary'] == source_hashes['summary.json'] and
            complete['protocol'] == source_hashes['protocol.json'], 'source completion hashes changed')
    require(len({str(Path(j['folder']).resolve()) for j in jobs}) == len(jobs), 'duplicate job folders')
    binding = dict(format=FORMAT, source_root=str(root), expected_count=expected_count,
                   source_hashes=source_hashes, allow_incomplete_tail=allow_incomplete_tail,
                   code_sha256={n: file_hash(REPO / 'demo' / n) for n in CODE_FILES})
    immutable_write(output / 'protocol.json', encoded_json(binding), verify_only=verify_only)
    records, record_hashes = [], {}
    already_complete = (output / 'complete.json').exists()
    for index, job in enumerate(jobs):
        require(time.monotonic() - began < max_seconds, 'audit time limit reached; completed points are resumable')
        snapshot = resource_snapshot()
        require(all(v['used'] / v['total'] < .8 for k, v in snapshot['disks'].items()
                    if k in ('/root/autodl-tmp', '/root/autodl-fs')), 'data disk usage reached 80%; resume later')
        record = inspect_point(root, job, allow_incomplete_tail=allow_incomplete_tail)
        name = f'points/{index:04d}.json'
        data = encoded_json(record)
        immutable_write(output / name, data, verify_only=verify_only)
        records.append(record)
        record_hashes[name] = digest(data)
        progress = dict(complete=False, completed=index + 1, points=len(jobs),
                        elapsed_seconds_this_attempt=time.monotonic() - began, resources=snapshot)
        if not verify_only and not already_complete:
            atomic_write(output / 'progress.json', encoded_json(progress))
        print(f'BYTE_AUDIT {index + 1}/{len(jobs)} bytes={record["total_bytes"]} '
              f'E_payload={record["bytes"]["e_payload_bytes"]} '
              f'elapsed={progress["elapsed_seconds_this_attempt"]:.1f}s', flush=True)
        if stop_after and index + 1 == stop_after and index + 1 < len(jobs):
            return progress
    require(all(file_hash(root / n) == h for n, h in source_hashes.items()), 'source manifest changed during audit')
    for record in records:
        folder = Path(record['source_folder'])
        require(all(file_hash(folder / name) == value
                    for name, value in record['source_artifacts_sha256'].items()),
                'source point artifact changed during audit')
    summary = dict(complete=True, format=FORMAT, points=len(records), binding=binding,
                   source_role=protocol.get('role'), groups=aggregate(records),
                   source_artifacts_unchanged=True, no_decoding=True, no_metric_recomputation=True,
                   mask_removal_savings_bytes=0,
                   scope='Sum/mean across operating points, not one video transfer; historical artifacts untouched.',
                   notes=['E packet headers include addressing, timing, quantization, length and checksum.',
                          'E payload includes internal neural entropy framing; it is not ideal entropy bits.',
                          'No separate E/G/protection map is serialized. This audit saves zero stream bytes.',
                          'Native UF component includes its own native SPS/NAL headers.',
                          'Pre-shared model weights, offline JSON/NPZ and visualizations are not transmitted.'])
    immutable_write(output / 'summary.json', encoded_json(summary), verify_only=verify_only)
    fields = ('sample_id', 'dataset', 'method', 'ratio', 'max_g', 'packet_count', 'total_bytes',
              'bpp', *BYTE_KEYS, *MASK_KEYS, 'wire_sha256')
    csv_data = io.StringIO(newline='')
    writer = csv.DictWriter(csv_data, fieldnames=fields)
    writer.writeheader()
    for row in records:
        flat = dict(row, **row['bytes'], **row['separate_mask_bytes'])
        writer.writerow({k: flat[k] for k in fields})
    immutable_write(output / 'points.csv', csv_data.getvalue().encode(), verify_only=verify_only)
    artifact_hashes = dict(record_hashes, **{n: file_hash(output / n)
                                           for n in ('protocol.json', 'summary.json', 'points.csv')})
    completion = dict(complete=True, format=FORMAT, points=len(records),
                      artifacts_sha256=artifact_hashes, total_bytes=summary['groups']['All']['total_bytes'])
    immutable_write(output / 'complete.json', encoded_json(completion), verify_only=verify_only)
    if not verify_only and not already_complete:
        atomic_write(output / 'progress.json', encoded_json(dict(complete=True, completed=len(records),
                     points=len(records), elapsed_seconds_this_attempt=time.monotonic() - began,
                     resources=resource_snapshot())))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--expected-count', type=int, default=120)
    parser.add_argument('--allow-incomplete-tail', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--stop-after', type=int, default=0)
    parser.add_argument('--max-seconds', type=float, default=600.)
    args = parser.parse_args(argv)
    require(bool(os.environ.get('TMUX')), 'run the audit CLI in tmux')
    result = audit(**vars(args))
    print(json.dumps(dict(complete=result['complete'], points=result['points'],
                          output=str(args.output)), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
