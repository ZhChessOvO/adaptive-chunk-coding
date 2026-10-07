"""Width3 real regional E prefixes with a fixed, untrained center-out order.

This measures a byte/quality path, not optimized Router selection. Region count
is not byte fraction. Spatially coupled native synthesis is left unchanged.
"""
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from demo.scalable_codec import atomic_bytes, atomic_json, atomic_npz, file_hash
from demo.scalable_format import frame_hash
from tools.latent_probe import SAMPLES, read
from tools.latent_stream_probe import DEFAULT as STAGE_B, verify

DEFAULT = STAGE_B.parent / 'regional'
ORDER = (5, 6, 9, 10, 1, 2, 4, 7, 8, 11, 13, 14, 0, 3, 12, 15)
COUNTS = (0, 4, 8, 12, 16)


def regional_change(base, output, region_ids):
    """Measure coupling outside nominal image tiles; not a semantic ROI metric."""
    h, w = base.shape[1:3]
    coverage = np.zeros((h, w), dtype=bool)
    for index in region_ids:
        row, col = divmod(index, 4)
        coverage[row*h//4:(row+1)*h//4, col*w//4:(col+1)*w//4] = True
    difference = np.abs(output[1:].astype(np.float32) - base[1:].astype(np.float32)).mean((0, 3))
    return dict(nominal_coverage=float(coverage.mean()),
        inside_base_change_mae=None if not coverage.any() else float(difference[coverage].mean()),
        outside_base_change_mae=None if coverage.all() else float(difference[~coverage].mean()),
        scope='P8 mean absolute uint8 RGB change from B; intended tile footprint, not guaranteed locality')


def receiver(args):
    import torch
    from routervc.latent.regional_codec import RegionalCodec
    def audit(event, values):
        if event == 'open' and isinstance(values[0], (str, bytes)):
            path = Path(os.fsdecode(values[0])).resolve()
            if not path.is_relative_to(args.output.resolve()) and path.name in (
                'source.npz', 'pair.npz', 'pixels.npz', 'expected.npz', 'symbols.npz', 'protocol.json'):
                raise RuntimeError('source/cache read forbidden in regional receiver')
    sys.addaudithook(audit)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    codec = RegionalCodec()
    base, out, detail = codec.decode_regional(args.stream.read_bytes(),
        allow_incomplete_tail=args.allow_incomplete_tail)
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_npz(args.output / 'pixels.npz', base=base, reconstruction=out)
    detail.update(seconds=time.monotonic()-started, pid=os.getpid(),
        stream_sha256=file_hash(args.stream), base_hash=frame_hash(base), output_hash=frame_hash(out),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    atomic_json(args.output / 'complete.json', dict(detail,
        artifacts={'pixels.npz': file_hash(args.output / 'pixels.npz')}))


def fresh(run, stream, output, allow=False):
    if (output / 'complete.json').exists():
        result = verify(output)
        if result['stream_sha256'] != file_hash(stream):
            raise ValueError('regional worker stream changed')
        return result
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-m', 'tools.latent_regional_probe', 'decode',
               '--stream', str(stream), '--output', str(output)]
    if allow:
        command.append('--allow-incomplete-tail')
    with (output / 'worker.log').open('a') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            began = time.monotonic()
            while child.poll() is None:
                run.check()
                if time.monotonic()-began > 600:
                    raise TimeoutError('regional decoder timeout')
                time.sleep(.3)
            if child.returncode:
                raise RuntimeError(f'regional receiver failed: {output}/worker.log')
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
    return verify(output)


def execute(args):
    import torch
    from routervc.latent.regional_codec import RegionalCodec, regional_hash
    from demo.chunk_enhancement_experiment import Run
    from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
    from demo.stage_c_three_path_roi_probe import LPIPSAlex, evaluate_variant
    if not os.environ.get('TMUX'):
        raise RuntimeError('tmux required')
    if not 1 <= args.limit <= 4:
        raise ValueError('limit must be 1..4')
    samples = SAMPLES[:args.limit]
    inputs = read(STAGE_B / 'protocol.json')['inputs']
    protocol = dict(profile='RVLR1-stage-D-first', profile_hash=regional_hash(),
        driver_sha256=file_hash(Path(__file__)), q_star=48, bin_width=3, I_qp=32,
        grid=4, order=list(ORDER), counts=list(COUNTS),
        inputs={s: inputs[s] for s in samples},
        parent_streams={s: file_hash(STAGE_B / s / 'w3_BE.rvl') for s in samples},
        selection='fixed center-out order, no content ranking; not Router gain',
        generation=False, router=False, training=False,
        encoding_scope='repartition the SAME stage B encoding; not fresh analysis-encoder timing')
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'protocol.json'
    if path.exists():
        if read(path) != protocol:
            raise ValueError('changed regional protocol; use another output')
    elif args.command == 'verify':
        raise RuntimeError('no completed regional experiment')
    else:
        atomic_json(path, protocol)
    for item in protocol['inputs'].values():
        if file_hash(Path(item['source_path'])) != item['source_sha256']:
            raise ValueError('regional source changed')
    if args.command == 'verify' or (args.output / 'complete.json').exists():
        verify(args.output)
        for sid in samples:
            result = verify(args.output / sid)
            for name in result['points']:
                verify(args.output / sid / name)
        print('Verified regional streams without inference', flush=True)
        return
    run = Run(args)
    run.thread.start()
    results = []
    try:
        with exclusive_native_evaluation(run):
            metric = LPIPSAlex(True)
            for sid in samples:
                run.check()
                folder = args.output / sid
                folder.mkdir(parents=True, exist_ok=True)
                if (folder / 'complete.json').exists():
                    results.append(verify(folder))
                    continue
                run.update(sample=sid, completed=len(results), phase='repacketize_original_encoding')
                verify(STAGE_B / sid)
                with np.load(inputs[sid]['source_path'], allow_pickle=False) as f:
                    source = f['source'][:9].copy()
                if not (folder / 'encoded.json').exists():
                    began = time.monotonic()
                    codec = RegionalCodec()
                    base, packets = codec.repacketize((STAGE_B / sid / 'w3_BE.rvl').read_bytes())
                    streams = {f'E{count}': base+b''.join(packets[i] for i in ORDER[:count])
                               for count in COUNTS}
                    streams['single5'] = base+packets[5]
                    if sid == samples[0]:
                        streams.update(reverse=base+b''.join(packets[i] for i in reversed(ORDER)),
                            repeat=streams['E16']+packets[5],
                            reverse8=base+b''.join(packets[i] for i in reversed(ORDER[:8])),
                            incomplete=streams['E8']+packets[ORDER[8]][:-1])
                    for name, data in streams.items():
                        atomic_bytes(folder / f'{name}.rvlr', data)
                    for low, high in zip(COUNTS, COUNTS[1:]):
                        if not streams[f'E{high}'].startswith(streams[f'E{low}']):
                            raise RuntimeError('regional prefix rewrote earlier bytes')
                    encoded = dict(repacketize_seconds=time.monotonic()-began,
                        streams=list(streams), packets_bytes=[len(p) for p in packets],
                        artifacts={f'{n}.rvlr': file_hash(folder / f'{n}.rvlr') for n in streams})
                    atomic_json(folder / 'encoded.json', encoded)
                    del codec
                    gc.collect()
                    torch.cuda.empty_cache()
                encoded = read(folder / 'encoded.json')
                for name, sha in encoded['artifacts'].items():
                    if file_hash(folder / name) != sha:
                        raise ValueError('regional checkpoint changed')
                base_hash = read(STAGE_B / sid / 'w3_B/complete.json')['output_hash']
                endpoint_hash = read(STAGE_B / sid / 'w3_BE/complete.json')['output_hash']
                points = {}
                native_bytes = (STAGE_B / sid / 'native.bin').stat().st_size
                full_bytes = (folder / 'E16.rvlr').stat().st_size
                for name in encoded['streams']:
                    run.update(phase='regional_fresh_decode', state=name)
                    point = fresh(run, folder / f'{name}.rvlr', folder / name, name == 'incomplete')
                    if point['base_hash'] != base_hash:
                        raise RuntimeError('regional E changed base reconstruction')
                    if point['actual_bytes'] != (folder / f'{name}.rvlr').stat().st_size:
                        raise RuntimeError('regional byte count wrong')
                    if point['base_bytes']+point['E_wire_bytes']+point['ignored_tail_bytes'] != point['actual_bytes']:
                        raise RuntimeError('regional byte decomposition wrong')
                    if name in ('E16', 'reverse', 'repeat') and point['output_hash'] != endpoint_hash:
                        raise RuntimeError('full regional E failed native endpoint')
                    with np.load(folder / name / 'pixels.npz', allow_pickle=False) as f:
                        base = f['base'].copy()
                        out = f['reconstruction'].copy()
                    points[name] = dict(point, quality_all9=evaluate_variant(source, out, metric),
                        quality_P8=evaluate_variant(source[1:], out[1:], metric),
                        spatial_change=regional_change(base, out, point['received_regions']),
                        saving_vs_native_full=1-point['actual_bytes']/native_bytes,
                        saving_vs_regional_full=1-point['actual_bytes']/full_bytes)
                    print(f'{sid} {name}: {point["actual_bytes"]} bytes; '
                          f'save vs native full {points[name]["saving_vs_native_full"]:.2%}', flush=True)
                if sid == samples[0]:
                    if any(points[n]['output_hash'] != points['E8']['output_hash']
                           for n in ('reverse8', 'incomplete')):
                        raise RuntimeError('partial reorder/fallback display changed')
                names = list(encoded['artifacts'])+['encoded.json']+[f'{n}/complete.json' for n in points]
                result = dict(complete=True, sample=sid, points=points, native_bytes=native_bytes,
                    packet_bytes=encoded['packets_bytes'],
                    artifacts={n: file_hash(folder / n) for n in names})
                atomic_json(folder / 'complete.json', result)
                results.append(result)
                run.update(completed=len(results), total=len(samples))
        atomic_json(args.output / 'summary.json', dict(results=results))
        atomic_json(args.output / 'complete.json', dict(complete=True, samples=len(results),
            fresh_decodes=sum(len(r['points']) for r in results),
            seconds=time.monotonic()-run.started,
            artifacts={'summary.json': file_hash(args.output / 'summary.json')}))
    finally:
        run.stop.set()
        run.thread.join(timeout=2)
        run.log_resources()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'verify', 'decode'))
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--stream', type=Path)
    parser.add_argument('--limit', type=int, default=4)
    parser.add_argument('--max-hours', type=float, default=4)
    parser.add_argument('--allow-incomplete-tail', action='store_true')
    args = parser.parse_args()
    if args.command == 'decode':
        receiver(args)
    else:
        execute(args)


if __name__ == '__main__':
    main()
