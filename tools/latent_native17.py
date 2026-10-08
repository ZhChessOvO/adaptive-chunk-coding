"""Matched native-UF QP anchors for the frozen-latent diagnostic (not training).

Same four saved inputs, I32 + two P8 chunks; scalar P QPs are declared before running.
Real native streams and fresh receivers, no source input to the receiving worker.
The QP48 stream is reused from the continuous regional check; previous pinned code remains unchanged.
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
from tools.latent_probe import SAMPLES, read, REPO
from tools.latent_stream_probe import DEFAULT as STAGE_B, verify
from routervc.latent.codec import MODEL_I, MODEL_P

CONTINUOUS = Path('/root/autodl-fs/DCVC/runs/routervc_latent_20261008/continuous')
DEFAULT = CONTINUOUS.parent / 'native17'
QPS = (0, 8, 16, 24, 32, 40, 48)


def receiver(args):
    import torch
    from demo.scalable_codec import BaseCodec
    from demo.chunk_enhancement_codec import configure_torch
    from demo.stage_c_three_path_roi_probe import dcvc_stream_breakdown

    def audit(event, values):
        if event == 'open' and isinstance(values[0], (str, bytes)):
            path = Path(os.fsdecode(values[0])).resolve()
            if not path.is_relative_to(args.output.resolve()) and path.name in (
                    'source.npz', 'expected.npz', 'pixels.npz', 'protocol.json', 'pair.npz'):
                raise RuntimeError('source/cache access forbidden in native receiver')
    sys.addaudithook(audit)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    codec = BaseCodec(MODEL_I, MODEL_P)
    configure_torch()
    setup = time.monotonic() - started
    data = args.stream.read_bytes()
    begin = time.monotonic()
    out = codec.decode(data, 17)
    seconds = time.monotonic() - begin
    breakdown = dcvc_stream_breakdown(data, 17)
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_npz(args.output / 'pixels.npz', reconstruction=out)
    atomic_json(args.output / 'complete.json', dict(
        complete=True, source_frames_read=False, stream_sha256=file_hash(args.stream),
        output_hash=frame_hash(out), frame_count=17, actual_bytes=len(data),
        bpp=len(data) * 8 / np.prod(out.shape[:3]), setup_seconds=setup,
        decode_seconds=seconds, total_seconds=time.monotonic() - started,
        timing_scope='first native decode in fresh process, not warmed throughput',
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), breakdown=breakdown,
        artifacts={'pixels.npz': file_hash(args.output / 'pixels.npz')}))


def fresh(run, stream, out):
    if (out / 'complete.json').exists():
        result = verify(out)
        if result['stream_sha256'] != file_hash(stream):
            raise ValueError('native worker input changed')
        return result
    out.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-m', 'tools.latent_native17', 'decode',
               '--stream', str(stream), '--output', str(out)]
    with (out / 'worker.log').open('a') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            started = time.monotonic()
            while process.poll() is None:
                run.check()
                if time.monotonic() - started > 600:
                    raise TimeoutError('native receiver timeout')
                time.sleep(.3)
            if process.returncode:
                raise RuntimeError(f'native receiver failed: {out}/worker.log')
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    return verify(out)


def execute(args):
    import torch
    import inference_extensions_cuda as native
    from demo.scalable_codec import BaseCodec
    from demo.chunk_enhancement_codec import configure_torch
    from demo.chunk_enhancement_experiment import Run
    from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
    from demo.stage_c_three_path_roi_probe import encode_dcvc_stream, LPIPSAlex, evaluate_variant

    if not os.environ.get('TMUX'):
        raise RuntimeError('tmux required')
    if not 1 <= args.limit <= 4:
        raise ValueError('limit must be 1..4')
    base = read(STAGE_B / 'protocol.json')
    samples = SAMPLES[:args.limit]
    legacy = ('demo/scalable_codec.py', 'demo/stage_c_three_path_roi_probe.py',
              'demo/chunk_enhancement_codec.py', 'src/models/video_model_ht.py')
    protocol = dict(
        schema='native-anchors-I32-two-P8-v1', I_qp=32, P_qps=list(QPS), frame_count=17,
        inputs={s: base['inputs'][s] for s in samples}, source_stage=str(STAGE_B),
        stage_B_complete_sha256=file_hash(STAGE_B / 'complete.json'),
        continuous_complete_sha256=file_hash(CONTINUOUS / 'complete.json'),
        driver_sha256=file_hash(Path(__file__)),
        model_i_sha256=file_hash(MODEL_I), model_p_sha256=file_hash(MODEL_P),
        native_library_sha256=file_hash(Path(native.__file__)),
        legacy_sources={p: file_hash(REPO / p) for p in legacy},
        scope='four reused diagnostics; fixed bootstrap; not standard full-video RD',
        generation=False, training=False, router=False)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'protocol.json'
    if path.exists():
        if read(path) != protocol:
            raise ValueError('changed anchor protocol; use another output')
    elif args.command == 'verify':
        raise RuntimeError('no completed anchor experiment')
    else:
        atomic_json(path, protocol)
    for item in protocol['inputs'].values():
        if file_hash(Path(item['source_path'])) != item['source_sha256']:
            raise ValueError('anchor source changed')
    if args.command == 'verify' or (args.output / 'complete.json').exists():
        verify(args.output)
        for sid in samples:
            for qp in QPS:
                verify(args.output / sid / f'q{qp}')
                verify(args.output / sid / f'q{qp}' / 'fresh')
        print('Verified native anchors without inference', flush=True)
        return
    run = Run(args)
    run.thread.start()
    rows = []
    try:
        with exclusive_native_evaluation(run):
            configure_torch()
            metric = LPIPSAlex(True)
            for sid in samples:
                with np.load(protocol['inputs'][sid]['source_path'], allow_pickle=False) as f:
                    source = f['source'][:17].copy()
                for qp in QPS:
                    run.check()
                    folder = args.output / sid / f'q{qp}'
                    if (folder / 'complete.json').exists():
                        rows.append(verify(folder))
                        continue
                    folder.mkdir(parents=True, exist_ok=True)
                    run.update(sample=sid, qp=qp, completed=len(rows), phase='native_anchor')
                    if (folder / 'encoded.json').exists():
                        encoded = read(folder / 'encoded.json')
                        for name, sha in encoded['artifacts'].items():
                            if file_hash(folder / name) != sha:
                                raise ValueError('native encode checkpoint changed')
                    else:
                        if qp == 48:
                            parent = CONTINUOUS / f'{sid}_n17'
                            verify(parent)
                            data = (parent / 'native_own.bin').read_bytes()
                            timing = dict(reused_continuous=True, encode_seconds=None)
                        else:
                            codec = BaseCodec(MODEL_I, MODEL_P)
                            configure_torch()
                            torch.cuda.set_stream(codec.stream)
                            with torch.inference_mode():
                                data, details = encode_dcvc_stream(list(source), 32, qp,
                                    codec.i_net, codec.p_net, codec.device, 32)
                            timing = dict(reused_continuous=False, encode_seconds=details['seconds'])
                            del codec
                            gc.collect()
                            torch.cuda.empty_cache()
                        atomic_bytes(folder / 'native.bin', data)
                        atomic_json(folder / 'encoded.json', dict(**timing,
                            artifacts={'native.bin': file_hash(folder / 'native.bin')}))
                    received = fresh(run, folder / 'native.bin', folder / 'fresh')
                    with np.load(folder / 'fresh/pixels.npz', allow_pickle=False) as f:
                        out = f['reconstruction'].copy()
                    if qp == 48:
                        with np.load(CONTINUOUS / f'{sid}_n17' / 'native_own.npz', allow_pickle=False) as f:
                            expected = frame_hash(f['reconstruction'])
                        if received['output_hash'] != expected:
                            raise RuntimeError('fresh native anchor changed continuous own-reference endpoint')
                    result = dict(received, sample=sid, qp=qp,
                        quality_all17=evaluate_variant(source, out, metric),
                        quality_P16=evaluate_variant(source[1:], out[1:], metric))
                    result['artifacts'] = {n: file_hash(folder / n) for n in
                        ('encoded.json', 'native.bin', 'fresh/complete.json')}
                    atomic_json(folder / 'complete.json', result)
                    rows.append(result)
                    print(f'{sid} P{qp}: {result["bpp"]:.6f} bpp, '
                          f'{result["quality_all17"]["psnr_db"]:.3f} dB, '
                          f'LPIPS {result["quality_all17"]["lpips_alex"]:.5f}', flush=True)
            atomic_json(args.output / 'summary.json', dict(rows=rows))
            atomic_json(args.output / 'complete.json', dict(complete=True, points=len(rows),
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
    args = parser.parse_args()
    if args.command == 'decode':
        receiver(args)
    else:
        execute(args)


if __name__ == '__main__':
    main()

