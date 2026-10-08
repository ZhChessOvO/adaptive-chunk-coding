"""Fresh frozen G on compact E: same regions, noise, and weights across prefixes."""
import argparse
import gc
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np
from demo.scalable_codec import atomic_bytes, atomic_json, atomic_npz, file_hash
from demo.scalable_format import frame_hash
from tools.latent_probe import read, SAMPLES
from tools.latent_stream_probe import verify
from tools.latent_packet_probe import ROOT, receiver_guard, source_pixels
from routervc.latent import generation as g

COUNTS = (0, 4, 8, 16)


def receive(args):
    import torch
    from routervc.latent.packet_codec import PacketCodec
    from demo.chunk_enhancement_codec import configure_torch
    from demo import scalable_cooperation_format as blend
    from demo.internal_condition_decode import restore
    receiver_guard(args.output)
    configure_torch()
    begun = time.monotonic()
    peaks, reserved = [], []
    original_reset = torch.cuda.reset_peak_memory_stats
    def measured_reset(device=None):
        peaks.append(torch.cuda.max_memory_allocated(device))
        reserved.append(torch.cuda.max_memory_reserved(device))
        original_reset(device)
    with patch.object(torch.cuda, 'reset_peak_memory_stats', measured_reset):
        torch.cuda.reset_peak_memory_stats()
        data = args.stream.read_bytes()
        inner, identity, parsed = g.parse(data)
        codec = PacketCodec()
        base, enhanced, detail = codec.decode_packets(inner)
        del codec; gc.collect(); torch.cuda.empty_cache()
        codec_seconds = time.monotonic()-begun
        hashes, runtime = None, None
        if args.disable_generation:
            out = enhanced.copy()
        else:
            hashes = g.assets(args.adapter)
            if g.asset_hash(hashes) != identity:
                raise ValueError('G shared asset identity mismatch')
            control = g.control(enhanced.shape, hashes)
            alpha = blend.weights(enhanced.shape, control)
            generated, runtime = restore(enhanced, control, args.adapter, [])
            out = blend.combine(enhanced, generated, alpha)
            np.testing.assert_array_equal(out[alpha == 0], enhanced[alpha == 0])
    peaks.append(torch.cuda.max_memory_allocated()); reserved.append(torch.cuda.max_memory_reserved())
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_npz(args.output/'pixels.npz', base=base, enhanced=enhanced, reconstruction=out)
    detail.update(generation=not args.disable_generation, router=False, sender_router_loaded=False,
        generation_assets_validated=not args.disable_generation, generation_runtime=runtime,
        generation_asset_identity=identity, generation_policy='fixed shared four center ROIs; no trained Router',
        generation_header_bytes=g.HEADER.size, actual_bytes=len(data),
        bpp=len(data)*8/np.prod(base.shape[:3]), base_hash=frame_hash(base),
        enhanced_hash=frame_hash(enhanced), output_hash=frame_hash(out),
        stream_sha256=file_hash(args.stream), pid=os.getpid(), codec_seconds=codec_seconds,
        seconds=time.monotonic()-begun, peak_cuda_allocated_bytes=max(peaks),
        peak_cuda_reserved_bytes=max(reserved), memory_counter_segments=len(peaks),
        outside_generate_exact=True, source_frames_read=False,
        explicit_G_mask_bytes=0, protection_mask_bytes=0,
        generation_all_calls_peak_cuda_bytes=max((v['runtime']['peak_cuda_allocated_bytes']
            for v in (runtime or {}).get('windows', [])), default=0))
    atomic_json(args.output/'complete.json', dict(detail, artifacts={'pixels.npz': file_hash(args.output/'pixels.npz')}))
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def fresh(run, stream, output, disabled=False):
    if (output/'complete.json').exists():
        done = verify(output)
        if done['stream_sha256'] != file_hash(stream) or done['generation'] == disabled:
            raise ValueError('changed G receiver request')
        return done
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=1',
               '-m', 'tools.latent_generate_probe', 'decode', '--stream', str(stream), '--output', str(output)]
    if disabled:
        command += ['--disable-generation', '--adapter', '/nonexistent/G-off-must-not-read.pt']
    with (output/'worker.log').open('a') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            start = time.monotonic()
            while child.poll() is None:
                run.check()
                if time.monotonic()-start > 900:
                    raise TimeoutError('G transfer timeout')
                time.sleep(.5)
            if child.returncode:
                raise RuntimeError(f'G transfer failed: {output}/worker.log')
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()
    return verify(output)


def quality_regions(source, output, metric):
    from demo.routervc_policy import grid_rois
    from demo.stage_c_three_path_roi_probe import evaluate_variant
    rois = grid_rois(*source.shape[1:3])
    local = []
    for index in g.REGIONS:
        x, y, w, h = rois[index]
        local.append(evaluate_variant(source[:, y:y+h, x:x+w], output[:, y:y+h, x:x+w], metric))
    return dict(whole=evaluate_variant(source, output, metric),
                G_regions_mean={k:float(np.mean([v[k] for v in local])) for k in local[0]})


def execute(args):
    from demo.chunk_enhancement_experiment import Run
    from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
    from demo.stage_c_three_path_roi_probe import LPIPSAlex
    from demo.online_eg_eval_core import noise_pair
    if not os.environ.get('TMUX'):
        raise RuntimeError('tmux required')
    if not 1 <= args.limit <= 4:
        raise ValueError('invalid limit')
    continuous = ROOT/'continuous'
    if not (continuous/'complete.json').exists():
        raise RuntimeError('finish continuous regional checks before reconnecting G')
    verify(continuous)
    cases = read(continuous/'protocol.json')['cases'][:args.limit]
    hashes = g.assets()
    protocol = dict(profile=g.profile(), driver=file_hash(Path(__file__)), assets=hashes, cases=cases,
        counts=list(COUNTS), fixed_G_regions=list(g.REGIONS), seed=20261008,
        condition='received mixed RGB, RGB-first mean-BF16; old joint adapter unchanged',
        masks_sent=False, training=False, router=False,
        scope='untrained transfer diagnostic; same noise/ROI across new E budgets, not full benchmark',
        parent_sha256=file_hash(continuous/'complete.json'))
    args.output = args.output or ROOT/'generation'
    args.output.mkdir(parents=True, exist_ok=True)
    pp = args.output/'protocol.json'
    if pp.exists():
        if read(pp) != protocol:
            raise ValueError('changed G probe binding; choose new directory')
    elif args.command == 'verify':
        raise ValueError('G probe not complete')
    else:
        atomic_json(pp, protocol)
    if args.command == 'verify' or (args.output/'complete.json').exists():
        verify(args.output)
        for case in cases:
            folder = args.output/case['sample']
            for name in verify(folder)['points']:
                verify(folder/name)
        print('Verified frozen G probe without inference', flush=True)
        return
    run = Run(args); run.thread.start(); results = []
    try:
        with exclusive_native_evaluation(run):
            metric = LPIPSAlex(True)
            for case in cases:
                folder = args.output/case['sample']; folder.mkdir(parents=True, exist_ok=True)
                if (folder/'complete.json').exists():
                    results.append(verify(folder)); continue
                source = source_pixels(case['source'])
                parent = continuous/case['name']; old = verify(parent)
                points, files = {}, []
                for count in COUNTS:
                    stem = f'E{count}'
                    path = folder/f'{stem}.rvlg'
                    wire = g.wrap((parent/f'{stem}.rvlp').read_bytes(), g.asset_hash(hashes))
                    if path.exists() and path.read_bytes() != wire:
                        raise ValueError('G envelope changed')
                    if not path.exists():
                        atomic_bytes(path, wire)
                    run.update(sample=case['sample'], phase='frozen_G_fresh_receive', prefix=stem, completed=len(results))
                    point = fresh(run, path, folder/stem)
                    baseline = old['points'][stem]
                    if point['base_hash'] != baseline['base_hash'] or point['enhanced_hash'] != baseline['output_hash']:
                        raise RuntimeError('G receiver changed its B/E input')
                    if point['base_reference_hashes'] != baseline['base_reference_hashes']:
                        raise RuntimeError('G/E altered temporal reference')
                    if point['actual_bytes'] != path.stat().st_size or point['actual_bytes'] != baseline['actual_bytes']+g.HEADER.size:
                        raise RuntimeError('G header not charged exactly')
                    with np.load(folder/stem/'pixels.npz', allow_pickle=False) as f:
                        scores = quality_regions(source, f['reconstruction'], metric)
                        before = quality_regions(source, f['enhanced'], metric)
                    points[stem] = dict(point, quality=scores, G_off_quality=before,
                                        bare_E_bytes=baseline['actual_bytes'])
                    if count:
                        noise_pair(points['E0'], points[stem])
                    files += [path.name, f'{stem}/complete.json']
                    print(json.dumps(dict(sample=case['sample'], prefix=stem,
                        LPIPS_before=before['whole']['lpips_alex'], LPIPS_after=scores['whole']['lpips_alex'],
                        local_before=before['G_regions_mean']['lpips_alex'], local_after=scores['G_regions_mean']['lpips_alex'])), flush=True)
                # Idempotent prefix construction and independent repeat/G-off checks.
                for lo, hi in zip(COUNTS, COUNTS[1:]):
                    if not (folder/f'E{hi}.rvlg').read_bytes().startswith((folder/f'E{lo}.rvlg').read_bytes()):
                        raise RuntimeError('G envelope broke E append-only prefix')
                if case['sample'] == SAMPLES[0]:
                    for name, disabled in (('repeat_E8', False), ('G_off_E8', True)):
                        point = fresh(run, folder/'E8.rvlg', folder/name, disabled)
                        target = points['E8']['enhanced_hash'] if disabled else points['E8']['output_hash']
                        if point['output_hash'] != target:
                            raise RuntimeError('repeat/G-off output mismatch')
                        if not disabled:
                            noise_pair(point, points['E8'], same_condition=True)
                        points[name] = point; files.append(f'{name}/complete.json')
                result = dict(sample=case['sample'], points=points,
                              artifacts={n:file_hash(folder/n) for n in files})
                atomic_json(folder/'complete.json', result); results.append(result)
                run.update(completed=len(results), total=len(cases))
        atomic_json(args.output/'summary.json', dict(results=results))
        atomic_json(args.output/'complete.json', dict(complete=True, cases=len(results),
            fresh_decodes=sum(len(r['points']) for r in results), seconds=time.monotonic()-run.started,
            artifacts={'protocol.json':file_hash(pp), 'summary.json':file_hash(args.output/'summary.json')}))
    finally:
        run.stop.set(); run.thread.join(timeout=2); run.log_resources()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('run', 'verify', 'decode'))
    p.add_argument('--output', type=Path)
    p.add_argument('--stream', type=Path)
    p.add_argument('--adapter', type=Path, default=g.ADAPTER)
    p.add_argument('--disable-generation', action='store_true')
    p.add_argument('--limit', type=int, default=4)
    p.add_argument('--max-hours', type=float, default=4)
    args = p.parse_args()
    receive(args) if args.command == 'decode' else execute(args)


if __name__ == '__main__':
    main()
