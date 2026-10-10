"""Resumable, source-free G halo capture for an already charged receiver stream."""
import argparse
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from demo.routervc_fullview_probe import read, save, digest, immutable, verify_artifacts
from demo.scalable_codec import atomic_npz
from demo.scalable_format import frame_hash
from demo.chunk_enhancement_experiment import Run
from demo.chunk_enhancement_codec import configure_torch
from demo.routervc_receiver_router import load_model
from demo import scalable_cooperation_format as fmt
from routervc.latent import routing, router_data
from routervc.latent.generation import assets, asset_hash
from routervc.fusion.capture import CaptureRGB
from tools.latent_router_worker import receiver_guard


def capture(args, run):
    receiver_guard(args.output, args.stream)
    root = args.output
    immutable(root/'binding.json', dict(stream_sha256=digest(args.stream),
        receiver_sha256=digest(args.receiver), routing=routing.identity(),
        capture_code=digest(Path(__file__)),
        observer_code=digest(Path(__file__).resolve().parents[1]/'routervc/fusion/capture.py')))
    if (root/'complete.json').exists():
        done = read(root/'complete.json'); verify_artifacts(root, done['artifacts']); return
    begin = time.monotonic()
    inner, config, _ = routing.parse(args.stream.read_bytes())
    if (root/'received.json').exists():
        receipt = read(root/'received.json'); verify_artifacts(root, receipt['artifacts'])
        with np.load(root/'received.npz') as z: base, received = z['base'], z['enhanced']
    else:
        base, received, detail = router_data.decode(inner)
        atomic_npz(root/'received.npz', base=base, enhanced=received)
        receipt = dict(detail=detail, base_hash=frame_hash(base), enhanced_hash=frame_hash(received),
                       artifacts={'received.npz':digest(root/'received.npz')})
        save(root/'received.json', receipt)
    model, _ = load_model(args.receiver, expected_sha256=config['receiver_sha256'])
    selected = routing.route(base, received, inner, config, model)
    immutable(root/'selection.json', selected)
    hashes = assets()
    if asset_hash(hashes) != config['assets_sha256']: raise ValueError('G assets changed')
    generator = None
    output = received.copy(); reports = []
    for number, region in enumerate(selected['indices']):
        run.check(); dest = root/f'g{region:02d}'; dest.mkdir(exist_ok=True)
        settings = routing.control(received.shape, hashes, region, config['seed'])
        if (dest/'result.json').exists():
            report = read(dest/'result.json'); verify_artifacts(dest, report['artifacts'])
            with np.load(dest/'raw.npz') as z: raw = z['pixels']
        else:
            if generator is None: generator = CaptureRGB()
            start = time.monotonic()
            with torch.no_grad(): current, raw, report = generator(received, settings)
            report.update(region=region, seconds=time.monotonic()-start,
                          current_hash=frame_hash(current), raw_hash=frame_hash(raw))
            atomic_npz(dest/'raw.npz', pixels=raw)
            report['artifacts'] = {'raw.npz':digest(dest/'raw.npz')}
            save(dest/'result.json', report)
        x0, y0, cw, ch = report['crop']; x, y, w, h = report['core']
        restored = received.copy(); restored[:, y:y+h, x:x+w] = raw[:, y-y0:y-y0+h, x-x0:x-x0+w]
        alpha = fmt.weights(received.shape, settings)
        replay = fmt.combine(received, restored, alpha)
        if frame_hash(replay) != report['current_hash']: raise ValueError('cached raw replay differs')
        output[alpha > 0] = replay[alpha > 0]
        reports.append(digest(dest/'result.json'))
        run.update(phase='raw_halo_capture', completed=number+1, total=len(selected['indices']), region=region)
    atomic_npz(root/'current.npz', pixels=output)
    names = ['binding.json', 'received.json', 'received.npz', 'selection.json', 'current.npz']
    names += [f'g{i:02d}/{n}' for i in selected['indices'] for n in ('result.json', 'raw.npz')]
    save(root/'complete.json', dict(complete=True, stream_sha256=digest(args.stream),
        base_hash=frame_hash(base), enhanced_hash=frame_hash(received), output_hash=frame_hash(output),
        generated=selected['indices'], actual_bytes=args.stream.stat().st_size,
        raw_halo_saved=True, original_feather_exact=True, source_frames_read=False,
        sender_router_loaded=False, no_additional_G_calls=True, no_additional_mask_bytes=True,
        seconds_this_process=time.monotonic()-begin, regions=reports,
        peak_G_allocated_bytes=max((read(root/f'g{i:02d}/result.json')['runtime']['windows'][0]
            ['runtime']['peak_cuda_allocated_bytes'] for i in selected['indices']), default=0),
        artifacts={n:digest(root/n) for n in names}))
    run.update(phase='capture_complete'); run.log_resources()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stream', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--receiver', type=Path, required=True)
    args = p.parse_args()
    if not os.environ.get('TMUX'): raise RuntimeError('tmux required')
    configure_torch()
    run = Run(SimpleNamespace(output=args.output, command='capture', max_hours=12))
    run.thread.start()
    try: capture(args, run)
    finally: run.stop.set(); run.thread.join(); run.lock.close()
    if torch.distributed.is_initialized(): torch.distributed.destroy_process_group()


if __name__ == '__main__': main()
