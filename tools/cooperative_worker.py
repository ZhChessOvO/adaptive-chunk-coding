"""Isolated source-free G probes and RVLCOOP1 fresh decoding."""
import argparse
import os
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import torch
from demo.chunk_enhancement_codec import configure_torch
from demo.chunk_enhancement_experiment import Run
from demo.routervc_fullview_probe import read, save, digest, immutable, verify_artifacts
from demo.scalable_codec import atomic_npz
from demo.scalable_format import frame_hash
from tools.latent_router_worker import receiver_guard
from routervc.latent import routing, router_data, generation
from routervc.fusion.capture import CaptureRGB


def capture(args, run):
    immutable(args.output/'binding.json', dict(stream=digest(args.stream), regions=args.regions,
        code=digest(Path(__file__)), capture=digest(Path(__file__).resolve().parents[1]/'routervc/fusion/capture.py')))
    inner, config, _ = routing.parse(args.stream.read_bytes())
    base, received, detail = router_data.decode(inner)
    hashes = generation.assets()
    if generation.asset_hash(hashes) != config['assets_sha256']: raise ValueError('G differs')
    generator = None
    artifacts = {'binding.json':digest(args.output/'binding.json')}
    for i in args.regions:
        run.check(); folder = args.output/f'g{i:02d}'; folder.mkdir(exist_ok=True)
        if (folder/'result.json').exists():
            report = read(folder/'result.json'); verify_artifacts(folder, report['artifacts'])
        else:
            if generator is None: generator = CaptureRGB()
            start = time.monotonic()
            with torch.no_grad(): current, raw, report = generator(received,
                routing.control(received.shape, hashes, i, config['seed']))
            atomic_npz(folder/'raw.npz', pixels=raw)
            report.update(region=i, current_hash=frame_hash(current), raw_hash=frame_hash(raw),
                          seconds=time.monotonic()-start, artifacts={'raw.npz':digest(folder/'raw.npz')})
            save(folder/'result.json', report)
        for name in ('raw.npz', 'result.json'): artifacts[f'g{i:02d}/{name}'] = digest(folder/name)
        run.update(phase='capture', region=i)
    save(args.output/'complete.json', dict(complete=True, stream=digest(args.stream),
        regions=args.regions, base_hash=frame_hash(base), enhanced_hash=frame_hash(received),
        source_frames_read=False, artifacts=artifacts))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('capture', 'decode'))
    p.add_argument('--stream', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--regions', type=int, nargs='+')
    p.add_argument('--receiver', type=Path)
    p.add_argument('--disable-generation', action='store_true')
    args = p.parse_args()
    if not os.environ.get('TMUX'): raise RuntimeError('tmux required')
    configure_torch(); receiver_guard(args.output, args.stream)
    run = Run(SimpleNamespace(output=args.output, command=args.command, max_hours=12)); run.thread.start()
    try:
        if (args.output/'complete.json').exists():
            done = read(args.output/'complete.json'); verify_artifacts(args.output, done['artifacts'])
        elif args.command == 'capture': capture(args, run)
        else:
            from routervc.cooperation.receive import decode
            base, received, output, detail = decode(args.stream.read_bytes(), args.receiver,
                                                    args.disable_generation, run.check)
            atomic_npz(args.output/'pixels.npz', base=base, enhanced=received, reconstruction=output)
            detail.update(stream=digest(args.stream), artifacts={'pixels.npz':digest(args.output/'pixels.npz')})
            save(args.output/'complete.json', detail)
        run.log_resources()
    except BaseException as error:
        save(args.output/'last_failure.json', dict(error=repr(error), progress=run.progress)); raise
    finally: run.stop.set(); run.thread.join(timeout=3); run.lock.close()
    if torch.distributed.is_initialized(): torch.distributed.destroy_process_group()


if __name__ == '__main__': main()
