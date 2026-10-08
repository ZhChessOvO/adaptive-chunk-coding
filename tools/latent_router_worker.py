"""Workers for new-latent R_g labels, fitting, and source-free fresh decode."""
import argparse
import gc
import os
from pathlib import Path
import signal
import sys
import time
from unittest.mock import patch

import numpy as np
import torch

from demo.routervc_fullview_probe import read, digest, save, verify_artifacts
from demo.scalable_codec import atomic_npz
from demo.scalable_format import frame_hash
from demo.chunk_enhancement_codec import configure_torch
from demo.chunk_enhancement_experiment import Run
from routervc.latent import routing, router_data as data


def receiver_guard(output, stream):
    output, stream = output.resolve(), stream.resolve()
    def guard(event, values):
        if event != 'open' or not isinstance(values[0], (str, bytes)):
            return
        path = Path(os.fsdecode(values[0])).resolve()
        if path == stream or path.is_relative_to(output):
            return
        text = str(path)
        if (path.suffix.lower() in ('.npz', '.png', '.jpg', '.jpeg')
                or path.name in ('bank.rvlp', 'protocol.json')
                or '/DCVC/data/' in text or '/mixedview_data/' in text
                or '/assets/evaluation/' in text or '/routervc_sender_' in text):
            raise RuntimeError('receiver attempted source/sender/unreceived-bank access: '+text)
    sys.addaudithook(guard)


def receive(args):
    from demo.routervc_receiver_router import load_model
    from demo.four_state_receive import PersistentRGB
    from routervc.latent.generation import assets, asset_hash
    receiver_guard(args.output, args.stream)
    started = time.monotonic()
    inner, config, parsed = routing.parse(args.stream.read_bytes())
    peaks, reserved = [], []
    original = torch.cuda.reset_peak_memory_stats
    def reset(device=None):
        peaks.append(torch.cuda.max_memory_allocated(device))
        reserved.append(torch.cuda.max_memory_reserved(device))
        original(device)
    with patch.object(torch.cuda, 'reset_peak_memory_stats', reset):
        torch.cuda.reset_peak_memory_stats()
        base, received, detail = data.decode(inner)
        if args.disable_generation:
            output, generated, reports = received, [], []
            selected = None
        else:
            model, _ = load_model(args.receiver, expected_sha256=config['receiver_sha256'])
            selected = routing.route(base, received, inner, config, model)
            generated = selected['indices']
            # No G model is needed if no positive predicted gain is selected.
            if generated:
                hashes = assets()
                if asset_hash(hashes) != config['assets_sha256']:
                    raise ValueError('frozen G identity differs')
                generator = PersistentRGB()
                with torch.no_grad():
                    output, reports = routing.render(received, generated, hashes, generator, seed=config['seed'])
            else:
                output, reports = received, []
    peaks.append(torch.cuda.max_memory_allocated()); reserved.append(torch.cuda.max_memory_reserved())
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_npz(args.output/'pixels.npz', base=base, enhanced=received, reconstruction=output)
    save(args.output/'complete.json', dict(complete=True, stream_sha256=digest(args.stream),
        base_hash=frame_hash(base), enhanced_hash=frame_hash(received), output_hash=frame_hash(output),
        detail=detail, selected=selected, generated=generated, generation_runtime=reports,
        generation_disabled=args.disable_generation, source_frames_read=False, sender_router_loaded=False,
        additional_mask_bytes=0, header_bytes=routing.HEADER_BYTES,
        actual_bytes=args.stream.stat().st_size, bpp=args.stream.stat().st_size*8/np.prod(base.shape[:3]),
        peak_cuda_allocated_bytes=max(peaks), peak_cuda_reserved_bytes=max(reserved),
        seconds=time.monotonic()-started, pid=os.getpid(),
        artifacts={'pixels.npz':digest(args.output/'pixels.npz')}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('labels', 'fit', 'decode'))
    p.add_argument('--root', type=Path)
    p.add_argument('--cache', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stream', type=Path)
    p.add_argument('--receiver', type=Path)
    p.add_argument('--disable-generation', action='store_true')
    p.add_argument('--stop-after', type=int, default=0)
    p.add_argument('--max-hours', type=float, default=48.)
    args = p.parse_args()
    if not os.environ.get('TMUX'):
        raise RuntimeError('tmux required for GPU workers')
    configure_torch()
    if args.command == 'decode':
        receive(args)
    else:
        protocol = read(args.root/'protocol.json')
        for name, expected in protocol['code'].items():
            if digest(data.REPO/name) != expected:
                raise ValueError('pinned training source changed: '+name)
        # Workers have their own SIGTERM checkpoint boundary and heartbeat.
        run = Run(args); run.thread.start()
        samples = data.Samples(args.root, args.cache, protocol, run.check, run.update)
        try:
            if args.command == 'labels':
                done = {}
                for i, row in enumerate(protocol['rows']):
                    run.check()
                    run.update(window=i+1, windows=len(protocol['rows']))
                    samples.prepare(row)
                    done[row['sample_id']] = digest(args.root/'samples'/row['sample_id']/'complete.json')
                save(args.root/'labels.complete.json', dict(complete=True, protocol=digest(args.root/'protocol.json'),
                    samples=done, measured_G_cells=len(done)*96, old_labels_reused=False))
                run.update(phase='labels_complete', samples=len(done))
            else:
                from routervc.latent.receiver_fit import fit
                labels = read(args.root/'labels.complete.json')
                if not labels['complete'] or labels['protocol'] != digest(args.root/'protocol.json'):
                    raise ValueError('all new labels must be complete before fitting TRAIN scales')
                for row in protocol['rows']:
                    if digest(args.root/'samples'/row['sample_id']/'complete.json') != labels['samples'][row['sample_id']]:
                        raise ValueError('label manifest changed')
                fit(args.output, protocol, samples.get, check=run.check, progress=run.update,
                    stop_after=args.stop_after)
        except BaseException as error:
            save(args.output/'last_failure.json', dict(error=repr(error), progress=run.progress))
            raise
        finally:
            samples.release()
            run.log_resources(); run.stop.set(); run.thread.join(timeout=3); run.lock.close()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
