"""Fresh source-guarded receiver for the explicit RVLFUS01 profile."""
import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from demo.chunk_enhancement_codec import configure_torch
from demo.chunk_enhancement_experiment import Run
from demo.routervc_fullview_probe import save, digest
from demo.scalable_codec import atomic_npz
from tools.latent_router_worker import receiver_guard
from routervc.fusion.receive import decode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stream', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--receiver', type=Path, required=True); p.add_argument('--checkpoint', type=Path)
    p.add_argument('--disable-generation', action='store_true')
    a = p.parse_args()
    if not os.environ.get('TMUX'): raise RuntimeError('tmux required')
    configure_torch()
    run = Run(SimpleNamespace(output=a.output, command='decode', max_hours=2)); run.thread.start()
    try:
        receiver_guard(a.output, a.stream)
        base, received, output, report = decode(a.stream.read_bytes(), a.receiver, a.checkpoint,
                                              disable_generation=a.disable_generation, check=run.check)
        atomic_npz(a.output/'pixels.npz', base=base, enhanced=received, reconstruction=output)
        report.update(stream_sha256=digest(a.stream), artifacts={'pixels.npz':digest(a.output/'pixels.npz')})
        save(a.output/'complete.json', report); run.update(phase='complete'); run.log_resources()
    finally: run.stop.set(); run.thread.join(); run.lock.close()
    if torch.distributed.is_initialized(): torch.distributed.destroy_process_group()


if __name__ == '__main__': main()
