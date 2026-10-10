"""P2: real smoke/restart/fresh checks -> lazy labels + formal receiver fit."""
import argparse
import gc
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch

from demo.chunk_enhancement_experiment import Run
from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
from demo.chunk_enhancement_codec import configure_torch
from demo.routervc_fullview_probe import read, save, digest, immutable, verify_artifacts
from demo.scalable_codec import atomic_bytes
from demo.routervc_mixed_queue import exact
from routervc.cooperation import data, training, stream
from routervc.latent import routing
from routervc.fusion.blend import geometry


def launch(run, command, options, output):
    output.mkdir(parents=True, exist_ok=True)
    binding = dict(command=command, options=options, code=digest(data.REPO/'tools/cooperative_worker.py'))
    immutable(output/'request.json', binding)
    if (output/'complete.json').exists():
        done = read(output/'complete.json'); verify_artifacts(output, done['artifacts']); return done
    args = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=1',
            '-m', 'tools.cooperative_worker', command, '--output', str(output), *options]
    with (output/'worker.log').open('a') as log:
        child = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while child.poll() is None: run.check(); time.sleep(.5)
            if child.returncode: raise RuntimeError('worker failed: '+str(output/'worker.log'))
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try: child.wait(timeout=30)
                except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL); child.wait()
    return read(output/'complete.json')


def smoke(root, protocol, samples, scale, run):
    folder = root/'smoke'
    if (folder/'complete.json').exists():
        result = read(folder/'complete.json'); verify_artifacts(folder, result['artifacts'])
        if result['protocol'] != digest(root/'protocol.json'): raise ValueError('smoke protocol changed')
        return
    rows = [next(r for r in protocol['rows'] if r['dataset'] == d and r['router_split'] == 'train')
            for d in ('REDS', 'UVG')]
    smoke_protocol = dict(protocol, rows=rows, epochs=2, smoke=True)
    resumed, direct = folder/'resumed', folder/'direct'
    if not (resumed/'resume.pt').exists():
        training.fit(resumed, smoke_protocol, samples.get, scale, check=run.check, progress=run.update, stop_after=2)
    training.fit(resumed, smoke_protocol, samples.get, scale, check=run.check, progress=run.update)
    training.fit(direct, smoke_protocol, samples.get, scale, check=run.check, progress=run.update)
    a = torch.load(resumed/'resume.pt', map_location='cpu', weights_only=True)
    b = torch.load(direct/'resume.pt', map_location='cpu', weights_only=True)
    exact(a, b)
    if a['state']['updates'] != 4: raise ValueError('smoke did not optimize')
    del a, b; samples.release(); gc.collect(); torch.cuda.empty_cache()
    checkpoint = resumed/'best.pt'; receipts = {}
    for row in rows:
        sid = row['sample_id']; dest = folder/sid; dest.mkdir(parents=True, exist_ok=True)
        inner, config, _ = routing.parse(Path(row['stream']).read_bytes())
        blob = stream.wrap(inner, digest(checkpoint), config['assets_sha256'], max_g=2)
        path = dest/'stream.rvlcoop'
        if path.exists() and path.read_bytes() != blob: raise ValueError('smoke stream changed')
        atomic_bytes(path, blob)
        reports = {}
        for name in ('fresh', 'repeat', 'off'):
            run.update(phase='smoke_sourcefree_decode', sample=sid, variant=name)
            options = ['--stream', str(path), '--receiver', str(checkpoint)]
            if name == 'off': options = ['--stream', str(path), '--disable-generation']
            reports[name] = launch(run, 'decode', options, dest/name)
        old = read(Path(row['capture'])/'complete.json')
        for name, report in reports.items():
            if (report['base_hash'] != old['base_hash'] or report['enhanced_hash'] != old['enhanced_hash']
                    or report['actual_bytes'] != len(blob) or report['mask_bytes'] != 0):
                raise ValueError('received bytes/pixels changed')
        if reports['fresh']['output_hash'] != reports['repeat']['output_hash']:
            raise ValueError('fresh receiver is not repeatable')
        if reports['off']['output_hash'] != old['enhanced_hash']: raise ValueError('G-off changed Y')
        with np.load(dest/'fresh/pixels.npz') as z:
            support, _, _ = geometry(z['enhanced'].shape, reports['fresh']['generated'])
            np.testing.assert_array_equal(z['reconstruction'][:, ~support], z['enhanced'][:, ~support])
        receipts[sid] = {n:digest(dest/n/'complete.json') for n in reports}
    artifacts = {n:digest(folder/n) for n in ('resumed/resume.pt', 'direct/resume.pt', 'resumed/best.pt')}
    for sid, result in receipts.items():
        for n, value in result.items(): artifacts[f'{sid}/{n}/complete.json'] = value
    save(folder/'complete.json', dict(complete=True, protocol=digest(root/'protocol.json'),
        optimizer_model_rng_exact=True, real_optimizer_updates=4, sourcefree_decodes=6,
        original_received_exact=True, G_off_model_free=True, masks_sent=False,
        header_bytes=stream.HEADER_BYTES, artifacts=artifacts))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('all', 'smoke', 'verify'), default='all', nargs='?')
    p.add_argument('--output', type=Path, default=data.ROOT)
    args = p.parse_args()
    if not os.environ.get('TMUX'): raise RuntimeError('tmux required')
    configure_torch(); torch.use_deterministic_algorithms(True)
    run = Run(SimpleNamespace(output=args.output, command='cooperation', max_hours=72)); run.thread.start()
    samples = None
    try:
        protocol = data.protocol(); immutable(args.output/'protocol.json', protocol)
        if args.command == 'verify':
            done = read(args.output/'router/complete.json'); verify_artifacts(args.output/'router', done['artifacts'])
            for row in protocol['rows']:
                folder = args.output/'samples'/row['sample_id']; result = read(folder/'complete.json')
                verify_artifacts(folder, result['artifacts'])
            print('COOPERATION_VERIFIED', done['updates'], flush=True); return
        with exclusive_native_evaluation(run):
            samples = data.Samples(args.output, protocol, run,
                                  lambda c, o, d:launch(run, c, o, d))
            calibration = [next(r for r in protocol['rows'] if r['dataset'] == d and r['router_split'] == 'train')
                           for d in ('REDS', 'UVG')]
            scale_file = args.output/'train_scales.json'
            if scale_file.exists():
                record = read(scale_file)
                if record['protocol'] != digest(args.output/'protocol.json'): raise ValueError('scales changed')
                scale = torch.tensor(record['scale'])
            else:
                scale = training.scales(calibration, samples.get)
                save(scale_file, dict(protocol=digest(args.output/'protocol.json'), scale=scale.tolist(),
                    train_only=True, calibration=[r['sample_id'] for r in calibration]))
            smoke(args.output, protocol, samples, scale, run)
            if args.command == 'all':
                training.fit(args.output/'router', protocol, samples.get, scale,
                             check=run.check, progress=run.update)
                names = ['protocol.json', 'train_scales.json', 'smoke/complete.json', 'router/complete.json']
                save(args.output/'complete.json', dict(complete=True,
                    artifacts={n:digest(args.output/n) for n in names},
                    next='paired old/new R_g review with frozen R_s and multiband; no auto R_s training'))
    except BaseException as error:
        save(args.output/'last_failure.json', dict(error=repr(error), progress=run.progress)); raise
    finally:
        if samples is not None: samples.release()
        run.log_resources(); run.stop.set(); run.thread.join(timeout=3); run.lock.close()


if __name__ == '__main__': main()
