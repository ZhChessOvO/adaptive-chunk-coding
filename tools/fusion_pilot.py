"""Paired, frozen-policy P1 controls using the 13 existing diagnostic streams."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

from demo.routervc_fullview_probe import read, save, digest, immutable, verify_artifacts
from demo.chunk_enhancement_experiment import Run
from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
from demo.scalable_codec import atomic_npz
from demo.scalable_experiment import quality
from demo.stage_c_three_path_roi_probe import LPIPSAlex
from routervc.latent.sender_data import RECEIVER
from routervc.fusion.blend import fuse
from routervc.fusion.boundaries import CATEGORIES, edges, measure, aggregate
from tools.latent_boundary_report import ROOT, EVALUATION


def worker(run, stream, output):
    if (output/'complete.json').exists():
        done = read(output/'complete.json'); verify_artifacts(output, done['artifacts'])
        if done['stream_sha256'] != digest(stream): raise ValueError('captured stream changed')
        return
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=1',
               '-m', 'tools.fusion_capture_worker', '--stream', str(stream),
               '--output', str(output), '--receiver', str(RECEIVER)]
    with (output/'worker.log').open('a') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while child.poll() is None: run.check(); time.sleep(.5)
            if child.returncode: raise RuntimeError(f'capture failed; see {output}/worker.log')
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try: child.wait(timeout=30)
                except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL); child.wait()


def load_capture(folder):
    done = read(folder/'complete.json'); verify_artifacts(folder, done['artifacts'])
    with np.load(folder/'received.npz') as z: base, received = z['base'], z['enhanced']
    with np.load(folder/'current.npz') as z: current = z['pixels']
    patches = []
    for region in done['generated']:
        p = folder/f'g{region:02d}'; report = read(p/'result.json')
        with np.load(p/'raw.npz') as z: pixels = z['pixels']
        patches.append(dict(region=region, pixels=pixels, crop=report['crop'], core=report['core']))
    return base, received, current, patches, done


def compare(run, row, point, folder, metric):
    dest = folder/'controls'; dest.mkdir(exist_ok=True)
    if (dest/'complete.json').exists():
        result = read(dest/'complete.json'); verify_artifacts(dest, result['artifacts']); return result
    base, received, current, patches, done = load_capture(folder)
    old = read(point/'receive/complete.json')
    for key in ('base_hash', 'enhanced_hash', 'output_hash', 'generated', 'actual_bytes'):
        if old[key] != done[key]: raise ValueError('observed G changed old receiver: '+key)
    if digest(Path(row['source_path'])) != row['source_sha256']: raise ValueError('source changed')
    with np.load(row['source_path']) as z: source = z['source']
    start = time.monotonic(); controls = fuse(received, current, patches, done['generated'])
    fusion_seconds = time.monotonic()-start
    variants = {'current':current, **controls}
    boundary = edges(source.shape, old['detail']['received_regions'], done['generated'])
    results = {}
    for name, output in variants.items():
        run.check()
        results[name] = dict(quality=quality(source, output, metric),
            boundaries={cat:aggregate([measure(source, output, e) for e in boundary if e['category'] == cat])
                        for cat in CATEGORIES})
    for key, value in results['current']['quality'].items():
        if abs(value-read(point/'result.json')['quality'][key]) > 1e-8:
            raise ValueError('cached current metric differs: '+key)
    atomic_npz(dest/'controls.npz', **controls)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    eligible = [e for e in boundary if e['category'] == 'G_G' and 8 in e['frames']]
    edge = eligible[0] if eligible else boundary[0]
    p, center = edge['pos'], (edge['lo']+edge['hi'])//2
    x, y = (p-48, center-48) if edge['axis'] == 'x' else (center-48, p-48)
    for j, (name, pixels) in enumerate([('source', source), *variants.items()]):
        axs[0, j].imshow(pixels[8]); axs[0, j].set_title(name); axs[0, j].axis('off')
        axs[1, j].imshow(pixels[8, y:y+96, x:x+96]); axs[1, j].axis('off')
    fig.suptitle(row['sample_id']+' | same packets, same G outputs | first G/G edge')
    fig.savefig(dest/'comparison.png', dpi=140); plt.close(fig)
    result = dict(complete=True, sample_id=row['sample_id'], dataset=row['dataset'],
        original_current_exact=True, same_generated_halos=True, actual_bytes=done['actual_bytes'],
        experiment='offline display-fusion controls; not a newly signaled fresh-decoder profile',
        mask_bytes=0, variants=results, two_controls_cpu_seconds=fusion_seconds,
        capture_receipt_sha256=digest(folder/'complete.json'),
        artifacts={n:digest(dest/n) for n in ('controls.npz', 'comparison.png')})
    save(dest/'complete.json', result); return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT/'p1_controls')
    args = p.parse_args()
    if not os.environ.get('TMUX'): raise RuntimeError('tmux required')
    summary = read(EVALUATION/'summary.json')
    rows = summary['scope']['rows']; rows = [rows[0], rows[6], *rows[1:6], *rows[7:]]
    run = Run(SimpleNamespace(output=args.output, command='controls', max_hours=12)); run.thread.start()
    try:
        immutable(run.root/'protocol.json', dict(version='overlap_controls_v1', rows=rows,
            E_byte_cap=.5, policies_frozen=True, output_crop='first eligible G/G, frame9',
            code={n:digest(Path(__file__).resolve().parents[1]/n) for n in
                  ('tools/fusion_pilot.py', 'routervc/fusion/blend.py', 'routervc/fusion/capture.py',
                   'tools/fusion_capture_worker.py', 'routervc/fusion/boundaries.py')}))
        results = []; metric = LPIPSAlex(True)
        with exclusive_native_evaluation(run):
            for index, row in enumerate(rows):
                sid = row['sample_id']; folder = run.root/'samples'/sid
                point = EVALUATION/'samples'/sid/'source_e050'
                run.update(phase='capture_and_paired_controls', completed=index, total=len(rows), sample=sid)
                worker(run, point/'stream.rvlrg', folder)
                results.append(compare(run, row, point, folder, metric))
        groups = {}
        for dataset in ('REDS', 'UVG'):
            subset = [r for r in results if r['dataset'] == dataset]
            groups[dataset] = {name:{key:float(np.mean([r['variants'][name]['quality'][key] for r in subset]))
                            for key in ('lpips_alex', 'psnr_db', 'temporal_delta_mae')}
                            for name in ('current', 'overlap', 'multiband')}
        save(run.root/'summary.json', dict(complete=True, results=results, groups=groups))
        save(run.root/'complete.json', dict(complete=True, points=len(results),
            artifacts={n:digest(run.root/n) for n in ('protocol.json', 'summary.json')}))
        run.update(phase='complete', completed=len(rows)); run.log_resources()
    except BaseException as error:
        save(run.root/'last_failure.json', dict(error=repr(error), progress=run.progress)); raise
    finally: run.stop.set(); run.thread.join(); run.lock.close()


if __name__ == '__main__': main()
