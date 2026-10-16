"""CPU-only completion audit and compact views of the bounded G finish.

No inference, metric recomputation, model promotion or training. Original
experiment files stay immutable; completed report re-entry verifies hashes.
"""
import argparse
from collections import Counter, defaultdict
import fcntl
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import torch

from demo.routervc_fullview_probe import read, digest, immutable, verify_artifacts
from demo.routervc_mixed_queue import exact
from demo.scalable_codec import atomic_json
from demo.scalable_format import frame_hash
from routervc.fusion.blend import geometry
from routervc.generation import integration as integration
from routervc.generation.texture_data import REPO
from routervc.latent import generation
from tools.cooperative_review import source_rgb
from tools.fusion_finish import memory_snapshot
from tools.g_bounded_finish import DEFAULT, CHAIN, verify
from tools.g_integration import validate
from tools.g_integration_report import canvas
from tools.storage_guard import disk_snapshot
from tools.texture_ablation_detail import resource_figure
from tools.texture_context_report import plotting

ARMS = ('received', 'legacy', 'start', 'step400', 'step800')
LABELS = ('Received Y / G off', 'Legacy production', 'Starting repaired G',
          'Same recipe +400', 'Same recipe +800')


def metrics(point):
    return {**point['quality'], 'phase_per_frame': point['G_core_phase']['per_frame_rms_mean']}


def paired(summary):
    reviews = {r['step']: r['cases'] for r in summary['reviews']}
    if len(summary['reviews']) != 2 or set(reviews) != {400, 800}:
        raise ValueError('exactly two milestone reviews required')
    indexed = {}
    for step, rows in reviews.items():
        indexed[step] = {r['sample_id']: r for r in rows}
        if len(indexed[step]) != len(rows):
            raise ValueError('duplicate sample')
    if set(indexed[400]) != set(indexed[800]):
        raise ValueError('unpaired samples')
    out = []
    for a in reviews[400]:
        b = indexed[800][a['sample_id']]
        for key in ('cohort', 'dataset', 'actual_bytes', 'bpp', 'regions', 'references'):
            if a[key] != b[key]:
                raise ValueError('milestone pairing changed: ' + key)
        values = {n: metrics(a['references'][k]) for n, k in
                  (('received', 'received'), ('legacy', 'legacy'), ('start', 'control'))}
        values.update(step400=metrics(a), step800=metrics(b))
        out.append(dict(sample_id=a['sample_id'], cohort=a['cohort'], dataset=a['dataset'],
            actual_bytes=a['actual_bytes'], bpp=a['bpp'], models=values,
            resources={400: a['resources'], 800: b['resources']},
            step800_minus_start={k: values['step800'][k]-values['start'][k] for k in values['start']}))
    return out


def grouped(cases):
    buckets = defaultdict(list)
    for row in cases:
        buckets[row['cohort'], row['dataset']].append(row)
    groups = []
    for (cohort, dataset), rows in sorted(buckets.items()):
        values = {arm: {k: float(np.mean([r['models'][arm][k] for r in rows]))
                       for k in rows[0]['models'][arm]} for arm in ARMS}
        groups.append(dict(cohort=cohort, dataset=dataset, count=len(rows), models=values,
            mean_bpp=float(np.mean([r['bpp'] for r in rows])),
            lpips_improved=sum(r['step800_minus_start']['lpips_alex'] < 0 for r in rows),
            phase_increased=sum(r['step800_minus_start']['phase_per_frame'] > 0 for r in rows)))
    return groups


def audit(root, cfg):
    verify(root)
    train = root/'formal/continuation'
    saved = torch.load(train/'resume.pt', map_location='cpu', weights_only=True)
    for path in (train/'adapter.pt', train/'checkpoints/adapter_000800.pt'):
        exact(saved['adapter'], torch.load(path, map_location='cpu', weights_only=True))
    journal = [json.loads(s) for s in (train/'steps.jsonl').read_text().splitlines()]
    if Counter(r['dataset'] for r in journal) != {'REDS': 600, 'UVG': 200}:
        raise ValueError('dataset schedule changed')
    if Counter(r['crop'][2] for r in journal) != {256: 200, 512: 600}:
        raise ValueError('mixed-view schedule changed')
    smoke = root/'smoke/continuation'
    exact(*(torch.load(smoke/m/'resume.pt', map_location='cpu', weights_only=True)
            for m in ('direct', 'resumed')))
    evidence = []
    if len(cfg['chain']['jobs']) != 6:
        raise ValueError('expected six fixed windows')
    legacy = torch.load(cfg['chain']['models']['legacy']['path'], map_location='cpu', weights_only=True)
    for step in (400, 800):
        stage = root/'review'/f'step_{step:06d}'
        spec = read(stage/'protocol.json')['model']
        original = train/'checkpoints'/f'adapter_{step:06d}.pt'
        if digest(original) != spec['original_sha256'] or digest(spec['path']) != spec['sha256']:
            raise ValueError('milestone model identity changed')
        exact(torch.load(spec['path'], map_location='cpu', weights_only=True),
              integration.compatible_bundle(torch.load(original, map_location='cpu', weights_only=True), legacy))
        for j, job in enumerate(cfg['chain']['jobs']):
            sid = job['row']['sample_id']
            old_inner, _ = integration.stream.parse(Path(job['point']['wire']).read_bytes())
            folder = stage/'samples'/sid/f'continuation{step}'
            for tag in (('main', 'repeat') if step == 400 and j < 2 else ('main',)):
                dest = folder/tag
                receipt = read(dest/'complete.json')
                verify_artifacts(dest, receipt['artifacts'])
                validate(receipt, job['receipt'])
                wire = dest/'stream.rvlg'
                inner, header = integration.parse(wire.read_bytes())
                if inner != old_inner or wire.stat().st_size != receipt['actual_bytes']:
                    raise ValueError('B/E payload or real byte count changed')
                if digest(wire) != receipt['stream'] or header['assets_sha256'] != generation.asset_hash(spec['assets']):
                    raise ValueError('transmitted model identity differs')
                with np.load(dest/'pixels.npz', allow_pickle=False) as z:
                    for key, name in (('base', 'base_hash'), ('enhanced', 'enhanced_hash'), ('reconstruction', 'output_hash')):
                        if frame_hash(z[key]) != receipt[name]:
                            raise ValueError('saved pixels differ')
                    support = geometry(z['enhanced'].shape, receipt['generated'])[0]
                    np.testing.assert_array_equal(z['reconstruction'][:, ~support], z['enhanced'][:, ~support])
                if tag == 'repeat' and receipt['output_hash'] != read(folder/'main/complete.json')['output_hash']:
                    raise ValueError('actual reload differs')
                evidence.append(dict(sample=sid, step=step, tag=tag, actual_bytes=wire.stat().st_size))
            historical = read(CHAIN/'samples'/sid/'scores.json')
            score = read(stage/'scores'/f'{sid}.json')
            for name in ('received', 'legacy', 'control'):
                if historical['metrics'][name] != score['references'][name]:
                    raise ValueError('reference score changed')
            for arm in ('legacy', 'control'):
                dest = CHAIN/'samples'/sid/arm/'main'
                verify_artifacts(dest, read(dest/'complete.json')['artifacts'])
    return dict(complete=True, steps=800, optimizer_states=82, exact_restart=True,
        main_outputs=12, actual_reload_repeats=2, non_G_unchanged=True, conditions_noise_regions_fixed=True,
        same_inner_B_E=True, no_masks=True, no_promotion=True, evidence=evidence)


def charts(output, groups):
    plt = plotting()
    fig, axes = plt.subplots(len(groups), 3, figsize=(13, 9), layout='constrained')
    for i, group in enumerate(groups):
        for j, (key, label) in enumerate((('lpips_alex', 'Whole-frame LPIPS / lower'),
                ('psnr_db', 'Whole-frame PSNR dB / higher'), ('phase_per_frame', 'G-core phase error / auxiliary'))):
            ax = axes[i, j]
            vals = [group['models'][a][key] for a in ('start', 'step400', 'step800')]
            ax.plot([0, 400, 800], vals, 'o-', color='#248b83', label='Same-recipe continuation')
            ax.axhline(group['models']['legacy'][key], color='#ac713e', ls='--', label='Legacy production')
            for x, y in zip((0, 400, 800), vals):
                ax.annotate(f'{y:.4f}', (x, y), xytext=(0, 7), textcoords='offset points', ha='center', fontsize=8)
            ax.set_xticks([0, 400, 800]); ax.margins(x=.18, y=.35); ax.grid(alpha=.2)
            ax.set_title(f'{group["dataset"]} / {group["cohort"]} / n={group["count"]}\n{label}', fontsize=10)
            ax.set_xlabel('Additional training steps'); ax.legend(fontsize=7)
    fig.suptitle('Same E50 bytes, G regions, native geometry and noise / NOT an RD curve\nSix reused diagnostic/internal windows, not full test sets')
    fig.savefig(output/'trajectory.png', dpi=140); plt.close(fig)


def figures(output, root, job, point):
    sid = point['sample_id']; source = source_rgb(job['row']); arrays = {}
    for name, arm in (('legacy', 'legacy'), ('start', 'control')):
        with np.load(CHAIN/'samples'/sid/arm/'main/pixels.npz', allow_pickle=False) as z:
            arrays[name] = z['reconstruction'].copy()
            arrays['received'] = z['enhanced'].copy()
    for step in (400, 800):
        with np.load(root/'review'/f'step_{step:06d}'/'samples'/sid/f'continuation{step}/main/pixels.npz', allow_pickle=False) as z:
            arrays[f'step{step}'] = z['reconstruction'].copy()
    panels = [('Original / frame 9\nWhole 17-frame scores; no new crop', source)]
    for name, label in zip(ARMS, LABELS):
        q = point['models'][name]
        panels.append((f'{label}\nLPIPS {q["lpips_alex"]:.5f} | PSNR {q["psnr_db"]:.3f}', arrays[name]))
    canvas(panels, 8, width=min(768, source.shape[2])).save(output/(sid+'.png'))
    for name in ('start', 'step800'):
        Image.fromarray(arrays[name][8]).save(output/(sid+'_'+name+'_native.png'))
    path = output/(sid+'.mp4'); tmp = path.with_suffix('.partial.mp4')
    panels[0] = ('Original / 17 frames / 4fps slow preview\nNot playback speed or metric input', source)
    first = canvas(panels, 0, width=512)
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{first.width}x{first.height}', '-r', '4', '-i', 'pipe:0', '-an', '-c:v', 'libx264',
        '-threads', '2', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(tmp)]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as child:
        try:
            for t in range(17):
                child.stdin.write(canvas(panels, t, width=512).tobytes())
        finally:
            child.stdin.close()
        if child.wait() != 0:
            raise RuntimeError('preview failed')
    tmp.replace(path)


def resources(root, output, cases):
    log = [json.loads(s) for s in (root/'memory.jsonl').read_text().splitlines()]
    training = read(root/'formal/continuation/complete.json')
    measured = resource_figure(root, output)
    infer = []
    for step in (400, 800):
        for dataset in ('REDS', 'UVG'):
            rows = [p['resources'][step] for p in cases if p['dataset'] == dataset]
            infer.append(dict(step=step, dataset=dataset, count=len(rows),
                mean_seconds=float(np.mean([p['seconds'] for p in rows])),
                peak_allocated_gib=max(p['peak_cuda_allocated_bytes'] for p in rows)/2**30,
                peak_reserved_gib=max(p['peak_cuda_reserved_bytes'] for p in rows)/2**30))
    return dict(**measured, training_minutes=training['seconds']/60,
        training_allocated_peak_gib=training['peak_cuda_allocated_bytes']/2**30, inference=infer,
        inference_scope='fresh worker includes model hashing/loading, B/E, Rg/G/fusion, per-region writes; excludes final NPZ and scoring',
        cgroup_peak_charged_gib=max(m['charged'] for m in log)/2**30,
        cgroup_peak_rss_gib=max(m['rss'] for m in log)/2**30,
        cgroup_peak_shmem_gib=max(m['shmem'] for m in log)/2**30,
        failcnt_start=log[0]['failcnt'], failcnt_end=log[-1]['failcnt'],
        cgroup_current=memory_snapshot(), current_disks=disk_snapshot(),
        run_logical_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()))


def main(args):
    if not os.environ.get('TMUX'):
        raise RuntimeError('tmux required')
    root = args.root; output = root/'report'; output.mkdir(exist_ok=True)
    with (root/'bounded_finish.lock').open('a') as lock, (output/'report.lock').open('a') as report_lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(report_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if read(root/'bounded_finish.progress.json')['phase'] != 'complete_review_required_no_more_training':
            raise RuntimeError('training/evaluation still incomplete')
        names = ('complete.json', 'summary.json', 'protocol.json', 'heartbeat.jsonl', 'memory.jsonl')
        inputs = {str(root/n): digest(root/n) for n in names}
        inputs.update({str(CHAIN/'samples'/job['row']['sample_id']/'scores.json'):
            digest(CHAIN/'samples'/job['row']['sample_id']/'scores.json') for job in read(root/'protocol.json')['chain']['jobs']})
        code = {n: digest(REPO/n) for n in ('tools/g_bounded_finish_report.py',
            'tools/test_g_bounded_finish_report.py', 'tools/g_integration_report.py',
            'tools/texture_ablation_detail.py', 'tools/texture_context_report.py', 'tools/cooperative_review.py')}
        immutable(output/'protocol.json', dict(inputs=inputs, code=code,
            scope='saved results only; no inference, metric recomputation or promotion'))
        if (output/'complete.json').exists():
            verify_artifacts(output, read(output/'complete.json')['artifacts'])
            print('READ_ONLY_REENTRY_OK'); return
        cfg = read(root/'protocol.json'); evidence = audit(root, cfg)
        cases = paired(read(root/'summary.json')); groups = grouped(cases)
        atomic_json(output/'audit.json', evidence)
        atomic_json(output/'compact.json', dict(cases=cases, groups=groups, no_promotion=True))
        charts(output, groups)
        for job, point in zip(cfg['chain']['jobs'], cases, strict=True):
            if job['row']['sample_id'] != point['sample_id']:
                raise ValueError('case order differs')
            figures(output, root, job, point)
            print('REVIEW_READY '+point['sample_id'], flush=True)
        atomic_json(output/'resources.json', resources(root, output, cases))
        files = [p for p in output.iterdir() if p.is_file() and p.suffix != '.lock']
        immutable(output/'complete.json', dict(complete=True, no_promotion=True,
            artifacts={p.name: digest(p) for p in files}))
        print(json.dumps(dict(complete=True, audit=evidence, groups=groups)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=DEFAULT)
    main(p.parse_args())
