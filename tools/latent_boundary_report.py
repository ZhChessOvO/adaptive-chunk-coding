"""CPU-only resumable P0: inspect immutable completed width3 sender outputs."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from demo.chunk_enhancement_experiment import Run
from demo.routervc_fullview_probe import read, digest, save, immutable, verify_artifacts
from routervc.fusion.boundaries import CATEGORIES, edges, measure, aggregate

ROOT = Path('/root/autodl-fs/DCVC/runs/routervc_fusion_20261010')
EVALUATION = Path('/root/autodl-fs/DCVC/runs/routervc_latent_sender_20261009/evaluation')


def figure(path, source, received, output, boundaries, sid):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, axs = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True)
    frame = 8
    for ax, pixels, title in zip(axs[0], (source, received, output), ('Source', 'B + E (Y)', 'Current G + feather')):
        ax.imshow(pixels[frame]); ax.set_title(title); ax.axis('off')
    colors = dict(E_nonE='orange', G_nonG='cyan', G_G='magenta')
    chosen = []
    for ax, category in zip(axs[1], CATEGORIES):
        candidates = [e for e in boundaries if e['category'] == category and frame in e['frames']]
        if not candidates:
            ax.text(.5, .5, 'No eligible edge', ha='center'); ax.axis('off'); chosen.append(None); continue
        edge = candidates[0]  # Fixed raster order, not a selected best/worst example.
        p, lo, hi = edge['pos'], edge['lo'], edge['hi']
        if edge['axis'] == 'x':
            x, y, w, h = p-48, max(0, (lo+hi)//2-48), 96, 96
            axs[0, 2].plot([p, p], [lo, hi], color=colors[category], linewidth=1)
        else:
            x, y, w, h = max(0, (lo+hi)//2-48), p-48, 96, 96
            axs[0, 2].plot([lo, hi], [p, p], color=colors[category], linewidth=1)
        crop = np.concatenate([v[frame, y:y+h, x:x+w] for v in (source, received, output)], axis=1)
        ax.imshow(crop); ax.set_title(category + ': Source | Y | Current', color=colors[category]); ax.axis('off')
        chosen.append(dict(edge=edge, crop=[x, y, w, h]))
    fig.suptitle(sid + ' | frame 9 | first eligible edge per category')
    fig.savefig(path, dpi=130); plt.close(fig)
    return chosen


def report(run, evaluation):
    summary = read(evaluation/'summary.json')
    immutable(run.root/'protocol.json', dict(version='boundary_p0_v1',
        evaluation=str(evaluation), summary_sha256=digest(evaluation/'summary.json'),
        code={n:digest(Path(__file__).resolve().parents[1]/n) for n in
              ('tools/latent_boundary_report.py', 'routervc/fusion/boundaries.py')},
        arms=['source_e025', 'source_e050', 'source_e100'], radius=24, band=16,
        frame=8, illustrative_edges='first eligible in fixed raster order',
        categories_overlap=True, source_referenced=True, training=False))
    records = []
    for row in summary['scope']['rows']:
        source = None
        for suffix in ('025', '050', '100'):
            run.check(); sid = row['sample_id']; arm = 'source_e'+suffix
            folder = evaluation/'samples'/sid/arm
            dest = run.root/'samples'/sid/arm
            dest.mkdir(parents=True, exist_ok=True)
            if (dest/'result.json').exists():
                result = read(dest/'result.json'); verify_artifacts(dest, result['artifacts'])
                if result['receipt_sha256'] != digest(folder/'receive/complete.json'):
                    raise ValueError('input receipt changed')
            else:
                if source is None:
                    if digest(Path(row['source_path'])) != row['source_sha256']:
                        raise ValueError('source changed')
                    with np.load(row['source_path']) as z: source = z['source']
                receipt = read(folder/'receive/complete.json')
                verify_artifacts(folder/'receive', receipt['artifacts'])
                if digest(folder/'stream.rvlrg') != receipt['stream_sha256']:
                    raise ValueError('wire changed')
                with np.load(folder/'receive/pixels.npz') as z:
                    received, output = z['enhanced'], z['reconstruction']
                boundary = edges(source.shape, receipt['detail']['received_regions'], receipt['generated'])
                values = {}
                for category in CATEGORIES:
                    eligible = [e for e in boundary if e['category'] == category]
                    values[category] = {name:aggregate([measure(source, pixels, e) for e in eligible])
                                        for name, pixels in [('Y', received), ('current', output), ('source', source)]}
                artifacts = {}; chosen = []
                if suffix == '050':
                    chosen = figure(dest/'boundaries.png', source, received, output, boundary, sid)
                    artifacts['boundaries.png'] = digest(dest/'boundaries.png')
                result = dict(sample_id=sid, dataset=row['dataset'], arm=arm,
                    receipt_sha256=digest(folder/'receive/complete.json'), boundaries=boundary,
                    metrics=values, illustrations=chosen, artifacts=artifacts)
                save(dest/'result.json', result)
            records.append(result)
            run.update(phase='boundary_diagnostics', completed=len(records), total=39, sample=sid, arm=arm)
    groups = []
    for dataset in ('REDS', 'UVG'):
        for suffix in ('025', '050', '100'):
            for category in CATEGORIES:
                selected = [r for r in records if r['dataset'] == dataset and r['arm'] == 'source_e'+suffix
                            and r['metrics'][category]['current'] is not None]
                values = {}
                for name in ('Y', 'current', 'source'):
                    values[name] = {key:np.mean([r['metrics'][category][name][key] for r in selected], axis=0).tolist()
                                   for key in ('crossing_gradient_mae', 'band_mae', 'band_temporal_mae',
                                               'normal_detail_profile')} if selected else None
                groups.append(dict(dataset=dataset, arm='source_e'+suffix, category=category,
                                   eligible_views=len(selected), metrics=values))
    save(run.root/'summary.json', dict(complete=True, records=records, groups=groups,
        view_weighting='equal views within each dataset; categories overlap',
        interpretation='Descriptive boundary errors, not proof of perceptually visible seams.'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 3, figsize=(13, 6), constrained_layout=True)
    for i, dataset in enumerate(('REDS', 'UVG')):
        for j, category in enumerate(CATEGORIES):
            group = next(g for g in groups if g['dataset'] == dataset and g['category'] == category and g['arm'] == 'source_e050')
            ax = axs[i, j]
            for name in ('source', 'Y', 'current'):
                if group['metrics'][name]:
                    ax.plot(np.arange(-23, 24), group['metrics'][name]['normal_detail_profile'], label=name)
            ax.axvline(0, color='gray', linestyle=':'); ax.set_title(f'{dataset} {category} (n={group["eligible_views"]})')
            ax.set_xlabel('Distance from boundary [pixels]'); ax.set_ylabel('Normal RGB gradient magnitude'); ax.legend()
    fig.suptitle('Descriptive detail profile | E-byte cap 50% | no new inference')
    fig.savefig(run.root/'profiles.png', dpi=150); plt.close(fig)
    save(run.root/'complete.json', dict(complete=True, points=len(records), inference=False,
        artifacts={n:digest(run.root/n) for n in ('protocol.json', 'summary.json', 'profiles.png')}))
    run.update(phase='complete', completed=len(records)); run.log_resources()
    print(json.dumps({'complete': True, 'points':len(records)}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT/'p0')
    p.add_argument('--evaluation', type=Path, default=EVALUATION)
    args = p.parse_args()
    if not os.environ.get('TMUX'): raise RuntimeError('tmux required')
    run = Run(SimpleNamespace(output=args.output, command='boundary', max_hours=4))
    run.thread.start()
    try: report(run, args.evaluation)
    finally: run.stop.set(); run.thread.join(); run.lock.close()


if __name__ == '__main__': main()
