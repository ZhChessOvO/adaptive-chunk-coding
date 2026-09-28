"""CPU-only results digest; no new inference, tuning, or receiver changes."""
import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
from statistics import mean


DEFAULT = Path('/root/autodl-fs/DCVC/runs/a800_conditioned_generation_20260928')
MODES = ('none', 'partial', 'full')
METRICS = ('lpips_alex', 'psnr_db', 'temporal_delta_mae')
CLIPS = {'mechanism-00-reds': 'REDS000', 'mechanism-01-uvg': 'UVG Beauty',
         'mechanism-02-reds': 'REDS001', 'mechanism-03-uvg': 'UVG Jockey'}


def read(path):
    return json.loads(path.read_text())


def aggregate(results, field):
    """Equal clip means, separately within each prefix and dataset."""
    groups = {}
    for mode in MODES:
        groups[mode] = {}
        for dataset in ('all', 'REDS', 'UVG'):
            rows = [r for r in results if r['mode'] == mode and
                    (dataset == 'all' or r['sample_id'].endswith(dataset.lower()))]
            groups[mode][dataset] = {}
            for candidate in ('baseline', 'direct', 'latent', 'image'):
                points = [r for r in rows if r['candidate'] ==
                          ('image' if candidate in ('baseline', 'direct') else candidate)]
                if len(points) != (4 if dataset == 'all' else 2) or len({
                        r['sample_id'] for r in points}) != len(points):
                    raise ValueError('expected four distinct clips, two per dataset, per candidate/prefix')
                if candidate in ('baseline', 'direct'):
                    points = [r[candidate] for r in points]
                groups[mode][dataset][candidate] = {
                    k: mean(r[field][k] for r in points) for k in METRICS}
    return groups


def make_figures(root, rows, conditional):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    labels = {'direct': 'No G: Base / partial E / full E', 'baseline': 'Old generator',
              'latent': 'Latent adaptation', 'image': 'Image-objective adaptation'}
    for metric, ylabel in [('lpips_alex', 'Local LPIPS (lower is better)'),
                           ('psnr_db', 'Local PSNR (dB; higher is better)')]:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
        for ax, row in zip(axes.flat, rows):
            for key, label in labels.items():
                points = [row['prefixes'][m][key] for m in MODES]
                ax.plot([r['bytes'] for r in points],
                        [r['roi_quality'][metric] for r in points], 'o-', label=label)
            ax.set(title=row['name'], xlabel='Actual total stream bytes', ylabel=ylabel)
            ax.grid(alpha=.3)
            ax.legend(fontsize=8)
        fig.savefig(root/f'prefix_with_direct_{metric}.png', dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    x = np.arange(len(rows))
    for i, key in enumerate(('baseline', 'latent', 'image')):
        for ax, metric in zip(axes, ('lpips_benefit_from_E', 'psnr_benefit_from_E')):
            ax.bar(x+(i-1)*.25, [r['gains'][key][metric] for r in conditional],
                   width=.25, label=labels[key])
    for ax, title in zip(axes, ('Extra E: LPIPS reduction', 'Extra E: PSNR increase (dB)')):
        ax.set(title=title, xticks=x, xticklabels=[r['name'] for r in rows])
        ax.grid(axis='y', alpha=.3)
        ax.legend(fontsize=8)
    fig.suptitle('Same generator: no E to full E; E costs additional bytes')
    fig.savefig(root/'incremental_E_benefit.png', dpi=150)
    plt.close(fig)


def digest(root):
    evaluation = root/'evaluation'
    summary = read(evaluation/'summary.json')
    analysis = read(evaluation/'analysis.json')
    audit = read(evaluation/'audit.json')
    assert summary['complete'] and not summary['smoke']
    assert len(summary['results']) == 24
    assert audit['checks']['fresh_decodes'] == 27
    assert all(audit['checks'].values())
    results = summary['results']
    local, whole = [aggregate(results, field) for field in ('roi_quality', 'quality')]
    for mode in MODES:
        for dataset in ('all', 'REDS', 'UVG'):
            for candidate in ('baseline', 'direct', 'latent', 'image'):
                for metric in METRICS:
                    assert abs(local[mode][dataset][candidate][metric] -
                               analysis['groups'][mode][dataset][candidate][metric]) < 1e-12
    rows = []
    assert {r['sample_id'] for r in results} == set(CLIPS)
    for sid, name in CLIPS.items():
        prefixes = {}
        for mode in MODES:
            image = next(r for r in results if (r['sample_id'], r['mode'], r['candidate']) ==
                         (sid, mode, 'image'))
            latent = next(r for r in results if (r['sample_id'], r['mode'], r['candidate']) ==
                          (sid, mode, 'latent'))
            prefixes[mode] = {key: {k: p[k] for k in ('bytes', 'roi_quality', 'quality', 'per_region')}
                             for key, p in dict(baseline=image['baseline'], direct=image['direct'],
                                                latent=latent, image=image).items()}
        rows.append(dict(sample_id=sid, name=name, prefixes=prefixes))
    heartbeats = [json.loads(line) for line in (root/'heartbeat.jsonl').read_text().splitlines()]
    phases = {}
    for mode in ('train', 'evaluation'):
        events = [v for v in heartbeats if v['mode'] == mode]
        first, last = events[0], events[-1]
        phases[mode] = dict(
            start_utc=(datetime.fromisoformat(first['utc']) -
                       timedelta(seconds=first['elapsed_seconds'])).isoformat(),
            end_utc=last['utc'], seconds=last['elapsed_seconds'],
            sampled_gpu_peak_mib=max(int(v['gpu'].split(',')[2]) for v in events),
            disks_at_end=last['disks'])
    timing = {}
    for candidate in ('latent', 'image'):
        full = [r for r in results if r['candidate'] == candidate and r['mode'] == 'full']
        timing[candidate] = dict(
            receiver_seconds=mean(r['fresh_decode']['seconds'] for r in full),
            process_and_metrics_seconds=mean(r['process_wall_seconds'] for r in full),
            generation_seconds=mean(sum(w['runtime']['seconds_model_load_excluded']
                for w in r['fresh_decode']['generation_runtime']['windows']) for r in full),
            peak_cuda_allocated_bytes=max(r['fresh_decode']['peak_cuda_allocated_bytes'] for r in full))
    values = dict(local=local, whole_frame=whole, clips=rows, phases=phases, timing=timing,
                  conditional_effect=analysis['conditional_effect'],
                  full_prefix_image_lpips_reduction_fraction=1-local['full']['all']['image']['lpips_alex']/
                  local['full']['all']['baseline']['lpips_alex'],
                  image_lpips_improved_points=sum(r['roi_quality']['lpips_alex'] <
                      r['baseline']['roi_quality']['lpips_alex'] for r in results if r['candidate']=='image'),
                  whole_frame_label='All frames: 17,17,33,17; equal clip means, not pooled pixels',
                  local_label='First 17 frames of fixed ROIs; area-weighted within clip, equal clip means')
    conditional = {r['sample_id']: r for r in analysis['conditional_effect']}
    make_figures(evaluation, rows, [conditional[r['sample_id']] for r in rows])
    target = evaluation/'digest.json'
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(values, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(target)
    print(json.dumps({k: values[k] for k in ('phases', 'timing',
                     'full_prefix_image_lpips_reduction_fraction', 'image_lpips_improved_points')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT)
    digest(parser.parse_args().output)
