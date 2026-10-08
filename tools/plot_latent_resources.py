"""All-call accumulated fresh-worker memory, distinct from device sampling."""
import os
import time
from pathlib import Path
import numpy as np
from demo.scalable_codec import file_hash, atomic_json
from tools.latent_probe import read
from tools.latent_packet_probe import ROOT
from tools.latent_stream_probe import verify


def main():
    if not os.environ.get('TMUX'):
        raise RuntimeError('tmux required')
    began = time.monotonic()
    while not (ROOT/'report/complete.json').exists():
        if time.monotonic()-began > 6*3600:
            raise TimeoutError('main report unfinished')
        time.sleep(20)
    out = ROOT/'resources'
    out.mkdir(parents=True, exist_ok=True)
    if (out/'complete.json').exists():
        done = verify(out)
        for p, sha in done['inputs'].items():
            if file_hash(Path(p)) != sha:
                raise ValueError('resource report input changed')
        return
    paths = [ROOT/s/'summary.json' for s in ('continuous', 'generation')]
    inputs = {str(p):file_hash(p) for p in paths+[Path(__file__)]}
    plain = read(paths[0])['results']; generated = read(paths[1])['results']
    groups = {}
    for group, prefix in [('REDS', 'reds'), ('UVG crops', 'uvg')]:
        groups[group] = {}
        for name, cases in [('B/E fresh', [c for c in plain if c['frames'] == 17]), ('B/E + G4 fresh', generated)]:
            points = [c['points'][f'E{n}'] for c in cases if c['sample'].startswith(prefix) for n in (0, 4, 8, 16)]
            groups[group][name] = dict(max_GiB=max(p['peak_cuda_allocated_bytes']/2**30 for p in points),
                mean_GiB=float(np.mean([p['peak_cuda_allocated_bytes']/2**30 for p in points])),
                mean_seconds=float(np.mean([p['seconds'] for p in points])))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    labels = [f'{group}\n{name}' for group, items in groups.items() for name in items]
    for ax, field, label in zip(axes, ('max_GiB', 'mean_seconds'), ('Max CUDA allocated (GiB)', 'Mean fresh-worker seconds')):
        vals = [v[field] for items in groups.values() for v in items.values()]
        ax.bar(range(4), vals, color=['#4682a9', '#dc8b38']*2)
        for i, v in enumerate(vals):
            ax.text(i, v, f'{v:.2f}', ha='center', va='bottom')
        ax.set_xticks(range(4), labels, fontsize=8); ax.set_ylabel(label); ax.set_ylim(0, max(vals)*1.18)
    fig.suptitle('17 frames; 2 REDS + 2 UVG crops; E0/E4/E8/E16\n'
                 'G peak accumulated across all ROI resets; time includes loading, not realtime throughput')
    fig.savefig(out/'resources.png', dpi=150); plt.close(fig)
    atomic_json(out/'summary.json', groups)
    atomic_json(out/'complete.json', dict(complete=True, inputs=inputs,
        artifacts={n:file_hash(out/n) for n in ('summary.json', 'resources.png')}))


if __name__ == '__main__':
    main()
