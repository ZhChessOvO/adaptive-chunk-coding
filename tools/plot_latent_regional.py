"""Width3 measured prefix savings, RD and fixed images; no neural inference."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

import numpy as np

from demo.scalable_codec import atomic_json, file_hash
from tools.latent_probe import read
from tools.latent_stream_probe import DEFAULT as STAGE_B, verify
from tools.latent_regional_probe import DEFAULT as REGIONAL, ORDER, COUNTS
from tools.latent_native_baseline import DEFAULT as ANCHORS
from tools.plot_latent_diagnostics import native_means, comparison_image

DEFAULT = REGIONAL.parent / 'regional_report'


def aggregate(cases):
    result = {}
    for group, prefix in (('REDS', 'reds'), ('UVG crop', 'uvg')):
        samples = [c for c in cases if c['sample'].startswith(prefix)]
        if not samples:
            raise ValueError('missing dataset group')
        result[group] = {}
        for count in COUNTS:
            points = [s['points'][f'E{count}'] for s in samples]
            result[group][str(count)] = dict(
                samples=len(samples), region_count=count,
                **{k: float(np.mean([p[k] for p in points])) for k in (
                    'actual_bytes', 'bpp', 'saving_vs_native_full', 'saving_vs_regional_full',
                    'base_bytes', 'E_wire_bytes', 'E_header_bytes', 'region_wrapper_bytes',
                    'seconds', 'peak_cuda_allocated_bytes')},
                **{k: float(np.mean([p['quality_all9'][k] for p in points]))
                   for k in ('psnr_db', 'lpips_alex')})
    return result


def draw(summary, native, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for row, (group, values) in enumerate(summary.items()):
        points = [values[str(n)] for n in COUNTS]
        refs = native[group]
        for col, (key, label) in enumerate((('psnr_db', 'PSNR (dB, higher better)'),
                                           ('lpips_alex', 'LPIPS (lower better)'))):
            ax = axes[row, col]
            ax.plot([p['bpp'] for p in refs], [p[key] for p in refs], 'k.-',
                    label='Native UF scalar QP sweep (I32)')
            ax.plot([p['bpp'] for p in points], [p[key] for p in points], 'o-',
                    color='#2675b6', label='Width3: one B, appended regional E')
            for point in points:
                ax.annotate(f'E{point["region_count"]}', (point['bpp'], point[key]),
                            xytext=(3, 5), textcoords='offset points', fontsize=8)
            ax.set_title(group)
            ax.set_xlabel('Actual file bpp, all 9 frames')
            ax.set_ylabel(label)
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
    fig.suptitle('Fixed center-out regional prefixes; width3, q*=48; G / Router off\n'
                 'E counts mean regions, NOT byte fractions; four diagnostic windows', fontsize=12)
    fig.savefig(output / 'regional_rd.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, (group, values) in zip(axes, summary.items()):
        saving = [100*values[str(n)]['saving_vs_native_full'] for n in COUNTS]
        ax.bar(range(len(COUNTS)), saving, color=['#2675b6' if v >= 0 else '#b64040' for v in saving])
        for x, value in enumerate(saving):
            ax.text(x, value+1 if value >= 0 else value-1, f'{value:.1f}%', ha='center',
                    va='bottom' if value >= 0 else 'top')
        ax.axhline(0, color='black', linewidth=.8)
        ax.set_xticks(range(len(COUNTS)), [f'E{n}' for n in COUNTS])
        ax.set_ylim(min(-12, min(saving)-8), max(saving)+10)
        ax.set_ylabel('Fewer bytes vs native I32/P48 (%)')
        ax.set_title(group)
    fig.suptitle('Real savings relative to the HIGH-QUALITY endpoint, NOT equal-quality gains', fontsize=11)
    fig.savefig(output / 'regional_savings.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 5, figsize=(11, 2.7), constrained_layout=True)
    for ax, count in zip(axes, COUNTS):
        mask = np.zeros((4, 4))
        for index in ORDER[:count]:
            mask[divmod(index, 4)] = 1
        ax.imshow(mask, cmap='Blues', vmin=0, vmax=1)
        for index in range(16):
            y, x = divmod(index, 4)
            ax.text(x, y, str(index), ha='center', va='center',
                    color='white' if mask[y, x] else 'black', fontsize=8)
        ax.set_title(f'E{count} / 16')
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle('Fixed packet order visualization only; this mask is NOT transmitted', fontsize=11)
    fig.savefig(output / 'regional_coverage.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    if not os.environ.get('TMUX'):
        raise RuntimeError('tmux required')
    if args.wait:
        started = time.monotonic()
        while not (REGIONAL / 'complete.json').exists():
            if time.monotonic()-started > 4*3600:
                raise TimeoutError('regional evidence incomplete')
            time.sleep(15)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'complete.json').exists():
        done = verify(out)
        for name, sha in done['inputs'].items():
            if file_hash(Path(name)) != sha:
                raise ValueError('regional report input changed')
        print('Verified regional report without redrawing', flush=True)
        return
    verify(REGIONAL)
    verify(ANCHORS)
    cases = read(REGIONAL / 'summary.json')['results']
    summary = aggregate(cases)
    refs = native_means(read(ANCHORS / 'summary.json')['rows'])
    bindings = {}
    for root in (REGIONAL, ANCHORS):
        for name in ('complete.json', 'protocol.json', 'summary.json'):
            bindings[str(root / name)] = file_hash(root / name)
    bindings[str(Path(__file__))] = file_hash(Path(__file__))
    images = []
    inputs = read(REGIONAL / 'protocol.json')['inputs']
    for case in cases:
        sid = case['sample']
        folder = REGIONAL / sid
        verify(folder)
        bindings[str(folder / 'complete.json')] = file_hash(folder / 'complete.json')
        source_path = Path(inputs[sid]['source_path'])
        if file_hash(source_path) != inputs[sid]['source_sha256']:
            raise ValueError('regional visual source changed')
        bindings[str(source_path)] = inputs[sid]['source_sha256']
        with np.load(source_path, allow_pickle=False) as f:
            panels = [('Source', f['source'][5].copy())]
        for count in COUNTS:
            path = folder / f'E{count}'
            point = verify(path)
            bindings[str(path / 'pixels.npz')] = file_hash(path / 'pixels.npz')
            with np.load(path / 'pixels.npz', allow_pickle=False) as f:
                panels.append((f'E{count}: {point["bpp"]:.5f} bpp', f['reconstruction'][5].copy()))
        name = f'{sid}.png'
        comparison_image(panels, out / name, f'{sid} | fixed frame 5 | actual received prefixes; G off')
        images.append(name)
    draw(summary, refs, out)
    atomic_json(out / 'summary.json', dict(summary=summary, cases=cases, native=refs,
        saving_scope='per-window 1 - actual_prefix_bytes/native_I32_P48_bytes, then equal mean; not BD-rate'))
    names = ['regional_rd.png', 'regional_savings.png', 'regional_coverage.png', 'summary.json']+images
    atomic_json(out / 'complete.json', dict(complete=True, inputs=bindings, images=images,
        artifacts={name: file_hash(out / name) for name in names}))
    print('Created regional savings/RD/visual report', flush=True)


if __name__ == '__main__':
    main()
