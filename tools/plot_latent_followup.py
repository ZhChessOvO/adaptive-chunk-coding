"""CPU-only collected packet/continuity/G evidence. Does not rerun inference."""
import argparse
import os
from pathlib import Path
import time

import numpy as np
from demo.scalable_codec import atomic_json, file_hash
from tools.latent_probe import read, SAMPLES
from tools.latent_stream_probe import verify
from tools.latent_packet_probe import ROOT, source_pixels, OLD_R
from tools.latent_regional_probe import COUNTS
from tools.latent_native_baseline import DEFAULT as OLD_NATIVE
from tools.plot_latent_diagnostics import comparison_image


def grouped_single(results):
    summary = {}
    for group, prefix in [('REDS', 'reds'), ('UVG crops', 'uvg')]:
        subset = [r for r in results if r['sample'].startswith(prefix)]
        summary[group] = {}
        for count in COUNTS:
            points = [r['points'][f'E{count}'] for r in subset]
            native = [read(OLD_R/r['sample']/'complete.json')['native_bytes'] for r in subset]
            summary[group][count] = dict(
                new_bytes=float(np.mean([p['actual_bytes'] for p in points])),
                old_bytes=float(np.mean([p['old_bytes'] for p in points])),
                new_bpp=float(np.mean([p['bpp'] for p in points])),
                old_bpp=float(np.mean([p['bpp']*p['old_bytes']/p['actual_bytes'] for p in points])),
                E_saving=None if count == 0 else float(np.mean([1-p['E_wire_bytes']/p['old_E_bytes'] for p in points])),
                total_saving=float(np.mean([1-p['actual_bytes']/p['old_bytes'] for p in points])),
                saving_vs_native=float(np.mean([1-p['actual_bytes']/n for p, n in zip(points, native)])),
                **{key:float(np.mean([p['quality'][key] for p in points])) for key in ('psnr_db', 'lpips_alex')})
    return summary


def compact_plot(single, old_native, out):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for row, (group, values) in enumerate(single.items()):
        prefix = 'reds' if group == 'REDS' else 'uvg'
        native = []
        for q in (0, 8, 16, 24, 32, 40, 48):
            pts = [r for r in old_native if r['sample'].startswith(prefix) and r['qp'] == q]
            native.append(dict(bpp=np.mean([p['bpp'] for p in pts]), **{
                k:np.mean([p['quality_all9'][k] for p in pts]) for k in ('psnr_db', 'lpips_alex')}))
        for col, key in enumerate(('psnr_db', 'lpips_alex')):
            ax = axes[row, col]
            ax.plot([p['bpp'] for p in native], [p[key] for p in native], 'k.-', label='Native UF (I32, P QP sweep)')
            for rate, style, label in [('old_bpp', 'o--', 'Old grouped E'), ('new_bpp', 's-', 'One entropy stream / region')]:
                ax.plot([values[c][rate] for c in COUNTS], [values[c][key] for c in COUNTS], style, label=label)
            ax.set_title(group); ax.set_xlabel('Actual bpp, same 9 frames')
            ax.set_ylabel('PSNR (dB) - higher better' if col == 0 else 'LPIPS - lower better')
            ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Packing only: all new/old pixels identical; width3; G and Router off')
    fig.savefig(out/'packet_rd.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, (group, values) in zip(axes, single.items()):
        x = np.arange(4)
        e = [100*values[c]['E_saving'] for c in COUNTS[1:]]
        total = [100*values[c]['total_saving'] for c in COUNTS[1:]]
        ax.bar(x-.18, e, .36, label='E bytes saved'); ax.bar(x+.18, total, .36, label='Whole stream saved')
        for xx, a, b in zip(x, e, total):
            ax.text(xx-.18, a+.2, f'{a:.1f}%', ha='center', fontsize=9)
            ax.text(xx+.18, b+.2, f'{b:.1f}%', ha='center', fontsize=9)
        ax.set_xticks(x, [f'E{c}/16' for c in COUNTS[1:]])
        ax.set_title(group); ax.set_ylabel('Fewer bytes than old regional format (%)'); ax.legend()
        ax.set_ylim(0, max(e+total)*1.2+1)
    fig.suptitle('No pixel change, no extra mask; means of per-window percentage savings')
    fig.savefig(out/'packet_savings.png', dpi=150); plt.close(fig)


def g_summary(generated):
    summary = {}
    for group, prefix in [('REDS', 'reds'), ('UVG crops', 'uvg')]:
        cases = [r for r in generated if r['sample'].startswith(prefix)]
        summary[group] = {}
        for c in (0, 4, 8, 16):
            points = [r['points'][f'E{c}'] for r in cases]
            summary[group][c] = dict(bpp=float(np.mean([p['bpp'] for p in points])),
                G_on={scope:{k:float(np.mean([p['quality'][scope][k] for p in points]))
                    for k in ('psnr_db', 'lpips_alex', 'temporal_delta_mae')} for scope in ('whole', 'G_regions_mean')},
                G_off={scope:{k:float(np.mean([p['G_off_quality'][scope][k] for p in points]))
                    for k in ('psnr_db', 'lpips_alex', 'temporal_delta_mae')} for scope in ('whole', 'G_regions_mean')},
                mean_GiB=float(np.mean([p['peak_cuda_allocated_bytes']/2**30 for p in points])),
                max_GiB=max(p['peak_cuda_allocated_bytes']/2**30 for p in points),
                seconds=float(np.mean([p['seconds'] for p in points])))
    return summary


def g_plots(summary, native, out):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for row, (group, vals) in enumerate(summary.items()):
        prefix = 'reds' if group == 'REDS' else 'uvg'
        points = [vals[c] for c in (0, 4, 8, 16)]
        refs = []
        for q in (0, 8, 16, 24, 32, 40, 48):
            subset = [r for r in native if r['sample'].startswith(prefix) and r['qp'] == q]
            refs.append(dict(bpp=np.mean([r['bpp'] for r in subset]), **{k:np.mean([
                r['quality_all17'][k] for r in subset]) for k in ('psnr_db', 'lpips_alex')}))
        for col, metric in enumerate(('psnr_db', 'lpips_alex')):
            ax = axes[row, col]
            ax.plot([p['bpp'] for p in refs], [p[metric] for p in refs], 'k.-', label='Native UF, own reference')
            for mode, marker in [('G_off', 'o--'), ('G_on', 's-')]:
                ax.plot([p['bpp'] for p in points], [p[mode]['whole'][metric] for p in points], marker,
                        label='New B/E + fixed G4' if mode == 'G_on' else 'Same bytes, G disabled')
            for c, p in zip((0, 4, 8, 16), points):
                ax.annotate(f'E{c}', (p['bpp'], p['G_on']['whole'][metric]), xytext=(3, 4), textcoords='offset points', fontsize=8)
            ax.set_title(group); ax.set_xlabel('Actual bpp, same 17 frames')
            ax.set_ylabel('PSNR (dB) - higher better' if col == 0 else 'LPIPS - lower better')
            ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Frozen G transfer: fixed center 4 regions; same noise across E prefixes\n'
                 'Existing G weights, NO new Router training; all headers charged')
    fig.savefig(out/'generation_rd.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, (group, vals) in zip(axes, summary.items()):
        for mode, marker in [('G_off', 'o--'), ('G_on', 's-')]:
            ax.plot((0, 4, 8, 16), [vals[c][mode]['G_regions_mean']['lpips_alex'] for c in (0, 4, 8, 16)], marker,
                    label='Before G' if mode == 'G_off' else 'After G')
        ax.set_title(group); ax.set_xlabel('Refined regions / 16, in BOTH P8 chunks')
        ax.set_ylabel('LPIPS inside fixed G cores - lower better'); ax.grid(alpha=.2); ax.legend()
    fig.suptitle('Local G/EG diagnostic; not a whole-video score; matched noise and support')
    fig.savefig(out/'generation_local.png', dpi=150); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wait', action='store_true')
    p.add_argument('--output', type=Path, default=ROOT/'report')
    args = p.parse_args()
    if not os.environ.get('TMUX'):
        raise RuntimeError('tmux required')
    began = time.monotonic()
    required = ('single', 'continuous', 'generation', 'native17')
    while not all((ROOT/s/'complete.json').exists() for s in required):
        if not args.wait or time.monotonic()-began > 6*3600:
            raise RuntimeError('followup evidence incomplete')
        time.sleep(20)
    if (args.output/'complete.json').exists():
        for name, sha in verify(args.output)['inputs'].items():
            if file_hash(Path(name)) != sha:
                raise ValueError('report input changed')
        print('Verified followup report without redrawing', flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output
    bindings = {str(Path(__file__)):file_hash(Path(__file__))}
    summaries = {}
    for stage in required:
        verify(ROOT/stage)
        for name in ('complete.json', 'protocol.json', 'summary.json'):
            bindings[str(ROOT/stage/name)] = file_hash(ROOT/stage/name)
        summaries[stage] = read(ROOT/stage/'summary.json')
    single = grouped_single(summaries['single']['results'])
    generated = g_summary(summaries['generation']['results'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    compact_plot(single, read(OLD_NATIVE/'summary.json')['rows'], out)
    g_plots(generated, summaries['native17']['rows'], out)
    for case in read(ROOT/'continuous/protocol.json')['cases'][:4]:
        sid = case['sample']; source = source_pixels(case['source'])
        folder = ROOT/'generation'/sid; result = verify(folder)
        gpoint = result['points']['E8']
        anchors = [r for r in summaries['native17']['rows'] if r['sample'] == sid]
        near = min(anchors, key=lambda r:abs(np.log(r['bpp']/gpoint['bpp'])))
        panels = [('Source', source[8])]
        with np.load(ROOT/'continuous'/case['name']/'E0/pixels.npz', allow_pickle=False) as f:
            panels.append(('B only', f['reconstruction'][8].copy()))
        with np.load(ROOT/'native17'/sid/f'q{near["qp"]}'/'fresh/pixels.npz', allow_pickle=False) as f:
            panels.append((f'UF q{near["qp"]}: {near["bpp"]:.4f} bpp', f['reconstruction'][8].copy()))
        for stem, key, title in [('E8', 'enhanced', 'E8 only'), ('E0', 'reconstruction', 'B + G4'), ('E8', 'reconstruction', 'E8 + G4')]:
            point = verify(folder/stem)
            path = folder/stem/'pixels.npz'; bindings[str(path)] = file_hash(path)
            with np.load(path, allow_pickle=False) as f:
                panels.append((title, f[key][8].copy()))
        comparison_image(panels, out/f'fixed_{sid}.png',
            f'{sid} | frame 8 | E8+G4={gpoint["bpp"]:.5f} bpp | nearest measured UF, NOT exactly matched rate')
    # Reference reset at the end of frame32; chunk beginning33 must use B only.
    case = next(c for c in read(ROOT/'continuous/protocol.json')['cases'] if c['frames'] == 41)
    source = source_pixels(case['source']); folder = ROOT/'continuous'/case['name']
    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    for stem in ('E0', 'E4', 'E8', 'E16'):
        path = folder/stem/'pixels.npz'; verify(folder/stem); bindings[str(path)] = file_hash(path)
        with np.load(path, allow_pickle=False) as f:
            rgb = f['reconstruction']
        mse = ((source.astype(np.float32)-rgb.astype(np.float32))**2).mean((1, 2, 3))
        ax.plot(np.arange(1, 41), 10*np.log10(255**2/np.maximum(mse[1:], 1e-8)), label=stem)
    ax.axvline(33, color='black', linestyle='--', label='First P8 after reference reset')
    ax.set_xlabel('Frame index'); ax.set_ylabel('Per-frame PSNR (dB)'); ax.legend(ncol=5, fontsize=8); ax.grid(alpha=.2)
    fig.suptitle('41 consecutive REDS frames; fixed width3, B-only temporal reference')
    fig.savefig(out/'continuous_quality.png', dpi=150); plt.close(fig)
    # A compact machine-readable ledger. No duplicate prose report in Git.
    atomic_json(out/'summary.json', dict(single=single, generation=generated,
        stages={s:read(ROOT/s/'complete.json') for s in required},
        continuous=[dict(sample=r['sample'], frames=r['frames'], native=r['native'],
            points={n:{k:v for k,v in p.items() if k != 'artifacts'} for n,p in r['points'].items()})
            for r in summaries['continuous']['results']]))
    images = sorted(p.name for p in out.glob('*.png'))
    atomic_json(out/'complete.json', dict(complete=True, inputs=bindings, figures=images,
        artifacts={n:file_hash(out/n) for n in images+['summary.json']}))
    print('Followup figures and audit complete', flush=True)


if __name__ == '__main__':
    main()
