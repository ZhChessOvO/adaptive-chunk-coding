"""CPU-only native RD anchors and temporal B-reference diagnostics.

Uses saved fresh-decoder outputs only. No model inference or configuration choice.
Completion re-entry verifies all bound inputs/artifacts and never redraws them.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from demo.scalable_codec import atomic_json, file_hash
from tools.latent_probe import read, SAMPLES
from tools.latent_stream_probe import DEFAULT as STAGE_B, verify
from tools.latent_native_baseline import DEFAULT as ANCHORS, QPS
from tools.plot_latent_probe import collect, aggregate

CHAIN = STAGE_B.parent / 'stage_c'
DEFAULT = STAGE_B.parent / 'diagnostics'


def native_means(rows):
    result = {}
    for group, prefix in (('REDS', 'reds'), ('UVG crop', 'uvg')):
        result[group] = []
        for qp in QPS:
            subset = [r for r in rows if r['sample'].startswith(prefix) and r['qp'] == qp]
            if not subset:
                raise ValueError('missing native QP/sample group')
            if len({r['sample'] for r in subset}) != len(subset):
                raise ValueError('duplicate native point')
            result[group].append(dict(qp=qp, count=len(subset),
                bpp=float(np.mean([r['bpp'] for r in subset])),
                **{k: float(np.mean([r['quality_all9'][k] for r in subset]))
                   for k in ('psnr_db', 'lpips_alex')}))
    return result


def interpolate_native(points, bpp, metric):
    """Log-rate guide only: reject extrapolation and ambiguous equal rates."""
    ordered = sorted(points, key=lambda r: r['bpp'])
    rates = np.array([r['bpp'] for r in ordered])
    if bpp <= 0 or np.any(rates <= 0) or np.any(np.diff(rates) <= 0):
        raise ValueError('invalid/duplicate rate')
    if bpp < rates[0] or bpp > rates[-1]:
        return None
    return float(np.interp(np.log(bpp), np.log(rates), [r[metric] for r in ordered]))


def verified_inputs():
    bindings = {}

    def bind(path):
        bindings[str(path)] = file_hash(path)

    anchors = verify(ANCHORS)
    rows = read(ANCHORS / 'summary.json')['rows']
    if len(rows) != len(SAMPLES) * len(QPS) or anchors['points'] != len(rows):
        raise ValueError('incomplete native anchors')
    for row in rows:
        folder = ANCHORS / row['sample'] / f'q{row["qp"]}'
        verify(folder)
        verify(folder / 'fresh')
        if row['actual_bytes'] != (folder / 'native.bin').stat().st_size:
            raise ValueError('native byte mismatch')
        bind(folder / 'complete.json')
        bind(folder / 'fresh/pixels.npz')
    new_rows, native, stage_bindings = collect(STAGE_B)
    bindings.update(stage_bindings)
    for row in new_rows:
        bind(STAGE_B / row['sample'] / f'w{row["width"]}_{row["level"]}/pixels.npz')
    chains = read(CHAIN / 'summary.json')['results']
    verify(CHAIN)
    for case in chains:
        folder = CHAIN / f'{case["sample"]}_n{case["frames"]}'
        verify(folder)
        for name in case['arrivals']:
            verify(folder / name)
            bind(folder / name / 'complete.json')
        bind(folder / 'expected.npz')
        bind(folder / 'B/pixels.npz')
        bind(folder / 'all_E/pixels.npz')
    for stage in (STAGE_B, ANCHORS, CHAIN):
        bind(stage / 'complete.json')
        bind(stage / 'protocol.json')
        bind(stage / 'summary.json')
    for item in read(STAGE_B / 'protocol.json')['inputs'].values():
        path = Path(item['source_path'])
        if file_hash(path) != item['source_sha256']:
            raise ValueError('report source changed')
        bind(path)
    bind(Path(__file__))
    return rows, aggregate(new_rows, native), chains, bindings


def plot_rates(native, layered, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    diagnostics = {}
    for i, (group, points) in enumerate(native.items()):
        diagnostics[group] = {}
        for j, (key, ylabel) in enumerate((('psnr_db', 'PSNR (dB, higher better)'),
                                         ('lpips_alex', 'LPIPS (lower better)'))):
            ax = axes[i, j]
            ax.plot([v['bpp'] for v in points], [v[key] for v in points], 'k.-',
                    label='Native UF scalar P-QP sweep (I=32)')
            for point in points:
                ax.annotate(str(point['qp']), (point['bpp'], point[key]),
                            xytext=(3, 5), textcoords='offset points', fontsize=7)
            for width, color in ((3, '#2675b6'), (9, '#d07824'), (17, '#8e4bb2')):
                states = layered[group][str(width)]
                ax.scatter([states['B']['bpp']], [states['B'][key]], marker='o',
                           color=color, label=f'B, width {width}')
                ax.scatter([states['BE']['bpp']], [states['BE'][key]], marker='^',
                           facecolors='none', edgecolors=color, label=f'B+E, width {width}')
                guide = interpolate_native(points, states['B']['bpp'], key)
                diagnostics[group].setdefault(str(width), {})[key] = dict(
                    B=states['B'][key], native_lograte_interpolation=guide,
                    B_minus_interpolation=None if guide is None else states['B'][key]-guide)
            ax.set_xlabel('Actual file bpp (all 9 frames)')
            ax.set_ylabel(ylabel)
            ax.set_title(group)
            ax.grid(alpha=.25)
            ax.legend(fontsize=7)
    fig.suptitle('Native UF versus the coarse prefix / complete endpoint\n'
                 'G off; four diagnostic windows; no claim of full-video BD-rate', fontsize=12)
    fig.savefig(out / 'native_rd.png', dpi=150)
    plt.close(fig)
    return diagnostics


def comparison_image(panels, path, title):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    h, w = panels[0][1].shape[:2]
    pw, ph = 384, round(384*h/w)
    canvas = Image.new('RGB', (pw*len(panels), ph*2+96), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 4), title, font=font, fill='black')
    for i, (label, array) in enumerate(panels):
        draw.text((i*pw+5, 31), label, font=font, fill='black')
        im = Image.fromarray(array)
        canvas.paste(im.resize((pw, ph)), (i*pw, 56))
        cw, ch = w//3, h//3
        x, y = (w-cw)//2, (h-ch)//2
        canvas.paste(im.crop((x, y, x+cw, y+ch)).resize((pw, ph)), (i*pw, ph+80))
    canvas.save(path)


def pictures(rows, out):
    names = []
    protocol = read(STAGE_B / 'protocol.json')
    for sid in SAMPLES:
        base_row = next(r for r in read(STAGE_B / sid / 'complete.json')['rows']
                        if r['width'] == 3 and r['level'] == 'B')
        candidates = [r for r in rows if r['sample'] == sid]
        near = min(candidates, key=lambda r: abs(math.log(r['bpp']/base_row['bpp'])))
        with np.load(protocol['inputs'][sid]['source_path'], allow_pickle=False) as f:
            original = f['source'][5].copy()
        panels = [('Source', original)]
        for label, path in (
            (f'B w3: {base_row["bpp"]:.5f} bpp', STAGE_B / sid / 'w3_B/pixels.npz'),
            (f'UF P{near["qp"]}: {near["bpp"]:.5f} bpp',
             ANCHORS / sid / f'q{near["qp"]}/fresh/pixels.npz'),
            ('B+E = native same-context', STAGE_B / sid / 'w3_BE/pixels.npz')):
            with np.load(path, allow_pickle=False) as f:
                panels.append((label, f['reconstruction'][5].copy()))
        name = f'native_compare_{sid}.png'
        comparison_image(panels, out / name,
            f'{sid} | frame 5 | nearest measured UF rate, NOT exact matched bpp')
        names.append(name)
    return names


def chain_video(case, out):
    """Diagnostic 10-FPS playback; raw arrays are metrics, MP4 is visual only."""
    from PIL import Image, ImageDraw, ImageFont
    sid, count = case['sample'], case['frames']
    folder = CHAIN / f'{sid}_n{count}'
    path = read(CHAIN / 'protocol.json')['inputs'][sid]['source_path']
    with np.load(path, allow_pickle=False) as f:
        source = f['source'][:count].copy()
    with np.load(folder / 'B/pixels.npz', allow_pickle=False) as f:
        base = f['reconstruction'].copy()
    with np.load(folder / 'all_E/pixels.npz', allow_pickle=False) as f:
        full = f['reconstruction'].copy()
    with np.load(folder / 'expected.npz', allow_pickle=False) as f:
        own = f['native_own'].copy()
    pw, ph = 384, round(384*source.shape[1]/source.shape[2])
    height = ph + 64 + (ph % 2)
    name = f'chain_{sid}_n{count}.mp4'
    command = ['ffmpeg', '-nostdin', '-y', '-hide_banner', '-loglevel', 'error',
               '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{pw*4}x{height}',
               '-r', '10', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '18',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / name)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
    try:
        for index in range(count):
            canvas = Image.new('RGB', (pw*4, height), 'white')
            draw = ImageDraw.Draw(canvas)
            draw.text((5, 5), f'{sid} | frame {index}/{count-1} | diagnostic playback 10 FPS',
                      fill='black', font=font)
            for i, (label, pixels) in enumerate((('Source', source), ('B only', base),
                    ('B+E; B reference', full), ('Native UF; own full reference', own))):
                draw.text((i*pw+5, 31), label, fill='black', font=font)
                canvas.paste(Image.fromarray(pixels[index]).resize((pw, ph)), (i*pw, 60))
            process.stdin.write(canvas.tobytes())
        process.stdin.close()
        error = process.stderr.read().decode()
        if process.wait(timeout=60):
            raise RuntimeError(error)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    return name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    if not os.environ.get('TMUX'):
        raise RuntimeError('tmux required')
    if args.wait:
        began = time.monotonic()
        while not all((p / 'complete.json').exists() for p in (STAGE_B, ANCHORS, CHAIN)):
            if time.monotonic()-began > 4*3600:
                raise TimeoutError('diagnostic evidence incomplete')
            time.sleep(15)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'complete.json').exists():
        receipt = verify(out)
        for name, expected in receipt['inputs'].items():
            if file_hash(Path(name)) != expected:
                raise ValueError('completed diagnostic input changed')
        print('Verified diagnostic report without redrawing', flush=True)
        return
    rows, layered, chains, bindings = verified_inputs()
    native = native_means(rows)
    diagnostic = plot_rates(native, layered, out)
    images = pictures(rows, out)
    videos = [chain_video(c, out) for c in chains if c['frames'] == 17]
    atomic_json(out / 'summary.json', dict(native=native, layered=layered,
        interpolation_scope='linear quality versus log-rate, group means; not measured same-bpp or BD-rate',
        coarse_vs_native=diagnostic, continuous=chains))
    names = ['native_rd.png', 'summary.json'] + images + videos
    atomic_json(out / 'complete.json', dict(complete=True, inputs=bindings,
        artifacts={name: file_hash(out / name) for name in names}, images=images, videos=videos))
    print('Created verified native/continuous diagnostics', flush=True)


if __name__ == '__main__':
    main()
