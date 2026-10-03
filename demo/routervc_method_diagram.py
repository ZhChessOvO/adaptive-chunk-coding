"""Draw the current RouterVC implementation, without inference or source images.

Run inside CPU tmux. The SVG is native vector artwork, not an edited image.
Existing complete outputs are hash-checked and reused without changing timing.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
DEFAULT = Path('/root/autodl-fs/DCVC/runs/routervc_20261003/method_figure')
CODE = ('routervc_method_diagram.py', 'routervc_encode.py', 'routervc_decode.py',
        'routervc_policy.py', 'routervc_format.py', 'compact_enhancement_format.py')
INK, MUTED = '#203449', '#55677b'
UF, E, R, G = '#dcecff', '#d8f2e9', '#fff0cc', '#eee1fa'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, data):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def resources():
    return {str(path): dict(zip(('total', 'used', 'free'), shutil.disk_usage(path)))
            for path in map(Path, ('/root', '/root/autodl-tmp', '/root/autodl-fs'))}


def draw():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                         'svg.hashsalt': 'routervc-v1-method-20261003'})
    fig, ax = plt.subplots(figsize=(24, 14.4), dpi=150)
    fig.patch.set_facecolor('#ffffff')
    ax.set(xlim=(0, 24), ylim=(0, 14.4))
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    def text(x, y, value, size=13, *, color=INK, weight='normal', ha='center'):
        return ax.text(x, y, value, fontsize=size, color=color, fontweight=weight,
                       ha=ha, va='center', linespacing=1.45, zorder=5)

    def panel(x, y, w, h, color, edge='#d5dfe8', rounding=.13):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
            boxstyle=f'round,pad=0.025,rounding_size={rounding}',
            facecolor=color, edgecolor=edge, linewidth=1.2, zorder=1))

    def box(x, y, w, h, title, body, color):
        panel(x, y, w, h, color)
        text(x+w/2, y+h-.34, title, 14.7, weight='bold')
        text(x+w/2, y+h/2-.23, body, 12.8)

    def arrow(points, label=None, label_at=None, color=MUTED, dashed=False):
        for a, b in zip(points[:-2], points[1:-1]):
            ax.plot((a[0], b[0]), (a[1], b[1]), color=color, linewidth=1.6,
                    linestyle='--' if dashed else '-', zorder=2)
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle='-|>',
            mutation_scale=15, linewidth=1.6, color=color,
            linestyle='--' if dashed else '-', zorder=2))
        if label:
            text(*label_at, label, 11.5, color=color)

    text(.65, 13.82, 'RouterVC: one base, two optional ways to improve it',
         27, weight='bold', ha='left')
    text(.68, 13.19, 'Uniform low-rate UF base + genuine enhancement packets + local generation',
         15.5, color=MUTED, ha='left')

    panel(.4, 9.04, 23.2, 3.68, '#f7fafd')
    text(.8, 12.39, 'SENDER', 15, weight='bold', ha='left')
    text(5.5, 12.39, 'Source X is available here; never at the receiver.',
         12.5, color=MUTED, ha='left')
    box(.75, 10.08, 1.72, 1.75, 'Source', 'Original\nvideo X', '#e9edf2')
    box(3.12, 10.08, 3.16, 1.75, 'Frozen DCVC-UF',
        'Uniform QP8 base\nLocal decode: B, F', UF)
    box(6.99, 10.08, 3.56, 1.75, 'Encode ALL 16 E candidates',
        'UF-conditioned q1 bundles\nInputs: X, B, F', E)
    box(11.24, 10.08, 3.47, 1.75, 'Decode actual E packets',
        'Candidate RGB: Y_i\nExact bundle cost: c_i', E)
    box(15.4, 10.08, 3.36, 1.75, 'Shared utility Router',
        'Inputs: B, Y_i, coverage\nOne backbone; no oracle', R)
    box(19.46, 10.08, 3.51, 1.75, 'Fixed greedy order',
        'Rank predicted gain / real byte\nKeep whole-bundle prefix\nSum of c_i <= E budget', R)
    for x1, x2 in ((2.47, 3.12), (6.28, 6.99), (10.55, 11.24),
                   (14.71, 15.4), (18.76, 19.46)):
        arrow([(x1, 10.95), (x2, 10.95)])
    arrow([(1.61, 11.85), (1.61, 12.03), (8.77, 12.03), (8.77, 11.85)],
          color='#7b8b9c')
    text(7.17, 9.48, 'Candidate entropy encoding + decoding is real sender work, and its cost is recorded.',
         12.3, color=MUTED, ha='left')

    panel(3.13, 6.92, 17.98, 1.72, '#f0f3f7', edge='#a5b3c1')
    text(12.12, 8.32, 'ONE TRANSMITTED RTVC STREAM  /  EVERY BYTE COUNTS',
         14.5, weight='bold')
    entries = [('RTVC: 310 B', 'Shared policy + model hashes'),
               ('ACSE2: 190 B', 'Shared container framing'),
               ('UF base: |b0|', 'One native QP8 base stream'),
               ('E: sum of c_i', 'Selected, complete packet bundles')]
    for i, (title, body) in enumerate(entries):
        x = 5.37+i*4.5
        text(x, 7.83, title, 14, weight='bold')
        text(x, 7.40, body, 11.8)
        if i < 3:
            text(x+2.25, 7.70, '+', 18, color='#758799')
    arrow([(4.70, 10.06), (4.70, 8.66)], 'b0', (4.97, 9.42), color='#4c83b4')
    arrow([(21.22, 10.06), (21.22, 9.0), (19.44, 9.0), (19.44, 8.66)],
          color='#3b907b')

    panel(.4, 2.57, 23.2, 3.76, '#f7fafd')
    text(12, 5.88, 'RECEIVER  /  source-free reconstruction', 15, weight='bold')
    arrow([(4.70, 6.90), (4.70, 6.59), (2.42, 6.59), (2.42, 5.42)],
          color='#4c83b4')
    arrow([(19.44, 6.90), (19.44, 6.59), (6.35, 6.59), (6.35, 5.42)],
          color='#3b907b')
    text(14.25, 6.69, 'received E packets (coordinates imply coverage)',
         11.3, color='#307968')
    box(.82, 3.91, 3.20, 1.50, 'Decode UF base',
        'Reconstruct B and F\nUF reference stays unchanged', UF)
    box(4.75, 3.91, 3.20, 1.50, 'Decode received E',
        'B, F + E packets -> Y\nNo packet: keep base B', E)
    box(8.68, 3.91, 3.84, 1.50, 'Same shared Router',
        'Inputs: B, Y, coverage\nChoose at most K G regions\nOptional boundary penalty', R)
    box(13.24, 3.91, 3.73, 1.50, 'SeedVR2 BF16 + joint LoRA',
        'Generate selected ROIs\nRGB condition: actual mixed Y', G)
    box(17.69, 3.91, 3.09, 1.50, 'Fixed ROI feather',
        '16-px output blending\nUnselected pixels: Y', G)
    box(21.49, 3.91, 1.77, 1.50, 'Output', 'Displayed\nvideo', '#e9edf2')
    for x1, x2 in ((4.02, 4.75), (7.95, 8.68), (12.52, 13.24),
                   (16.97, 17.69), (20.78, 21.49)):
        arrow([(x1, 4.66), (x2, 4.66)])
    arrow([(6.35, 3.89), (6.35, 3.35), (15.11, 3.35), (15.11, 3.89)],
          'RGB condition Y', (10.7, 3.12), color='#8763a8', dashed=True)
    text(1.0, 2.88, 'E and G improve the displayed video only. Neither writes back into the UF temporal reference.',
         12.1, color=MUTED, ha='left')

    panel(.65, .42, 10.75, 1.77, '#f9fbfd')
    text(2.00, 1.83, 'Four states', 13, weight='bold')
    text(5.02, 1.83, 'No E packet', 12.5, weight='bold')
    text(8.74, 1.83, 'E received', 12.5, weight='bold')
    text(1.90, 1.23, 'G off', 12.5, weight='bold')
    text(1.90, .72, 'G on', 12.5, weight='bold')
    for x, y, value, color in ((5.02, 1.23, 'B: keep base', UF),
                               (8.74, 1.23, 'E: use Y', E),
                               (5.02, .72, 'G: generate from B', G),
                               (8.74, .72, 'EG: generate from Y', G)):
        panel(x-1.60, y-.20, 3.20, .40, color, rounding=.06)
        text(x, y, value, 11.9)
    text(12.0, 1.87, 'No per-region G map is transmitted.', 15, weight='bold', ha='left')
    text(12.0, 1.36, 'The receiver derives G from received content and the shared policy.',
         12.5, color=MUTED, ha='left')
    text(12.0, .91, 'Boundary penalty selects regions; feathering blends output edges.',
         12.5, color=MUTED, ha='left')
    text(12.0, .48, 'F conditions the E codec only. The current generator uses RGB, not extra features.',
         11.8, color=MUTED, ha='left')
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    args = parser.parse_args()
    if not os.environ.get('TMUX'):
        parser.error('run this CPU drawing task inside tmux')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error("set CUDA_VISIBLE_DEVICES='' for this CPU-only task")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    code_hashes = {str(REPO/'demo'/name): digest(REPO/'demo'/name) for name in CODE}
    manifest_path = output/'manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous['source_hashes'] != code_hashes:
            raise RuntimeError('diagram source changed; use a new output directory')
        for name, sha in previous['artifacts'].items():
            if digest(output/name) != sha:
                raise RuntimeError(f'changed diagram artifact: {name}')
        print(json.dumps(dict(complete=True, reused=True, manifest=str(manifest_path))))
        return
    began = time.monotonic()
    before = resources()
    fig = draw()
    artifacts = {}
    for extension in ('png', 'svg'):
        data = io.BytesIO()
        metadata = {'Date': None} if extension == 'svg' else None
        fig.savefig(data, format=extension, dpi=150, metadata=metadata)
        path = output/f'routervc_pipeline.{extension}'
        atomic(path, data.getvalue())
        artifacts[path.name] = digest(path)
    plt.close(fig)
    with Image.open(output/'routervc_pipeline.png') as im:
        dimensions = list(im.size)
    manifest = dict(complete=True, cpu_only=True, no_model_inference=True,
        purpose='Current implementation diagram, not experimental evidence or a quality metric',
        figure='RouterVC v1: RGB-only generation and whole-bundle E packet prefixes',
        schema=dict(rtvc_shared_header_bytes=310, acse2_shared_header_bytes=190,
                    native_uf_payload_and_all_selected_e_packet_bytes_charged=True),
        png_dimensions=dimensions, source_hashes=code_hashes, artifacts=artifacts,
        software=dict(matplotlib=matplotlib.__version__),
        wall_seconds=time.monotonic()-began, resources_before=before, resources_after=resources())
    atomic(manifest_path, (json.dumps(manifest, indent=2)+'\n').encode())
    print(json.dumps(dict(complete=True, manifest=str(manifest_path),
                         dimensions=dimensions, wall_seconds=manifest['wall_seconds'])))


if __name__ == '__main__':
    main()
