"""Read-only generated-output zoom and crossed-model visuals from measured points."""
import argparse
from pathlib import Path
from statistics import mean
import sys

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from demo.chunk_enhancement_experiment import read
from demo.online_eg_eval_core import CLIPS, METRICS, OLD
from demo.online_eg_evaluate import DEFAULT, grouped
from demo.patch_prefix_probe import load_frames, verify_artifacts
from demo.scalable_codec import atomic_json, file_hash
from demo.scalable_experiment import load_source
from demo.scalable_format import frame_hash

COMBINATIONS = {
    'fixed_G_full': 'Old E + control G',
    'cross_oldE_newG': 'Old E + joint G',
    'cross_newE_fixedG': 'Joint E + control G',
    'joint_G_full': 'Joint E + joint G',
}


def crossed_summary(results):
    lookup = grouped(results)
    domains = {}
    for domain in ('all', 'REDS', 'UVG'):
        domains[domain] = {}
        for name in COMBINATIONS:
            rows = [lookup[sid, name] for sid in CLIPS
                    if domain == 'all' or lookup[sid, name]['dataset'] == domain]
            assert len(rows) == (4 if domain == 'all' else 2)
            domains[domain][name] = dict(bytes=mean(r['bytes'] for r in rows),
                **{scope: {m: mean(r[scope][m] for r in rows) for m in METRICS}
                   for scope in ('quality', 'roi_quality')})
    return domains


def report(output):
    assert read(output/'run.complete.json')['complete']
    assert read(output/'evaluation.resume_audit.json')['complete']
    root = output/'evaluation'
    summary = read(root/'summary.json')
    assert summary['complete']
    lookup = grouped(summary['results'])
    refs = {r['sample']['sample_id']: r for r in read(OLD/'summary.json')['results']}
    for row in summary['results']:
        folder = root/row['sample_id']/row['name']
        verify_artifacts(folder, row['artifacts'])
        assert read(folder/'result.json') == row
    dest = root/'cross_report'
    dest.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, (sid, label) in zip(axes.flat, CLIPS.items()):
        for arm, color, name in [('fixed', 'tab:blue', 'Old E + control G'),
                                 ('joint', 'tab:orange', 'Joint E + joint G')]:
            rows = [lookup[sid, f'{arm}_G_{case}']
                    for case in ('full_q2', 'full', 'full_q05')]
            rows.sort(key=lambda r: r['bytes'])
            ax.plot([r['bytes'] for r in rows], [r['roi_quality']['lpips_alex'] for r in rows],
                    'o-', color=color, label=f'{name}: q2 / 1 / 0.5')
        for key, color, marker in [('cross_oldE_newG', 'tab:red', 's'),
                                    ('cross_newE_fixedG', 'tab:green', 'D')]:
            r = lookup[sid, key]
            ax.scatter(r['bytes'], r['roi_quality']['lpips_alex'], color=color,
                       marker=marker, s=60, label=COMBINATIONS[key]+' (q1 only)')
        ax.set(title=label, xlabel='Actual complete stream bytes', ylabel='Local LPIPS (lower better)')
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)
    fig.suptitle('Generated outputs only: zoomed axes; crossed combinations are single measured points')
    fig.savefig(dest/'generated_rd_zoom.png', dpi=150)
    plt.close(fig)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
    for sid, label in CLIPS.items():
        source = load_source(refs[sid]['sample'])
        assert frame_hash(source) == refs[sid]['source_hash']
        videos = {'GT': source}
        for key, name in COMBINATIONS.items():
            r = lookup[sid, key]
            pixels = load_frames(Path(r['output_path']))
            assert frame_hash(pixels) == r['fresh_decode']['output_hash']
            videos[name] = pixels
        regions = refs[sid]['metric_regions']
        width = max(r[4] for r in regions)*2
        canvas = Image.new('RGB', (len(videos)*width, sum(r[5]*2+52 for r in regions)), 'white')
        draw = ImageDraw.Draw(canvas)
        top = 0
        for i, (t, n, x, y, w, h) in enumerate(regions):
            assert t <= 8 < t+n, 'fixed ninth frame must lie in the measured interval'
            for j, (name, pixels) in enumerate(videos.items()):
                crop = Image.fromarray(pixels[8, y:y+h, x:x+w]).resize((w*2, h*2), Image.Resampling.NEAREST)
                canvas.paste(crop, (j*width, top+52))
                draw.text((j*width+3, top+5), name, font=font, fill='black')
                draw.text((j*width+3, top+26), f'{label} ROI {i+1} | frame 9', font=font, fill='black')
            top += h*2+52
        canvas.save(dest/f'{sid}_crossed.png')
    atomic_json(dest/'summary.json', dict(complete=True, source_summary_sha256=file_hash(root/'summary.json'),
        source_code_sha256=file_hash(Path(__file__)), means=crossed_summary(summary['results']),
        figures={p.name: file_hash(p) for p in sorted(dest.glob('*.png'))},
        note='Read-only reuse, no new inference. Crossed models measured at full q1 only, not matched-rate curves.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=DEFAULT)
    report(p.parse_args().output)
