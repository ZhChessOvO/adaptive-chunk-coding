"""CPU-only audit and figures for the completed paired visual Router evaluation.

Never changes the formal evaluation, its timing, models, streams or metrics.
Optional enhanced-image scoring is a separately labelled saved-pixel ablation,
not a new decoder invocation or a newly compressed E-only stream.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch

from demo.routervc_fullview_probe import read, digest, save, immutable, verify_artifacts

ROOT = Path('/root/autodl-fs/DCVC/runs/routervc_revision_20261003')
GROUPS = ('REDS_fullview', 'UVG_crop')
METRICS = ('lpips_alex', 'psnr_db', 'temporal_delta_mae')
CURVES = (
    ('uf', 'Native DCVC-UF', '#242424', 'o', '-'),
    ('global_local_g4', 'Global + local / G4', '#2166ac', 'o', '--'),
    ('global_local_g8', 'Global + local / G8', '#2166ac', 'o', '-'),
    ('local_g4', 'Local only / G4', '#db8425', '^', '--'),
    ('local_g8', 'Local only / G8', '#db8425', '^', '-'),
    ('wholeframe_g_one_roi', 'Base + whole-frame G', '#27854c', '*', 'None'),
)


def curve_name(name):
    if name.startswith('uf_qp'):
        return 'uf'
    if name.startswith(('global_local_e', 'local_e')):
        arm, _, budget = name.partition('_e')
        return arm + '_' + budget.split('_')[1]
    return name


def mean(rows, keys):
    if not rows:
        raise ValueError('empty aggregation')
    return {k: sum(r[k] for r in rows) / len(rows) for k in keys}


def interpolate_uf(rows, rate, metric):
    """Piecewise linear quality versus log(bpp), within measured support ONLY."""
    rows = sorted(rows, key=lambda r: r['bpp'])
    if len(rows) < 2 or any(r['bpp'] <= 0 for r in rows):
        raise ValueError('at least two positive measured UF rates required')
    if any(a['bpp'] >= b['bpp'] for a, b in zip(rows, rows[1:])):
        raise ValueError('duplicate/non-increasing UF rate')
    if not math.isfinite(rate) or rate < rows[0]['bpp'] or rate > rows[-1]['bpp']:
        return None
    for row in rows:
        if rate == row['bpp']:
            return row[metric]
    for a, b in zip(rows, rows[1:]):
        if a['bpp'] < rate < b['bpp']:
            weight = math.log(rate/a['bpp']) / math.log(b['bpp']/a['bpp'])
            return a[metric] + weight*(b[metric]-a[metric])
    raise AssertionError('in-range rate was not bracketed')


def comparisons(rows):
    paired, matched = [], []
    index = {(r['sample_id'], r['point']): r for r in rows}
    if len(index) != len(rows):
        raise ValueError('duplicate evaluation point')
    for row in rows:
        sid, point = row['sample_id'], row['point']
        if point.startswith('global_local_'):
            local = index[(sid, point.replace('global_local_', 'local_', 1))]
            paired.append(dict(sample_id=sid, group=row['group'], point=point,
                delta={k: row[k]-local[k] for k in ('bpp', *METRICS)},
                scope='same sample and nominal budgets, not exactly equal bytes'))
        if not point.startswith(('global_local_', 'local_')):
            continue
        uf = [r for r in rows if r['sample_id'] == sid and r['point'].startswith('uf_qp')]
        ref = {k: interpolate_uf(uf, row['bpp'], k) for k in METRICS}
        matched.append(dict(sample_id=sid, group=row['group'], point=point, bpp=row['bpp'],
            covered=ref['lpips_alex'] is not None, uf_interpolated=ref,
            delta={k: row[k]-ref[k] if ref[k] is not None else None for k in METRICS},
            measured_UF_lpips_dominators=[r['point'] for r in uf
                if r['bpp'] <= row['bpp'] and r['lpips_alex'] <= row['lpips_alex']]))
    result = {}
    for group in GROUPS:
        pair = [p for p in paired if p['group'] == group]
        match = [p for p in matched if p['group'] == group]
        covered = [p for p in match if p['covered']]
        result[group] = dict(context_pairs=len(pair),
            context_lower_lpips=sum(p['delta']['lpips_alex'] < 0 for p in pair),
            mean_global_minus_local=mean([p['delta'] for p in pair], ('bpp', *METRICS)),
            router_points=len(match), within_measured_UF_support=len(covered),
            router_lower_lpips_within_support=sum(p['delta']['lpips_alex'] < 0 for p in covered),
            mean_router_minus_interpolated_UF=mean([p['delta'] for p in covered], METRICS) if covered else None,
            measured_UF_lpips_dominates=sum(bool(p['measured_UF_lpips_dominators']) for p in match))
    return dict(groups=result, global_local_pairs=paired, matched_UF=matched,
        interpolation='linear quality in log(bpp); within EACH sample support; no extrapolation or BD-rate',
        dependence='operating points share videos; counts are not independent trials')


def snapshot(root):
    paths = [p for p in root.rglob('*') if p.is_file() and p.suffix in ('.json', '.jsonl', '.csv')]
    return {str(p.relative_to(root)): dict(sha256=digest(p), mtime_ns=p.stat().st_mtime_ns) for p in paths}


def audit(root, run):
    from demo import routervc_visual_evaluate as evaluation
    before = snapshot(root)
    protocol = read(root/'protocol.json')
    if protocol['code'] != evaluation.code_hashes():
        raise ValueError('formal source pins changed')
    models = evaluation.completed_models(ROOT/'visual_router')
    if models != protocol['models']:
        raise ValueError('formal model identity changed')
    # Redirect only progress; all formal point/pixel/prefix validation stays real.
    proxy = SimpleNamespace(root=root, check=run.check, update=run.update)
    forbidden = AssertionError('completed audit may not infer, rescore or rewrite formal results')
    with ExitStack() as stack:
        for name in ('demo.conditioned_generation_pipeline.execute', 'demo.scalable_experiment.quality',
                     'demo.routervc_visual_evaluate.finish'):
            stack.enter_context(patch(name, side_effect=forbidden))
        summary = evaluation.evaluate(proxy, protocol, verify_only=True)
    if before != snapshot(root):
        raise ValueError('formal JSON/timing/heartbeat changed during read-only audit')
    save(run.root/'audit.json', dict(complete=True, points=169, extra_smoke_decodes=2,
        literal_prefix_pairs=52, source_free=True, no_inference=True, no_metric_recomputation=True,
        original_records_and_timing_unchanged=True, formal_snapshot=before))
    return summary


def save_plot(figure, path):
    import matplotlib.pyplot as plt
    temporary = path.with_suffix('.tmp')
    figure.savefig(temporary, format='png', dpi=160, facecolor='white')
    plt.close(figure)
    os.replace(temporary, path)


def plot_curves(axis, rows, metric, *, labels=False):
    for curve, title, color, marker, style in CURVES:
        points = sorted((r for r in rows if curve_name(r['point']) == curve), key=lambda r:r['bpp'])
        axis.plot([r['bpp'] for r in points], [r[metric] for r in points], color=color,
            marker=marker, linestyle=style, markersize=10 if marker == '*' else 5,
            label=title if labels else None)
    axis.set_xlabel('Actual stream bits / pixel')
    axis.grid(alpha=.2)


def figures(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9))
    for i, group in enumerate(GROUPS):
        rows = [dict(point=name, **value) for name, value in summary['group_means'][group].items()]
        for j, metric in enumerate(METRICS[:2]):
            plot_curves(axes[i,j], rows, metric, labels=i == 0 and j == 0)
            axes[i,j].set_title(group.replace('_', ' ') + (' (6 windows)' if i == 0 else ' (7 windows)'))
            axes[i,j].set_ylabel('LPIPS (lower is better)' if j == 0 else 'PSNR dB (higher is better)')
    fig.suptitle('RouterVC visual Router: measured rate / quality\nREDS complete view resized to 1024 x 576; UVG existing 512 x 512 crops')
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(.5,.045), frameon=False)
    fig.text(.5,.015,'Equal-window means within each dataset. Lines join measured points; no extrapolation or model promotion.',ha='center',fontsize=9)
    fig.subplots_adjust(bottom=.19, top=.89, hspace=.38, wspace=.23)
    save_plot(fig, output/'rd_dataset_means.png')
    for group in GROUPS:
        ids = list(dict.fromkeys(r['sample_id'] for r in summary['rows'] if r['group'] == group))
        fig, axes = plt.subplots(2, 3 if len(ids) == 6 else 4, figsize=(15,7.7))
        for i, (axis, sid) in enumerate(zip(axes.flat, ids)):
            rows = [r for r in summary['rows'] if r['sample_id'] == sid]
            plot_curves(axis, rows, 'lpips_alex', labels=i == 0)
            axis.set_title(rows[0]['sequence']); axis.set_ylabel('LPIPS')
        for axis in list(axes.flat)[len(ids):]:
            axis.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(.5,.01), frameon=False)
        fig.suptitle(group.replace('_',' ')+' | Per-video LPIPS curves | 17 frames each')
        fig.subplots_adjust(bottom=.18, top=.9, wspace=.3, hspace=.37)
        save_plot(fig, output/(group+'_rd.png'))


def resources_digest(root):
    import json
    events = [json.loads(line) for line in (root/'heartbeat.jsonl').read_text().splitlines() if line]
    peak = max(int(e['gpu'].split(',')[2].strip()) for e in events if e.get('gpu'))
    stamp = (root/'complete.json').stat().st_mtime
    return dict(formal_complete_utc=datetime.fromtimestamp(stamp, timezone.utc).isoformat(),
        formal_elapsed_seconds=max(e['elapsed_seconds'] for e in events),
        sampled_peak_gpu_mib=peak, sampling_interval_seconds=30,
        output_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),
        final_disks=events[-1]['disks'], original_heartbeat_unchanged=True)


def score_enhanced(root, summary, run):
    """G-off SAME-WIRE pixels; do not invent a cheaper independent E-only rate."""
    import numpy as np
    import torch
    from demo.scalable_experiment import quality
    from demo.stage_c_three_path_roi_probe import LPIPSAlex
    from demo.scalable_format import frame_hash
    torch.set_num_threads(4)
    metric, scored, cached_source, source = None, [], None, None
    rows = [r for r in summary['rows'] if r['point'].startswith(('global_local_', 'local_'))]
    target = run.root/'saved_E_scores'; target.mkdir(exist_ok=True)
    for i, row in enumerate(rows):
        run.check()
        sample = root/'samples'/row['sample_id']; folder = sample/row['point']
        point = read(folder/'result.json')
        binding = dict(result_sha256=digest(folder/'result.json'),
                       source_sha256=digest(sample/'source.npz'),
                       source_rgb_sha256=read(sample/'source.complete.json')['rgb_sha256'],
                       received_rgb_sha256=point['decode']['generation_input_hash'])
        path = target/(row['sample_id']+'__'+row['point']+'.json')
        if path.exists():
            saved = read(path)
            if saved['binding'] != binding:
                raise ValueError('saved E-only score binding changed')
        else:
            if cached_source != row['sample_id']:
                with np.load(sample/'source.npz', allow_pickle=False) as z:
                    source = z['source'].copy()
                if frame_hash(source) != binding['source_rgb_sha256']:
                    raise ValueError('source pixels changed')
                cached_source = row['sample_id']
            if digest(folder/'fresh/reconstruction.npz') != point['artifacts']['fresh/reconstruction.npz']:
                raise ValueError('saved reconstruction changed')
            with np.load(folder/'fresh/reconstruction.npz', allow_pickle=False) as z:
                enhanced = z['enhanced'].copy()
            if frame_hash(enhanced) != binding['received_rgb_sha256']:
                raise ValueError('saved received-E pixels changed')
            if metric is None:
                metric = LPIPSAlex(True)
            measured = quality(source, enhanced, metric)
            saved = dict(binding=binding, sample_id=row['sample_id'], group=row['group'], point=row['point'],
                bytes=row['bytes'], bpp=row['bpp'], quality=measured,
                with_G_minus_without_G={k:row[k]-measured[k] for k in METRICS},
                scope='score already fresh-decoded enhanced RGB; same charged RouterVC stream; no new decode or latency')
            save(path,saved)
        scored.append(saved)
        run.update(phase='CPU_saved_E_metrics',completed=i+1,total=len(rows))
    groups = {}
    for group in GROUPS:
        groups[group] = {}
        for name in dict.fromkeys(r['point'] for r in scored):
            selected = [r for r in scored if r['group'] == group and r['point'] == name]
            groups[group][name] = dict(windows=len(selected),
                E_only_quality=mean([r['quality'] for r in selected], METRICS),
                G_increment=mean([r['with_G_minus_without_G'] for r in selected], METRICS),
                G_improves_lpips=sum(r['with_G_minus_without_G']['lpips_alex']<0 for r in selected))
    save(run.root/'saved_E_summary.json',dict(complete=True,points=len(scored),groups=groups,
        records=scored, new_decodes=0, transmitted_bytes_unchanged=True, metrics_recomputed_for_formal_output=False))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT/'visual_evaluation_recovered')
    parser.add_argument('--output',type=Path,default=ROOT/'visual_evaluation_analysis')
    parser.add_argument('--score-enhanced',action='store_true')
    args=parser.parse_args(argv)
    if not os.environ.get('TMUX'):
        raise RuntimeError('analysis and scoring require tmux')
    if args.output.resolve().is_relative_to(args.root.resolve()):
        raise ValueError('analysis must not write into the formal tree')
    from demo.chunk_enhancement_experiment import Run
    run=Run(SimpleNamespace(output=args.output,command='visual_analysis',max_hours=4))
    run.thread.start()
    try:
        immutable(run.root/'request.json',dict(root=str(args.root.resolve()),
            formal_complete_sha256=digest(args.root/'complete.json'),code_sha256=digest(__file__)))
        done=run.root/'complete.json'
        if done.exists():
            verify_artifacts(run.root,read(done)['artifacts'])
            prior=read(run.root/'audit.json')['formal_snapshot']
            if snapshot(args.root) != prior:
                raise ValueError('formal records changed since audit')
            summary=read(args.root/'summary.json')
        else:
            summary=audit(args.root,run)
            report=dict(comparisons=comparisons(summary['rows']),resources=resources_digest(args.root),
                group_means=summary['group_means'],no_model_promotion=True,semantic_fidelity_measured=False)
            save(run.root/'analysis.json',report)
            figures(summary,run.root)
            files=['audit.json','analysis.json','rd_dataset_means.png','REDS_fullview_rd.png','UVG_crop_rd.png']
            save(done,dict(complete=True,artifacts={n:digest(run.root/n) for n in files}))
        if args.score_enhanced:
            score_enhanced(args.root,summary,run)
        print('VISUAL_ANALYSIS_COMPLETE',flush=True)
    except BaseException as error:
        save(run.root/'last_failure.json',dict(error=repr(error),progress=run.progress)); raise
    finally:
        run.log_resources();run.stop.set();run.thread.join(timeout=3)


if __name__=='__main__':
    main()
