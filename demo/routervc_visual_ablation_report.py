"""Join completed diagnostic runs without inference or changing their artifacts.

The zero-E control keeps the G8 policy, not identical generated cells. Saved
G-off scores keep the charged stream unchanged, not a fictional cheaper codec.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from demo.routervc_fullview_probe import digest, read, save, verify_artifacts
from demo.routervc_visual_report import ROOT, GROUPS, METRICS, comparisons, mean, save_plot


def joined_rows(extended, zero):
    rows = list(extended) + [r for r in zero if r['point'].endswith('_e0_g8')]
    keys = [(r['sample_id'], r['point']) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate point in joined evidence')
    return rows


def paired_changes(rows):
    """Compare E additions within each clip, Router arm and G8 policy."""
    index = {(r['sample_id'], r['point']): r for r in rows}
    changes = []
    for row in rows:
        if not row['point'].endswith('_e0_g8'):
            continue
        arm = row['point'].removesuffix('_e0_g8')
        for left, right in (('0', '0.25'), ('0.25', '0.5')):
            a = index[(row['sample_id'], f'{arm}_e{left}_g8')]
            b = index[(row['sample_id'], f'{arm}_e{right}_g8')]
            changes.append(dict(sample_id=row['sample_id'], group=row['group'], arm=arm,
                step=f'{left}->{right}', delta={k:b[k]-a[k] for k in ('bpp', *METRICS)}))
    return changes


def aggregates(rows):
    return {group: {name: dict(windows=len(selected),
                **mean(selected, ('bpp', *METRICS, 'receiver_seconds')))
            for name in dict.fromkeys(r['point'] for r in rows)
            if (selected := [r for r in rows if r['group'] == group and r['point'] == name])}
        for group in GROUPS}


def completed(root):
    marker = read(root/'complete.json')
    if not marker.get('complete'):
        raise ValueError(f'incomplete run: {root}')
    verify_artifacts(root, marker['artifacts'])
    return read(root/'summary.json')


def processing_ratio(windows, source_pixels):
    """Repeated crop processing volume, not FLOPs or unique output coverage."""
    if source_pixels <= 0:
        raise ValueError('positive source volume required')
    return sum(math.prod(w['runtime']['processing_shape']) for w in windows) / source_pixels


def runtime_and_bytes(root, rows):
    records = []
    for row in rows:
        name = row['point']
        if name.endswith('_e0_g8'):
            folder = root/'visual_zero_E'
        elif name in ('uf_qp40', 'uf_qp48', 'uf_qp56'):
            folder = root/'visual_uf_rate_extension'
        else:
            folder = root/'visual_evaluation_recovered'
        path = folder/'samples'/row['sample_id']/name/'result.json'
        rec = read(path)
        verify_artifacts(path.parent, rec['artifacts'])
        if rec['bytes'] != row['bytes']:
            raise ValueError('joined byte count differs from bound result')
        d = rec['decode']; g = d.get('generation_runtime', {})
        records.append(dict(sample_id=row['sample_id'], group=row['group'], point=name,
            total_bytes=row['bytes'], native_base_bytes=rec['byte_ledger']['native_bytes'],
            additional_bytes=row['bytes']-rec['byte_ledger']['native_bytes'],
            fresh_seconds=d['seconds'], policy_seconds=d.get('policy_seconds', 0.),
            g_model_load_seconds=g.get('model_load_seconds', 0.),
            g_seconds_excluding_load=g.get('seconds_model_load_excluded', 0.),
            g_processing_volume_ratio=processing_ratio(g.get('windows', []),
                row['bytes']*8/row['bpp']),
            peak_cuda_gib=d['peak_cuda_allocated_bytes']/2**30,
            g_calls=d['route']['g_calls'],
            mask_bytes=sum(d.get(k, 0) for k in
                ('explicit_E_mask_bytes', 'explicit_G_map_bytes', 'protection_mask_bytes'))))
    keys = ('total_bytes', 'native_base_bytes', 'additional_bytes', 'fresh_seconds',
        'policy_seconds', 'g_model_load_seconds', 'g_seconds_excluding_load', 'peak_cuda_gib',
        'g_calls', 'mask_bytes', 'g_processing_volume_ratio')
    return {group: {name: mean(selected, keys)
            for name in dict.fromkeys(r['point'] for r in records)
            if (selected := [r for r in records if r['group'] == group and r['point'] == name])}
        for group in GROUPS}


def packet_costs(root, rows):
    from demo.routervc_byte_audit import inspect_stream, BYTE_KEYS
    records=[]
    for row in rows:
        name=row['point']
        if not name.startswith(('global_local_', 'local_')):
            continue
        folder='visual_zero_E' if name.endswith('_e0_g8') else 'visual_evaluation_recovered'
        wire=(root/folder/'samples'/row['sample_id']/name/'stream.rtvc').read_bytes()
        audit=inspect_stream(wire)
        if audit['total_bytes'] != row['bytes'] or any(audit['separate_mask_bytes'].values()):
            raise ValueError('stream bytes or zero-mask contract differ')
        records.append(dict(group=row['group'],point=name,**audit['bytes']))
    return {group: {name: mean(selected, BYTE_KEYS)
            for name in dict.fromkeys(r['point'] for r in records)
            if (selected := [r for r in records if r['group']==group and r['point']==name])}
        for group in GROUPS}


def figures(report, saved, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9))
    for i, group in enumerate(GROUPS):
        rows = report['group_means'][group]
        definitions = [('Native DCVC-UF', '#333333', 'o',
                        [f'uf_qp{q}' for q in (8,16,24,32,40,48,56)]),
            ('Global + local, G8', '#2166ac', 'o',
             [f'global_local_e{q}_g8' for q in ('0','0.25','0.5')]),
            ('Local only, G8', '#db8425', '^',
             [f'local_e{q}_g8' for q in ('0','0.25','0.5')]),
            ('Base + whole-frame G', '#27854c', '*', ['wholeframe_g_one_roi'])]
        for j, metric in enumerate(METRICS[:2]):
            ax = axes[i,j]
            for title, color, marker, names in definitions:
                ax.plot([rows[n]['bpp'] for n in names], [rows[n][metric] for n in names],
                    color=color, marker=marker, markersize=10 if marker=='*' else 5,
                    linestyle='None' if marker=='*' else '-', label=title)
            if j == 0:
                for q in ('0','0.25','0.5'):
                    r = rows[f'global_local_e{q}_g8']
                    ax.annotate('E'+{'0':'0','0.25':'25','0.5':'50'}[q],
                        (r['bpp'],r[metric]), xytext=(3,8), textcoords='offset points', fontsize=8)
            ax.set_xscale('log')
            ticks=(.01,.02,.05,.1) if group=='REDS_fullview' else (.005,.01,.02,.04)
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:g}'))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xlabel('Actual stream bits / pixel (log scale)')
            ax.set_ylabel('LPIPS (lower is better)' if j == 0 else 'PSNR dB (higher is better)')
            ax.set_title(group.replace('_', ' ') + (' / 6 windows' if i == 0 else ' / 7 windows'))
            ax.grid(alpha=.2, which='both')
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(.5,.04), frameon=False)
    fig.suptitle('Same regional G8 policy: zero E, then append E25 / E50 packets\nMeasured rates; UF support extended to quality index 56')
    fig.text(.5,.013, 'Equal-window dataset means. G cells are reselected after receiving E; not a fixed-mask ablation.',
        ha='center',fontsize=9)
    fig.subplots_adjust(bottom=.19, top=.89, hspace=.39, wspace=.25)
    save_plot(fig,output/'zero_E_rd.png')

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.5))
    for i, group in enumerate(GROUPS):
        for j, metric in enumerate(METRICS[:2]):
            ax=axes[i,j]
            for arm, title, color, marker in (('global_local','Global + local','#2166ac','o'),
                                              ('local','Local only','#db8425','^')):
                names=[f'{arm}_e{q}_g8' for q in ('0.25','0.5')]
                xs=[report['group_means'][group][n]['bpp'] for n in names]
                ax.plot(xs,[report['group_means'][group][n][metric] for n in names],
                    color=color,marker=marker,label=title+' / with G')
                ax.plot(xs,[saved['groups'][group][n]['E_only_quality'][metric] for n in names],
                    color=color,marker=marker,linestyle='--',label=title+' / G off')
            ax.set_xlabel('Unchanged charged stream bits / pixel');ax.grid(alpha=.2)
            ax.set_ylabel('LPIPS (lower is better)' if j == 0 else 'PSNR dB (higher is better)')
            ax.set_title(group.replace('_',' '))
    handles, labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,bbox_to_anchor=(.5,.04),frameon=False)
    fig.suptitle('What does G add after receiving the same E packets?')
    fig.text(.5,.012,'G-off metrics score saved fresh-decoded RGB: no new decode, no rate or latency saving claimed.',
        ha='center',fontsize=9)
    fig.subplots_adjust(bottom=.19,top=.92,hspace=.4,wspace=.25)
    save_plot(fig,output/'same_wire_G_off.png')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,default=ROOT/'visual_ablation_report')
    args=p.parse_args(argv)
    if args.output.resolve() == args.root.resolve() or any(args.output.resolve().is_relative_to(
            (args.root/name).resolve()) for name in ('visual_evaluation_recovered',
            'visual_uf_rate_extension','visual_zero_E','visual_evaluation_analysis')):
        raise ValueError('report output must not overwrite evidence')
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'complete.json').exists():
        marker=read(args.output/'complete.json')
        verify_artifacts(args.output,marker['artifacts'])
        if marker['code_sha256'] != digest(__file__):
            raise ValueError('completed report code changed')
        for path,sha in marker['inputs'].items():
            if digest(Path(path)) != sha:raise ValueError('completed report input changed')
        print('ABLATION_REPORT_VERIFIED_NO_RECOMPUTE');return
    extended=completed(args.root/'visual_uf_rate_extension')
    zero=completed(args.root/'visual_zero_E')
    saved=read(args.root/'visual_evaluation_analysis/saved_E_summary.json')
    if not saved.get('complete') or saved['points'] != 104:
        raise ValueError('saved-pixel G-off score set is incomplete')
    rows=joined_rows(extended['rows'],zero['rows'])
    if len(rows)!=234 or len({r['sample_id'] for r in rows})!=13:
        raise ValueError('unexpected joined evaluation scope')
    report=dict(complete=True,points=234,group_means=aggregates(rows),
        comparison=comparisons(rows),E_addition_pairs=paired_changes(rows),
        runtime_and_bytes=runtime_and_bytes(args.root,rows),rows=rows,
        packet_byte_decomposition=packet_costs(args.root,rows),
        original_results_unchanged=True,new_inference=False,
        scope='13 reused windows, two arms, seven UF indices; counts are not independent trials',
        runtime_scope='fresh decode includes model loading; CUDA peak is allocated bytes, not total board usage')
    save(args.output/'summary.json',report);figures(report,saved,args.output)
    paths=[args.root/f'{folder}/summary.json' for folder in ('visual_uf_rate_extension','visual_zero_E')]
    paths.append(args.root/'visual_evaluation_analysis/saved_E_summary.json')
    names=('summary.json','zero_E_rd.png','same_wire_G_off.png')
    save(args.output/'complete.json',dict(complete=True,code_sha256=digest(__file__),
        inputs={str(p):digest(p) for p in paths},artifacts={n:digest(args.output/n) for n in names}))
    print('ABLATION_REPORT_COMPLETE_NO_INFERENCE')


if __name__=='__main__':main()
