"""Matched-support, CPU-only analysis of completed RouterVC real streams.

The native UF curve uses ALL charged bytes (native bits plus transmitted
metadata), not bare payload bits. Interpolation is a labeled within-interval
linear diagnostic, never extrapolation or BD-rate. No metrics are recomputed.
"""
import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from demo.routervc_report import file_hash, atomic_json, save_figure, npz_shape, normalize_record

DEFAULT=Path('/root/autodl-fs/DCVC/runs/routervc_20261003/supplement')
ROUTERS=('context_raw','context_smooth','local_raw','local_smooth')
UF_METHODS=tuple(f'uf_qp{q}' for q in (8,16,24,32))
COLORS=dict(context_raw='#397bb5',context_smooth='#14559c',
            local_raw='#d8a254',local_smooth='#b96d15',base='#777777',
            e_only='#328454',g_only='#9463ad',full_g='#d16a98',full_frame_g='#a51b60')
METRICS=('lpips_alex','psnr_db','temporal_delta_mae')
G_TIMES=('g_execution_seconds_sum','g_model_load_seconds','g_restore_window_calls')


def read(path):
    with Path(path).open() as stream:return json.load(stream)


def sample_key(record):
    return record['dataset'],record['sample_id']


def point_key(record):
    return record['method'],record.get('ratio'),record.get('max_g')


def record_key(record):
    return (*sample_key(record),*point_key(record))


def generation_timing(decode):
    """Extract recorded intervals, never estimate milliseconds from area/calls."""
    runtime=decode.get('generation_runtime')
    executed=decode.get('generation_executed',False)
    if not executed:
        if runtime is not None:raise ValueError('G-off record unexpectedly has G runtime')
        return dict(g_execution_seconds_sum=0.,g_model_load_seconds=0.,
                    g_restore_window_calls=0,g_timing_status='generation_not_executed')
    if runtime is None:
        return dict(g_execution_seconds_sum=None,g_model_load_seconds=None,
                    g_restore_window_calls=None,g_timing_status='missing_recorded_runtime')
    def seconds(value):
        if value is None:return None
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
            raise ValueError('invalid recorded generation interval')
        return float(value)
    windows=runtime.get('windows')
    execution=seconds(runtime.get('seconds_model_load_excluded'))
    per_window=[]
    if isinstance(windows,list):
        for window in windows:
            per_window.append(seconds(window.get('runtime',{}).get('seconds_model_load_excluded')))
    if isinstance(windows,list) and windows and all(v is not None for v in per_window):
        summed=sum(per_window)
        if execution is not None and not math.isclose(execution,summed,rel_tol=1e-9,abs_tol=1e-7):
            raise ValueError('G aggregate interval differs from recorded window intervals')
        execution=summed
    return dict(g_execution_seconds_sum=execution,
                g_model_load_seconds=seconds(runtime.get('model_load_seconds')),
                g_restore_window_calls=len(windows) if isinstance(windows,list) else None,
                g_timing_status='recorded_intervals' if execution is not None else 'missing_recorded_execution_interval')


def optional_means(rows,keys):
    result={}
    for key in keys:
        values=[r.get(key) for r in rows if r.get(key) is not None]
        result[key]=dict(mean=float(np.mean(values)) if values else None,available=len(values),total=len(rows))
    return result


def indexed(records):
    result={}
    for record in records:
        key=record_key(record)
        if key in result:raise ValueError(f'duplicate actual operating point: {key}')
        result[key]=record
    return result


def common_support(records,max_g):
    """Intersect video support across UF's four QPs and every plotted route point."""
    groups=defaultdict(set)
    for record in records:
        if record['method'] in UF_METHODS:
            groups[(record['method'],)].add(sample_key(record))
        elif record['method'] in ROUTERS and record.get('max_g')==max_g:
            groups[point_key(record)].add(sample_key(record))
    if any((method,) not in groups for method in UF_METHODS):
        raise ValueError('four measured UF QPs are required for matched comparison')
    if not any(len(key)==3 for key in groups):return []
    support=set.intersection(*(values for values in groups.values()))
    return sorted(support)


def aggregate(rows):
    if not rows:raise ValueError('cannot average an empty measured point')
    return dict(windows=len(rows),samples=[list(v) for v in sorted(sample_key(r) for r in rows)],
        bpp=float(np.mean([r['bpp'] for r in rows])),
        bytes=float(np.mean([r['bytes'] for r in rows])),
        quality={m:float(np.mean([r['quality'][m] for r in rows])) for m in METRICS},
        seconds=float(np.mean([r['decode_seconds'] for r in rows])),
        boundary_edges=float(np.mean([r['boundary_edges'] for r in rows])),
        components=float(np.mean([r['components'] for r in rows])),
        actual_G_roi_calls=float(np.mean([r['actual_G_roi_calls'] for r in rows])),
        generation_timing=optional_means(rows,G_TIMES),
        peak_cuda_allocated_bytes=max(r['peak_cuda_allocated_bytes'] for r in rows),
        native_bytes_mean=float(np.mean([r['native_bytes'] for r in rows])))


def matched_points(records,support,max_g):
    selected=set(map(tuple,support));groups=defaultdict(list)
    for record in records:
        if sample_key(record) not in selected:continue
        method=record['method']
        if method in ROUTERS or method=='g_only':
            if record.get('max_g')!=max_g:continue
        if method not in {*ROUTERS,*UF_METHODS,'base','e_only','g_only','full_g','full_frame_g'}:continue
        groups[point_key(record)].append(record)
    points=[];omitted=[]
    for key,rows in sorted(groups.items(),key=lambda item:str(item[0])):
        got={sample_key(r) for r in rows}
        if got!=selected:
            omitted.append(dict(point=list(key),missing_samples=[list(v) for v in sorted(selected-got)]));continue
        points.append(dict(method=key[0],ratio=key[1],max_g=key[2],**aggregate(rows)))
    return points,omitted


def measured_values(record):
    return dict(bytes=float(record['bytes']),bpp=record['bpp'],seconds=record['decode_seconds'],
                boundary_edges=float(record['boundary_edges']),components=float(record['components']),
                **record['quality'])


def paired_summary(pairs,label):
    """Every difference is variant minus reference on the very same sample."""
    details=[]
    for reference,variant in pairs:
        if sample_key(reference)!=sample_key(variant):raise ValueError('unpaired video comparison')
        a,b=measured_values(reference),measured_values(variant)
        details.append(dict(dataset=reference['dataset'],sample_id=reference['sample_id'],
            reference=point_key(reference),variant=point_key(variant),
            delta={k:b[k]-a[k] for k in a},
            generation_timing_delta={k:variant[k]-reference[k]
                if variant.get(k) is not None and reference.get(k) is not None else None for k in G_TIMES}))
    groups={}
    for domain in ('REDS','UVG','All'):
        values=[r for r in details if domain=='All' or r['dataset']==domain]
        if not values:continue
        groups[domain]=dict(pairs=len(values),mean_delta={k:float(np.mean([r['delta'][k] for r in values]))
            for k in values[0]['delta']},
            generation_timing_delta=optional_means([r['generation_timing_delta'] for r in values],G_TIMES),
            variant_lower_lpips=sum(r['delta']['lpips_alex']<0 for r in values),
            variant_higher_psnr=sum(r['delta']['psnr_db']>0 for r in values))
    return dict(label=label,direction='variant minus reference; negative LPIPS is improvement',
                groups=groups,records=details)


def comparisons(records,support):
    selected=set(map(tuple,support));rows=[r for r in records if sample_key(r) in selected]
    idx=indexed(rows);groups={}
    for removed in ('route_no_g','route_no_e'):
        pairs=[]
        for variant in rows:
            if variant['method']!=removed:continue
            key=(*sample_key(variant),'context_smooth',.5,4)
            if key in idx:pairs.append((idx[key],variant))
        groups[removed]=paired_summary(pairs,f'{removed} minus fixed context_smooth r0.5 G4')
    for arm in ('context','local'):
        for cap in (4,8):
            for ratio in (.25,.5,.75):
                pairs=[]
                for sid in sorted(selected):
                    a=idx.get((*sid,f'{arm}_raw',ratio,cap));b=idx.get((*sid,f'{arm}_smooth',ratio,cap))
                    if a is not None and b is not None:pairs.append((a,b))
                if pairs:
                    groups[f'{arm}_smooth_minus_raw_r{ratio:g}_g{cap}']=paired_summary(pairs,'smooth minus raw')
                    groups[f'{arm}_smooth_minus_raw_r{ratio:g}_g{cap}']['same_E_indices']=all(
                        a.get('selected_E')==b.get('selected_E') for a,b in pairs)
    for mode in ('raw','smooth'):
        for cap in (4,8):
            for ratio in (.25,.5,.75):
                pairs=[]
                for sid in sorted(selected):
                    a=idx.get((*sid,f'local_{mode}',ratio,cap));b=idx.get((*sid,f'context_{mode}',ratio,cap))
                    if a is not None and b is not None:pairs.append((a,b))
                if pairs:groups[f'context_minus_local_{mode}_r{ratio:g}_g{cap}']=paired_summary(pairs,'context minus local; actual routed outputs')
    pairs=[]
    for sid in sorted(selected):
        a=idx.get((*sid,'full_g',0.,16));b=idx.get((*sid,'full_frame_g',0.,16))
        if a is not None and b is not None:pairs.append((a,b))
    groups['full_frame_minus_grid_G']=paired_summary(pairs,'one full-frame ROI minus 16 grid ROIs; processing context also changes')
    return groups


def interpolate_uf(points,target):
    """Optional linear-between-adjacent-measurements diagnostic, no extrapolation."""
    if not math.isfinite(target):raise ValueError('nonfinite target rate')
    ordered=sorted(points,key=lambda r:r['bpp'])
    if len(ordered)<2:return dict(status='insufficient_points')
    if any(not math.isfinite(r['bpp']) or r['bpp']<=0 for r in ordered):raise ValueError('invalid measured rates')
    if len({r['bpp'] for r in ordered})!=len(ordered):
        return dict(status='ambiguous_duplicate_rate')
    if target<ordered[0]['bpp'] or target>ordered[-1]['bpp']:
        return dict(status='outside_measured_range',minimum=ordered[0]['bpp'],maximum=ordered[-1]['bpp'])
    exact=[r for r in ordered if r['bpp']==target]
    if len(exact)>1:return dict(status='ambiguous_duplicate_rate')
    if exact:return dict(status='exact_measured_rate',quality=exact[0]['quality'],interval=[exact[0]['method']])
    for left,right in zip(ordered,ordered[1:]):
        if left['bpp']<target<right['bpp']:
            fraction=(target-left['bpp'])/(right['bpp']-left['bpp'])
            return dict(status='interpolated_inside_adjacent_interval',fraction=fraction,
                interval=[left['method'],right['method']],
                quality={m:left['quality'][m]+fraction*(right['quality'][m]-left['quality'][m]) for m in METRICS})
    raise ValueError('unable to identify measured interpolation interval')


def auxiliary_interpolation(records,support):
    selected=set(map(tuple,support));by_sample=defaultdict(list)
    for record in records:
        if sample_key(record) in selected:by_sample[sample_key(record)].append(record)
    comparisons=[]
    for key,rows in by_sample.items():
        uf=[r for r in rows if r['method'] in UF_METHODS]
        for record in rows:
            if record['method'] not in ROUTERS:continue
            estimate=interpolate_uf(uf,record['bpp'])
            item=dict(dataset=key[0],sample_id=key[1],point=list(point_key(record)),bpp=record['bpp'],uf=estimate)
            if 'quality' in estimate:
                item['router_minus_uf_estimate']={m:record['quality'][m]-estimate['quality'][m] for m in METRICS}
            comparisons.append(item)
    return dict(scope='auxiliary per-video linear interpolation in actual total bpp; no extrapolation, no BD-rate',records=comparisons)


def draw_curves(destination,curve_sets):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    outputs=[]
    for group in curve_sets:
        domain,cap,points=group['dataset'],group['max_g'],group['points']
        fig,axes=plt.subplots(1,2,figsize=(12.4,4.8))
        uf=sorted([p for p in points if p['method'] in UF_METHODS],key=lambda p:p['bpp'])
        for axis,metric in zip(axes,('lpips_alex','psnr_db')):
            axis.plot([p['bpp'] for p in uf],[p['quality'][metric] for p in uf],
                '-o',color='black',lw=2.2,label='DCVC-UF (QP8 / 16 / 24 / 32)')
            for p in uf:
                axis.annotate(p['method'].replace('uf_',''),(p['bpp'],p['quality'][metric]),
                              xytext=(4,4),textcoords='offset points',fontsize=7)
            for method in ROUTERS:
                rows=sorted([p for p in points if p['method']==method],key=lambda p:p['bpp'])
                if not rows:continue
                axis.plot([p['bpp'] for p in rows],[p['quality'][metric] for p in rows],
                    marker='o',linestyle='--' if method.endswith('raw') else '-',
                    color=COLORS[method],label=method)
            for method in ('g_only','full_g','full_frame_g'):
                rows=[p for p in points if p['method']==method]
                if rows:axis.scatter([p['bpp'] for p in rows],[p['quality'][metric] for p in rows],
                    marker={'g_only':'s','full_g':'^','full_frame_g':'D'}[method],s=50,
                    color=COLORS[method],label=f'{method} ({"1 ROI" if method=="full_frame_g" else "16 ROIs" if method=="full_g" else f"cap {cap}"})')
            axis.grid(alpha=.2);axis.set_xlabel('Actual total bits / pixel / frame (metadata included)')
        axes[0].set_ylabel('Whole-frame LPIPS (lower is better)')
        axes[1].set_ylabel('Whole-frame PSNR / dB (higher is better)')
        axes[1].legend(fontsize=7,loc='best')
        fig.suptitle(f'{domain} | {len(group["samples"])} identical videos at every point | Router G cap {cap}\n'
                     'UF includes its charged decoding sidecar; RouterVC includes its own headers. All-G has a different compute budget.',fontsize=10)
        fig.tight_layout();name=f'rd_{domain}_g{cap}.png'
        save_figure(fig,destination/name);plt.close(fig);outputs.append(name)

        fig,axis=plt.subplots(figsize=(8,5))
        for family,methods in [('DCVC-UF',UF_METHODS),*((m,(m,)) for m in (*ROUTERS,'g_only','full_g','full_frame_g'))]:
            rows=[p for p in points if p['method'] in methods]
            if not rows:continue
            axis.scatter([p['seconds'] for p in rows],[p['quality']['lpips_alex'] for p in rows],
                         color='black' if family=='DCVC-UF' else COLORS[family],label=family,s=44)
        axis.set_xlabel('Recorded receiver interval / s (loading + checks included)')
        axis.set_ylabel('Whole-frame LPIPS (lower is better)');axis.grid(alpha=.2)
        axis.legend(fontsize=8);axis.set_title(f'{domain} | same {len(group["samples"])} videos | Router G cap {cap}\nExcludes process startup/output writes; not pure-network latency. Full-frame G uses one ROI.',fontsize=10)
        fig.tight_layout();name=f'time_quality_{domain}_g{cap}.png'
        save_figure(fig,destination/name);plt.close(fig);outputs.append(name)
    return outputs


def load_records(root,run=None):
    summary=read(root/'summary.json');done=read(root/'complete.json');report=read(root/'report/summary.json')
    summary_hash=file_hash(root/'summary.json')
    if (not summary.get('complete') or not done.get('complete') or not report.get('complete')
            or done['summary']!=summary_hash or report['dependencies']['summary_sha256']!=summary_hash
            or summary['baseline_protocol']!=file_hash(root/'protocol.json')):
        raise ValueError('completed supplementary evaluation and matching report are required')
    hashes={str(root/name):file_hash(root/name) for name in ('summary.json','complete.json','report/summary.json','protocol.json')}
    for name,expected in {**report['input_hashes'],**{str(root/'report'/n):h for n,h in report['artifacts'].items()}}.items():
        if run:run.check()
        path=Path(name)
        if str(path) not in hashes:hashes[str(path)]=file_hash(path)
        if hashes[str(path)]!=expected:raise ValueError(f'changed saved report evidence: {path}')
    raw=indexed(summary['records']);normalized=indexed(report['records'])
    if raw.keys()!=normalized.keys():raise ValueError('report and actual result operating points differ')
    records=[]
    for index,(key,item) in enumerate(raw.items()):
        if run:run.check();run.update(phase='verifying_saved_artifacts',completed=index,total=len(raw))
        folder=Path(item['folder'])
        if not folder.is_absolute():folder=root/folder
        for name,expected in item['artifacts'].items():
            path=folder/name
            if str(path) not in hashes:hashes[str(path)]=file_hash(path)
            if hashes[str(path)]!=expected:raise ValueError(f'changed fresh artifact: {path}')
        source_path=Path(item['source_path'])
        if str(source_path) not in hashes:hashes[str(source_path)]=file_hash(source_path)
        if hashes[str(source_path)]!=item['source_hash']:raise ValueError('changed recorded evaluation source')
        d=item['decode'];norm=normalized[key]
        if read(folder/'fresh/decode.json')!=d:raise ValueError('embedded receiver metadata differs from fresh/decode.json')
        if d['source_frames_read'] or item['bytes']!=norm['bytes'] or item['bytes']!=d['total_bytes']:
            raise ValueError('source-free/actual-byte record mismatch')
        if item['quality']!=norm['quality'] and any(item['quality'][m]!=norm['quality'][m] for m in METRICS):
            raise ValueError('report metrics differ from the measured record')
        shape=npz_shape(folder/'fresh/reconstruction.npz')
        if normalize_record(item,shape)!=norm:
            raise ValueError('saved report fields differ from actual fresh record/geometry')
        calls=d.get('actual_G_roi_calls',len(d.get('route',{}).get('indices',[])))
        native=item.get('native_bytes',d.get('native_bytes',d.get('base_bytes')))
        if native is None:raise ValueError('missing native-byte accounting')
        records.append(dict(norm,actual_G_roi_calls=calls,native_bytes=native,
                            selected_E=item.get('selected_E'),**generation_timing(d)))
    return records,hashes


def analyze(root,destination=None,run=None):
    root=Path(root);destination=Path(destination or root/'analysis');destination.mkdir(parents=True,exist_ok=True)
    deps=dict(source_summary=file_hash(root/'summary.json'),source_report=file_hash(root/'report/summary.json'),
              source_complete=file_hash(root/'complete.json'),code=file_hash(Path(__file__)),
              report_helpers=file_hash(REPO/'demo/routervc_report.py'),
              wrapper=file_hash(REPO/'demo/run_routervc_analysis.sh'))
    saved=destination/'summary.json'
    if saved.exists():
        result=read(saved)
        if result['dependencies']!=deps:raise ValueError('analysis dependency changed; use a new output')
        for name,expected in result['input_hashes'].items():
            if run:run.check()
            if file_hash(Path(name))!=expected:raise ValueError(f'changed analysis input: {name}')
        for name,expected in result['artifacts'].items():
            if file_hash(destination/name)!=expected:raise ValueError(f'changed analysis figure: {name}')
        return result
    records,hashes=load_records(root,run)
    caps=sorted({r['max_g'] for r in records if r['method'] in ROUTERS})
    curves=[];supports={};all_support=set()
    for cap in caps:
        support=common_support(records,cap)
        if not support:raise ValueError(f'no matched UF/Router video support for G cap {cap}')
        supports[str(cap)]=[list(v) for v in support];all_support.update(support)
        for domain in ('REDS','UVG','All'):
            chosen=[v for v in support if domain=='All' or v[0]==domain]
            if not chosen:continue
            points,omitted=matched_points(records,chosen,cap)
            curves.append(dict(dataset=domain,max_g=cap,samples=[list(v) for v in chosen],points=points,omitted=omitted))
    artifacts=draw_curves(destination,curves)
    result=dict(complete=True,dependencies=deps,input_hashes=hashes,
        matched_support_by_G_cap=supports,curve_sets=curves,
        input_sample_count=len({sample_key(r) for r in records}),matched_sample_count=len(all_support),
        excluded_samples=[list(v) for v in sorted({sample_key(r) for r in records}-all_support)],
        recorded_timing=[dict(dataset=r['dataset'],sample_id=r['sample_id'],point=list(point_key(r)),
            receiver_seconds=r['decode_seconds'],actual_G_roi_calls=r['actual_G_roi_calls'],
            **{k:r[k] for k in (*G_TIMES,'g_timing_status')}) for r in records if sample_key(r) in all_support],
        paired=comparisons(records,all_support),auxiliary_uf_interpolation=auxiliary_interpolation(records,all_support),
        artifacts={name:file_hash(destination/name) for name in artifacts},
        interpretation='Real fresh whole-frame results; exact same sample support in each figure; no independent-test or universal-dominance claim.',
        byte_scope='Every RD point uses full charged bytes. Native UF payload bytes are retained separately, never mixed with RouterVC container rates.',
        native_byte_scope='native_bytes_mean is only the UF base bitstream component; it excludes E packets and all wrappers.',
        timing_scopes=dict(
            receiver_interval='Recorded receiver interval includes stream parsing, model loading and checks; excludes process startup, output NPZ/JSON writes and quality evaluation. Filesystem cold/warm state is not normalized.',
            receiver_timer_difference='Main RouterVC and route_no_g start before CUDA peak reset; UF and explicit-G baseline workers start after reset. These are recorded intervals, not identical end-to-end wall-clock instrumentation.',
            g_execution_seconds_sum='Sum of measured PersistentSeedVR2.restore windows: VAE encode, noise/condition, runner inference, output resize and CUDA synchronization. Excludes model load, prior input preprocessing, later RGB conversion and outer temporal/ROI blending/feathering. Not pure DiT latency.',
            g_model_load_seconds='Recorded PersistentSeedVR2 initialization/loading interval; excludes outer adapter/branch preparation. Cold disk loading is not separately identified.',
            g_restore_window_calls='Number of recorded ROI-by-temporal-window restore invocations, NOT total neural-network/DiT/VAE calls.',
            actual_G_roi_calls='Number of processing ROIs; full_frame_g has one even though its display grid contains 16 G cells.',
            missing='Missing measurements stay null with availability counts; G-off is explicitly zero. No area or call-count timing estimates.'),
        signs='All paired deltas are variant minus reference. Negative LPIPS/time/bytes is lower; positive PSNR is higher.',
        no_inference=True,no_metric_recalculation=True,no_bd_rate=True)
    atomic_json(saved,result)
    return result


def main(args):
    if not os.environ.get('TMUX'):raise RuntimeError('run saved-artifact analysis inside tmux')
    from demo.chunk_enhancement_experiment import Run
    destination=args.output or args.root/'analysis'
    run=Run(SimpleNamespace(output=destination,command='analysis',max_hours=args.max_hours))
    run.update(cpu_only=True)
    run.thread.start()
    try:
        if args.wait_complete:
            while not (args.root/'complete.json').exists():
                run.check();run.update(phase='waiting_for_supplement_completion',cpu_only=True);time.sleep(10)
        result=analyze(args.root,destination,run)
        run.update(phase='complete',cpu_only=True,matched_samples=result['matched_sample_count'],figures=len(result['artifacts']))
        print(json.dumps(dict(complete=True,matched_samples=result['matched_sample_count'],excluded_samples=result['excluded_samples'])))
    except BaseException as error:
        atomic_json(destination/'last_failure.json',dict(error=repr(error),phase=run.progress));raise
    finally:
        run.log_resources();run.stop.set();run.thread.join(timeout=3)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--wait-complete',action='store_true')
    parser.add_argument('--max-hours',type=float,default=24.)
    main(parser.parse_args())
