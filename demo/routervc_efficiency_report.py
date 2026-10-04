"""Read-only joined efficiency results, RD charts and fixed-frame comparisons."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

from demo.routervc_fullview_probe import read,digest,save,immutable,verify_artifacts
from demo.routervc_visual_report import interpolate_uf,save_plot

ROOT=Path('/root/autodl-fs/DCVC/runs/routervc_efficiency_20261004')
OLD=Path('/root/autodl-fs/DCVC/runs/routervc_revision_20261003')
METRICS=('lpips_alex','psnr_db','temporal_delta_mae')


def pair_q2(new_rows, old_rows):
    comparisons=[]
    for r in new_rows:
        old=[v for v in old_rows if v['sample_id']==r['sample_id']]
        reference=next(v for v in old if v['point']==r['point'])
        q1=[v for v in old if v['point'] in ('global_local_e0_g8','global_local_e0.25_g8','global_local_e0.5_g8')]
        uf=[v for v in old if v['point'].startswith('uf_qp')]
        comparisons.append(dict(sample_id=r['sample_id'],dataset=r['dataset'],point=r['point'],
            file_byte_ratio=r['bytes']/reference['bytes'],
            same_selection_delta={k:r['quality'][k]-reference[k] for k in METRICS},
            rate_matched={name:{k:(r['quality'][k]-value if (value:=interpolate_uf(curve,r['bpp'],k)) is not None else None)
                               for k in METRICS} for name,curve in (('old_q1_G8',q1),('native_UF',uf))}))
    groups={}
    for dataset in ('REDS','UVG'):
        rows=[r for r in comparisons if r['dataset']==dataset]
        groups[dataset]=dict(points=len(rows),mean_file_byte_ratio=sum(r['file_byte_ratio'] for r in rows)/len(rows),
            matched={name:dict(covered=len(covered),lower_lpips=sum(r['lpips_alex']<0 for r in covered),
                mean_delta={k:sum(r[k] for r in covered)/len(covered) for k in METRICS} if covered else None)
                for name in ('old_q1_G8','native_UF')
                if (covered:=[r['rate_matched'][name] for r in rows if r['rate_matched'][name]['lpips_alex'] is not None])})
    return dict(rows=comparisons,groups=groups,interpolation='per video, linear metric in log(bpp), no extrapolation or BD-rate')


def original_memory(records):
    rows=[]
    for sample in records:
        sid=sample['sample_id'];dataset=sample['dataset']
        for name in ('global_local_e0.5_g8','wholeframe_g_one_roi'):
            path=OLD/'visual_evaluation_recovered/samples'/sid/name/'fresh/decode.json'
            d=read(path);g=d['generation_runtime']
            rows.append(dict(sample_id=sid,dataset=dataset,point=name,
                last_call_GiB=d['peak_cuda_allocated_bytes']/2**30,
                all_G_peak_GiB=max(w['runtime']['peak_cuda_allocated_bytes'] for w in g['windows'])/2**30,
                G_seconds=g['seconds_model_load_excluded'],fresh_seconds=d['seconds']))
    grouped={}
    for ds in ('REDS','UVG'):
        grouped[ds]={}
        for name in ('global_local_e0.5_g8','wholeframe_g_one_roi'):
            selected=[r for r in rows if r['dataset']==ds and r['point']==name]
            grouped[ds][name]={k:sum(r[k] for r in selected)/len(selected)
                for k in ('last_call_GiB','all_G_peak_GiB','G_seconds','fresh_seconds')}
    return dict(rows=rows,groups=grouped,scope='all G calls maximum per video; model-loading/native-codec peaks not recoverable from old top-level field')


def plots(q2,schedule,old,zero,memory,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator,FuncFormatter,NullFormatter
    fig,axes=plt.subplots(2,2,figsize=(12.6,8.5))
    for i,(ds,group) in enumerate((('REDS','REDS_fullview'),('UVG','UVG_crop'))):
        q1=old['group_means'][group];z=zero['group_means'][group]['global_local_e0_g8']
        names=['global_local_e0.25_g8','global_local_e0.5_g8']
        entries=[('Native DCVC-UF','#303030','o',[q1[f'uf_qp{q}'] for q in (8,16,24,32,40,48,56)]),
                 ('qE=1, tiled G8','#2166ac','o',[z]+[q1[n] for n in names]),
                 ('qE=2, same E positions','#dd7824','s',[z]+[q2['group_means'][ds][n] for n in names]),
                 ('qE=1 E50, merged pairs','#8844aa','D',[dict(bpp=q1[names[1]]['bpp'],**schedule['group_means'][ds]['quality'])]),
                 ('Base + whole-frame G','#27854c','*',[q1['wholeframe_g_one_roi']])]
        for j,metric in enumerate(METRICS[:2]):
            ax=axes[i,j]
            for title,color,marker,rows in entries:
                ax.plot([r['bpp'] for r in rows],[r[metric] for r in rows],label=title,color=color,
                        marker=marker,markersize=9 if len(rows)==1 else 5,linestyle='-' if len(rows)>1 else 'None')
            ax.set_xscale('log')
            ticks=[.01,.02,.05,.1] if i==0 else [.005,.01,.02,.04]
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:g}'));ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xlabel('Actual complete-stream bits / pixel (log scale)')
            ax.set_ylabel('LPIPS (lower is better)' if j==0 else 'PSNR dB (higher is better)')
            ax.set_title('REDS / 6 resized full views' if i==0 else 'UVG / 7 existing crops');ax.grid(alpha=.2,which='both')
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=3,bbox_to_anchor=(.5,.04),frameon=False)
    fig.suptitle('Fixed-weight efficiency probes: cheaper E or bounded-pair G execution')
    fig.text(.5,.014,'qE changes precision, not E positions. Receiver G is reselected from actual Y. Lines join measured dataset means.',ha='center',fontsize=8.5)
    fig.subplots_adjust(top=.92,bottom=.2,hspace=.35,wspace=.23);save_plot(fig,output/'efficiency_rd.png')

    fig,axes=plt.subplots(1,2,figsize=(11.5,4.7))
    width=.24
    for offset,(title,color) in enumerate((('Original tiles','#2166ac'),('Merged pairs','#8844aa'),('Whole-frame G','#27854c'))):
        values=[];times=[]
        for ds in ('REDS','UVG'):
            if offset==1:
                r=schedule['group_means'][ds];values.append(r['new_G_peak_GiB']);times.append(r['new_G_seconds'])
            else:
                r=memory['groups'][ds]['global_local_e0.5_g8' if offset==0 else 'wholeframe_g_one_roi']
                values.append(r['all_G_peak_GiB']);times.append(r['G_seconds'])
        for ax,data in zip(axes,(values,times)):
            bars=ax.bar([i+(offset-1)*width for i in range(2)],data,width,color=color,label=title)
            ax.bar_label(bars,fmt='%.2f',fontsize=9)
    for ax in axes:ax.set_xticks([0,1],['REDS','UVG']);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_ylabel('All-G-call maximum allocated GiB / video');axes[1].set_ylabel('G seconds, excluding weight load')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,bbox_to_anchor=(.5,.06),frameon=False)
    fig.suptitle('Memory and observed runtime are separate objectives')
    fig.text(.5,.018,'Means over same windows; A800 only. Old per-call maxima corrected. Timings are not a controlled resident-throughput benchmark.',ha='center',fontsize=8)
    fig.subplots_adjust(top=.89,bottom=.23,wspace=.23);save_plot(fig,output/'memory_and_runtime.png')


def visuals(root,entries,output):
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=ImageFont.truetype(font_path,16)
    output.mkdir(exist_ok=True);records=[]
    for entry in entries:
        sample=entry['sample'];sid=sample['sample_id'];folder=OLD/'visual_evaluation_recovered/samples'/sid
        with np.load(folder/'source.npz',allow_pickle=False) as data:source=Image.fromarray(data['source'][8])
        images=[('Source',source),('UF bottom q8',Image.open(folder/'uf_qp8/fixed_frame.png')),
                ('qE=1 E50 / tiled G8',Image.open(folder/'global_local_e0.5_g8/fixed_frame.png')),
                ('qE=2 / same E cells',Image.open(root/'q2_probe/samples'/sid/'global_local_e0.5_g8/fixed_frame.png')),
                ('qE=1 / merged pairs',Image.open(root/'G_pair_probe/samples'/sid/'fixed_frame.png'))]
        width=320;height=round(source.height*width/source.width)
        canvas=Image.new('RGB',(5*width,height+62),'white');draw=ImageDraw.Draw(canvas)
        draw.text((8,5),sid+' / fixed frame 8 / display resize only',fill='black',font=font)
        for i,(label,image) in enumerate(images):
            draw.text((i*width+5,32),label,fill='black',font=font)
            canvas.paste(image.convert('RGB').resize((width,height),Image.Resampling.LANCZOS),(i*width,62))
            image.close()
        target=output/f'{sid}.png';canvas.save(target)
        records.append(dict(sample_id=sid,path=str(target),sha256=digest(target),display_only=True,fixed_frame=8))
    return records


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path,default=ROOT/'report')
    p.add_argument('--wait',action='store_true');args=p.parse_args(argv)
    if not os.environ.get('TMUX'):raise RuntimeError('run joined formal report in tmux')
    if any(args.output.resolve().is_relative_to((args.root/n).resolve()) for n in ('q2_probe','G_pair_probe','entropy_audit')):
        raise ValueError('report cannot overwrite source runs')
    from demo.chunk_enhancement_experiment import Run
    run=Run(SimpleNamespace(output=args.output,command='efficiency_report',max_hours=12));run.thread.start()
    try:
        required=[args.root/n/'complete.json' for n in ('q2_probe','G_pair_probe','entropy_audit')]
        while not all(p.exists() for p in required):
            if not args.wait:raise ValueError('wait for q2 and scheduled G completion')
            run.update(phase='waiting_for_completed_probes');run.check();time.sleep(10)
        if (run.root/'complete.json').exists():
            marker=read(run.root/'complete.json');verify_artifacts(run.root,marker['artifacts'])
            if marker['code_sha256']!=digest(__file__):raise ValueError('completed report code changed')
            for path,sha in marker['inputs'].items():
                if digest(Path(path))!=sha:raise ValueError('report evidence changed')
            print('EFFICIENCY_REPORT_VERIFIED_READ_ONLY');return
        run.update(phase='read_only_report')
        for p in required:verify_artifacts(p.parent,read(p)['artifacts'])
        inputs=required+[args.root/n/'summary.json' for n in ('q2_probe','G_pair_probe','entropy_audit')]
        inputs+=[OLD/'visual_uf_rate_extension/summary.json',OLD/'visual_zero_E/summary.json']
        binding={str(p):digest(p) for p in inputs};immutable(run.root/'inputs.json',binding)
        q2=read(args.root/'q2_probe/summary.json');scheduled=read(args.root/'G_pair_probe/summary.json')
        old=read(OLD/'visual_uf_rate_extension/summary.json');zero=read(OLD/'visual_zero_E/summary.json')
        old_rows=old['rows']+[r for r in zero['rows'] if r['point'].endswith('_e0_g8')]
        comparisons=pair_q2(q2['rows'],old_rows)
        protocol=read(args.root/'q2_probe/protocol.json');entries=protocol['original']['sources']
        memory=original_memory([e['sample'] for e in entries])
        plots(q2,scheduled,old,zero,memory,run.root)
        pictures=visuals(args.root,entries,run.root/'fixed_visuals')
        summary=dict(complete=True,q2= q2['group_means'],q2_comparisons=comparisons,
            scheduled_G=scheduled['group_means'],historical_G_memory_correction=memory,
            fixed_visuals=pictures,no_training=True,no_model_promotion=True,
            source_scope='same 13 development/evaluation windows, not an independent full test set')
        save(run.root/'summary.json',summary)
        names=['summary.json','inputs.json','efficiency_rd.png','memory_and_runtime.png']
        names+=[str(Path(r['path']).relative_to(run.root)) for r in pictures]
        save(run.root/'complete.json',dict(complete=True,code_sha256=digest(__file__),inputs=binding,
            artifacts={n:digest(run.root/n) for n in names}))
        run.update(phase='complete',completed=13,total=13);print('EFFICIENCY_REPORT_COMPLETE',flush=True)
    finally:
        run.log_resources();run.stop.set();run.thread.join(timeout=3);run.lock.close()


if __name__=='__main__':main()
