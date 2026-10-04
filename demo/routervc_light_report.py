"""Read-only q2 Router adaptation report: matched curves, routes and fixed images."""
import argparse
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

from demo.routervc_fullview_probe import read,digest,save,immutable,verify_artifacts
from demo.routervc_light_teacher import OUTPUT,HISTORY
from demo.routervc_light_evaluate import REVISION,ORIGINAL,ARMS
from demo.routervc_visual_report import interpolate_uf,save_plot

METRICS=('lpips_alex','psnr_db','temporal_delta_mae')


def mean(rows,keys):
    if not rows: return None
    return {k:sum(r[k] for r in rows)/len(rows) for k in keys}


def interpolate(curve,rate,key):
    # Prefixes can repeat when no additional complete bundle fits the cap.
    unique={}
    for row in curve:
        previous=unique.setdefault(row['bpp'],row)
        if abs(previous[key]-row[key])>1e-10:raise ValueError('same prefix rate has differing metrics')
    if rate in unique:return unique[rate][key]
    if len(unique)<2:return None
    return interpolate_uf(list(unique.values()),rate,key)


def comparisons(rows,old):
    result=[]
    for row in rows:
        if row['version']!='new':continue
        peers=[v for v in rows if v['sample_id']==row['sample_id'] and v['version']=='old' and v['arm']==row['arm']]
        same=next(v for v in peers if v['ratio']==row['ratio'])
        uf=[r for r in old if r['sample_id']==row['sample_id'] and r['point'].startswith('uf_qp')]
        result.append(dict(sample_id=row['sample_id'],dataset=row['dataset'],arm=row['arm'],ratio=row['ratio'],
            same_cap_delta={k:row[k]-same[k] for k in ('bpp',*METRICS)},
            E_changed=row['E_indices']!=same['E_indices'],G_changed=row['G_indices']!=same['G_indices'],
            rate_matched={name:{k:row[k]-value if (value:=interpolate(curve,row['bpp'],k)) is not None else None
                for k in METRICS} for name,curve in (('old_q2',peers),('native_UF',uf))}))
    groups={}
    for ds in ('REDS','UVG'):
        groups[ds]={}
        for arm in ARMS:
            selected=[r for r in result if r['dataset']==ds and r['arm']==arm and r['ratio']>0]
            groups[ds][arm]=dict(points=len(selected),E_changed=sum(r['E_changed'] for r in selected),
                G_changed=sum(r['G_changed'] for r in selected),
                same_cap_mean_delta=mean([r['same_cap_delta'] for r in selected],('bpp',*METRICS)),
                matched={name:dict(covered=len(covered),lower_lpips=sum(r['lpips_alex']<0 for r in covered),
                    mean_delta=mean(covered,METRICS)) for name in ('old_q2','native_UF')
                    for covered in [[r['rate_matched'][name] for r in selected if r['rate_matched'][name]['lpips_alex'] is not None]]})
    return dict(rows=result,groups=groups,interpolation='per-video log(bpp), no extrapolation, no BD-rate',
                dependence='E25/E50 share 13 videos, not independent trials; E0 reported separately')


def plot(groups,historical,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter,FixedLocator,NullFormatter
    for arm in ARMS:
        fig,axes=plt.subplots(2,2,figsize=(12,8.5))
        for i,(ds,group) in enumerate((('REDS','REDS_fullview'),('UVG','UVG_crop'))):
            old=historical[group]
            curves=[('Native DCVC-UF','#333333','o',[old[f'uf_qp{q}'] for q in (8,16,24,32,40,48,56)]),
                    ('q1 / old Router','#6684ab','s',[old[f'{arm}_e{r:g}_g8'] for r in (0.,.25,.5)]),
                    ('q2 / old Router, reallocated','#dd8420','o',[groups[ds][f'old_{arm}_e{r:g}_g8'] for r in (0.,.25,.5)]),
                    ('q2 / retrained Router','#864ab8','o',[groups[ds][f'new_{arm}_e{r:g}_g8'] for r in (0.,.25,.5)]),
                    ('Base + whole-frame G','#268c66','*',[old['wholeframe_g_one_roi']])]
            for j,key in enumerate(METRICS[:2]):
                ax=axes[i,j]
                for name,color,marker,points in curves:
                    ax.plot([v['bpp'] for v in points],[v[key] for v in points],label=name,color=color,
                            marker=marker,markersize=9 if len(points)==1 else 5,
                            linestyle='-' if len(points)>1 else 'None')
                ax.set_xscale('log');ax.xaxis.set_major_locator(FixedLocator([.005,.01,.02,.05,.1]))
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:g}'));ax.xaxis.set_minor_formatter(NullFormatter())
                ax.set_title('REDS / 6 resized full views' if ds=='REDS' else 'UVG / 7 existing crops')
                ax.set_xlabel('Actual complete-stream bits / pixel');ax.set_ylabel('LPIPS (lower)' if j==0 else 'PSNR dB (higher)')
                ax.grid(alpha=.2,which='both')
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=3,bbox_to_anchor=(.5,.035),frameon=False)
        fig.suptitle(f'Router-only adaptation to cheaper enhancement packets / {arm}')
        fig.text(.5,.015,'Frozen UF/E/G; same q2 byte caps, actual rates may differ. Lines join measured dataset means.',ha='center',fontsize=9)
        fig.subplots_adjust(top=.92,bottom=.20,hspace=.35,wspace=.24);save_plot(fig,output/f'rd_{arm}.png')


def fixed_visuals(root,rows,old,output):
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    output.mkdir(exist_ok=True);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    records=[]
    for sid in dict.fromkeys(r['sample_id'] for r in rows):
        selected=[next(r for r in rows if r['sample_id']==sid and r['point']==name) for name in
            ('old_global_local_e0.5_g8','new_global_local_e0.5_g8','new_local_e0.5_g8')]
        uf=min((r for r in old if r['sample_id']==sid and r['point'].startswith('uf_qp')),
               key=lambda r:abs(math.log(r['bpp']/selected[1]['bpp'])))
        uf_root=ORIGINAL if int(uf['point'].split('qp')[1])<=32 else REVISION/'visual_uf_rate_extension'
        uf_folder=uf_root/'samples'/sid/uf['point']
        verify_artifacts(uf_folder,read(uf_folder/'result.json')['artifacts'])
        with np.load(ORIGINAL/'samples'/sid/'source.npz',allow_pickle=False) as data:source=Image.fromarray(data['source'][8])
        images=[('Source',source,None),('UF / nearest measured rate',Image.open(uf_folder/'fixed_frame.png'),uf)]
        for title,r in zip(('q2 / old global-local','q2 / new global-local','q2 / new local'),selected):
            with np.load(root/'evaluation/samples'/sid/r['point']/'fresh/reconstruction.npz',allow_pickle=False) as data:
                images.append((title,Image.fromarray(data['reconstruction'][8]),r))
        width=320;height=round(source.height*width/source.width)
        canvas=Image.new('RGB',(width*5,height+90),'white');draw=ImageDraw.Draw(canvas)
        draw.text((8,4),sid+' / fixed frame 8 / display resized; metrics on original coded pixels',fill='black',font=font)
        for i,(title,img,score) in enumerate(images):
            draw.text((width*i+5,30),title,fill='black',font=font)
            if score:draw.text((width*i+5,53),f"{score['bpp']:.5f} bpp | LPIPS {score['lpips_alex']:.4f}",fill='black',font=font)
            canvas.paste(img.convert('RGB').resize((width,height),Image.Resampling.LANCZOS),(width*i,90));img.close()
        path=output/f'{sid}.png';canvas.save(path)
        records.append(dict(sample_id=sid,path=str(path),sha256=digest(path),UF_point=uf['point'],
            selection='nearest measured log-rate to new global E50, not equal-rate or quality-selected',fixed_frame=8))
    return records


def build(root,output):
    evaluation=root/'evaluation';prot=read(evaluation/'protocol.json')
    verify_artifacts(evaluation,read(evaluation/'complete.json')['artifacts'])
    rows=[];e_gain=[]
    from demo.routervc_light_evaluate import check_point
    for entry in prot['sources']:
        s=entry['sample'];sid=s['sample_id']
        for point in prot['points']:
            folder=evaluation/'samples'/sid/point['name'];r=check_point(folder,prot,s)
            e=read(folder/'encode.json');d=r['decode'];g=d['generation_runtime']
            rows.append(dict(sample_id=sid,dataset=s['dataset'],point=point['name'],
                **{k:point[k] for k in ('version','arm','ratio')},bytes=r['bytes'],bpp=r['bpp'],**r['quality'],
                Goff=r.get('same_wire_G_off_quality'),receiver_seconds=d['seconds'],
                G_seconds=g['seconds_model_load_excluded'] if g else 0.,
                G_peak_GiB=max((w['runtime']['peak_cuda_allocated_bytes'] for w in g['windows']),default=0)/2**30 if g else 0.,
                worker_peak_GiB=None if r['reused'] else d['peak_cuda_allocated_bytes']/2**30,
                E_indices=e['plan']['selected_indices'],G_indices=d['route']['indices'],
                unused_E_budget=e['plan']['unused_e_budget_bytes'],E_budget=e['plan']['budget_e_packet_bytes'],
                reused=r['reused']))
    groups={ds:{name:dict(windows=len(rs),**mean(rs,('bytes','bpp',*METRICS,'receiver_seconds','G_seconds','G_peak_GiB')))
            for name in dict.fromkeys(r['point'] for r in rows)
            for rs in [[r for r in rows if r['dataset']==ds and r['point']==name]]} for ds in ('REDS','UVG')}
    for version in ('old','new'):
        for arm in ARMS:
            for sid in dict.fromkeys(r['sample_id'] for r in rows):
                selected=sorted((r for r in rows if r['version']==version and r['arm']==arm and r['sample_id']==sid),key=lambda r:r['ratio'])
                for a,b in zip(selected,selected[1:]):
                    e_gain.append(dict(sample_id=sid,dataset=b['dataset'],version=version,arm=arm,
                        low=a['ratio'],high=b['ratio'],added_bytes=b['bytes']-a['bytes'],
                        metric_delta={k:b[k]-a[k] for k in METRICS}))
    historical=read(REVISION/'visual_ablation_report/summary.json')
    # Joined historical report stores rows under a different section in some versions.
    old=read(REVISION/'visual_uf_rate_extension/summary.json')['rows']
    old+= [r for r in read(REVISION/'visual_zero_E/summary.json')['rows'] if r['point'].endswith('_e0_g8')]
    matched=comparisons(rows,old);plot(groups,historical['group_means'],output)
    pictures=fixed_visuals(root,rows,old,output/'fixed_visuals')
    training={arm:dict(old=read(REVISION/'visual_router'/arm/'complete.json'),
                       new=read(root/'router'/arm/'complete.json')) for arm in ARMS}
    result=dict(complete=True,rows=rows,group_means=groups,comparisons=matched,E_prefix_changes=e_gain,
        training=training,fixed_visuals=pictures,only_Router_retrained=True,semantic_supervision=False,
        scope='13 historically used windows; REDS full-view resized, UVG crop; not full independent test set',
        runtime_scope='fresh process with loads, historical E0 unchanged; no consumer-GPU claim')
    save(output/'summary.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=OUTPUT)
    p.add_argument('--output',type=Path,default=OUTPUT/'report');p.add_argument('--wait',action='store_true');args=p.parse_args()
    if not os.environ.get('TMUX'):raise RuntimeError('formal report requires tmux')
    from demo.chunk_enhancement_experiment import Run
    run=Run(SimpleNamespace(output=args.output,command='light_report',max_hours=48));run.thread.start()
    try:
        while not (args.root/'evaluation/complete.json').exists():
            if not args.wait:raise ValueError('evaluation incomplete')
            run.update(phase='waiting_for_evaluation');run.check();run.stop.wait(30)
        done=args.output/'complete.json'
        if done.exists():
            value=read(done);verify_artifacts(args.output,value['artifacts'])
            if value['code_sha256']!=digest(__file__):raise ValueError('report source changed')
            for path,sha in value['inputs'].items():
                if digest(path)!=sha:raise ValueError('report input changed')
            print('LIGHT_REPORT_VERIFIED_READ_ONLY',flush=True);return
        inputs=[args.root/'evaluation/complete.json',args.root/'evaluation/protocol.json',args.root/'router/complete.json',
                REVISION/'visual_ablation_report/summary.json',REVISION/'visual_uf_rate_extension/summary.json',
                REVISION/'visual_zero_E/summary.json']
        binding={str(p):digest(p) for p in inputs};immutable(args.output/'inputs.json',binding)
        run.update(phase='read_only_report');result=build(args.root,args.output)
        names=['summary.json','inputs.json','rd_global_local.png','rd_local.png']
        names += [str(Path(r['path']).relative_to(args.output)) for r in result['fixed_visuals']]
        save(done,dict(complete=True,code_sha256=digest(__file__),inputs=binding,
                      artifacts={n:digest(args.output/n) for n in names}))
        run.update(phase='complete',completed=156,total=156)
    finally:
        run.log_resources();run.stop.set();run.thread.join(timeout=3);run.lock.close()


if __name__=='__main__':main()
