"""CPU report for completed, matched-receiver asymmetric sender diagnostics."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

from demo import routervc_sender_evaluate as evaluation
from demo.routervc_fullview_probe import read, digest, save, verify_artifacts

METRICS=('bpp','lpips_alex','psnr_db','temporal_delta_mae')
LABELS={'fixed_old_E':'Old E selection + fixed Rg', 'source':'Source-aware Rs + fixed Rg',
        'zero_source':'Zero-source Rs + fixed Rg'}
COLORS={'fixed_old_E':'#777777','source':'#276fbf','zero_source':'#ae4b82'}


def aggregate(rows, baselines):
    groups={}
    for dataset in ('REDS','UVG'):
        selected=[r for r in rows if r['dataset']==dataset]
        refs=[r for r in baselines if r['dataset']==dataset]
        group={}
        for point in dict.fromkeys(r['point'] for r in selected+refs):
            values=[r for r in selected+refs if r['point']==point]
            group[point]=dict(windows=len(values),**{
                k:sum(r[k] for r in values)/len(values) for k in METRICS})
        groups[dataset]=group
    return groups


def collect(root):
    folder=root/'evaluation';p=read(folder/'protocol.json')
    complete=read(folder/'complete.json');verify_artifacts(folder,complete['artifacts'])
    if p['code']!=evaluation.code_hashes() or p['models']!=evaluation.completed_models(root):
        raise ValueError('sender evaluation code/model binding changed')
    rows=[];inputs={str(folder/n):digest(folder/n) for n in ('protocol.json','complete.json','summary.json')}
    for entry in p['sources']:
        sample=entry['sample'];sid=sample['sample_id']
        evaluation.verify_sample(folder,p,sample)
        inputs[p['inputs'][sid]['source_path']]=p['inputs'][sid]['source_sha256']
        for point in p['points']:
            out=folder/'samples'/sid/point['name']
            r=evaluation.check_point(out,p,sample);encoded=read(out/'encode.json');d=r['decode']
            rows.append(dict(sample_id=sid,dataset=sample['dataset'],point=point['name'],
                arm=point['arm'],ratio=point['ratio'],bytes=r['bytes'],bpp=r['bpp'],
                **{k:r['quality'][k] for k in METRICS[1:]},Goff=r['same_wire_G_off_quality'],
                E_indices=encoded['ledger']['indices'],G_indices=d['route']['indices'],
                receiver_seconds=d['seconds'],policy_seconds=d['policy_seconds'],
                worker_peak_GiB=d['peak_cuda_allocated_bytes']/2**30,
                order_allocation_seconds=encoded['ledger']['allocation_seconds'],output_folder=str(out)))
            inputs[str(out/'result.json')]=digest(out/'result.json')
    old=Path(p['receiver']['complete_path']).parents[2]/'report'
    verify_artifacts(old,read(old/'complete.json')['artifacts'])
    baselines=read(old/'summary.json')['baselines']
    inputs[str(old/'complete.json')]=digest(old/'complete.json')
    inputs[str(old/'summary.json')]=digest(old/'summary.json')
    return p,rows,baselines,inputs


def figures(p,rows,baselines,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator,FuncFormatter,NullLocator
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    groups=aggregate(rows,baselines)
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for i,dataset in enumerate(('REDS','UVG')):
        group=groups[dataset]
        curves=[('Native DCVC-UF','#333333','o','-',
                 [group[f'uf_qp{q}'] for q in (8,16,24,32,40,48,56)])]
        for arm,marker,style in zip(evaluation.ARMS,('s','o','^'),('--','-',':')):
            curves.append((LABELS[arm],COLORS[arm],marker,style,
                           [group[f'{arm}_e{r:g}_g8'] for r in evaluation.RATIOS]))
        curves.append(('Whole-frame G (historical)','#23905a','*','None',[group['wholeframe_g_one_roi']]))
        for j,key in enumerate(('lpips_alex','psnr_db')):
            ax=axes[i,j]
            for label,color,marker,style,values in curves:
                ax.plot([v['bpp'] for v in values],[v[key] for v in values],label=label,
                        color=color,marker=marker,linestyle=style)
            ax.set_xscale('log');ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_major_locator(FixedLocator([.007,.015,.03,.07,.15] if dataset=='REDS' else [.004,.008,.016,.032]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:g}'))
            ax.grid(alpha=.2);ax.set_xlabel('Actual complete-stream bpp (log scale)')
            ax.set_ylabel('LPIPS (lower)' if j==0 else 'PSNR / dB (higher)')
            ax.set_title('REDS / 6 resized full views' if dataset=='REDS' else 'UVG / 7 existing crops')
    fig.suptitle('Asymmetric sender comparison: fixed Rg / UF / E / G')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=2,bbox_to_anchor=(.5,.035),frameon=False)
    fig.text(.5,.01,'Same E byte caps, possibly different realized rates. Diagnostic windows, not a full benchmark.',ha='center',fontsize=9)
    fig.subplots_adjust(top=.92,bottom=.22,hspace=.34,wspace=.23)
    fig.savefig(output/'rd_senders.png',dpi=160);plt.close(fig)
    visuals=output/'fixed_visuals';visuals.mkdir(exist_ok=True);pictures=[]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    state_colors=('#dddddd','#90c2e7','#f1c072','#bb9cce');state_names=('B','E','G','EG')
    for sid in dict.fromkeys(r['sample_id'] for r in rows):
        pts=[next(r for r in rows if r['sample_id']==sid and r['arm']==a and r['ratio']==.5)
             for a in evaluation.ARMS]
        with np.load(p['inputs'][sid]['source_path'],allow_pickle=False) as data:
            source=data['source'][8].copy()
        images=[('Source',source,None,None)]
        for row in pts:
            with np.load(Path(row['output_folder'])/'fresh/reconstruction.npz',allow_pickle=False) as data:
                if row['arm']=='fixed_old_E':
                    images.append(('Base B',data['base'][8].copy(),None,None))
                images.append((LABELS[row['arm']],data['reconstruction'][8].copy(),row['lpips_alex'],row['bpp']))
                if row['arm']=='source':
                    enhanced=data['enhanced'][8].copy();g_off=row['Goff']['lpips_alex'];bpp=row['bpp']
        images.append(('Source-aware E / G off',enhanced,g_off,bpp))
        width=300;height=round(source.shape[0]*width/source.shape[1])
        canvas=Image.new('RGB',(6*width,height+115),'white');draw=ImageDraw.Draw(canvas)
        draw.text((8,5),sid+' / fixed frame 8 / E50 cap / actual bytes may differ',font=font,fill='black')
        for i,(label,pixels,score,rate) in enumerate(images):
            draw.text((i*width+5,30),label,font=font,fill='black')
            if score is not None:
                draw.text((i*width+5,54),f'LPIPS {score:.4f}',font=font,fill='black')
                draw.text((i*width+5,76),f'Actual bpp {rate:.6f}',font=font,fill='black')
            canvas.paste(Image.fromarray(pixels).resize((width,height),Image.Resampling.LANCZOS),(i*width,115))
        image_path=visuals/f'{sid}.png';canvas.save(image_path)
        fig,axes=plt.subplots(1,3,figsize=(11,4.4))
        for ax,row in zip(axes,pts):
            states=[int(i in row['E_indices'])+2*int(i in row['G_indices']) for i in range(16)]
            ax.imshow(np.array(states).reshape(4,4),cmap=ListedColormap(state_colors),vmin=0,vmax=3)
            for i,value in enumerate(states):
                ax.text(i%4,i//4,f'{i:02d}\n{state_names[value]}',ha='center',va='center')
            ax.set_xticks([]);ax.set_yticks([])
            ax.set_title(f'{LABELS[row["arm"]]}\nLPIPS {row["lpips_alex"]:.4f} / bpp {row["bpp"]:.5f}',fontsize=9)
        fig.suptitle(sid+' / sender E selections and receiver G selections',fontsize=10)
        fig.legend(handles=[Patch(facecolor=c,label=n) for c,n in zip(state_colors,state_names)],loc='lower center',ncol=4,frameon=False)
        fig.subplots_adjust(top=.78,bottom=.17,wspace=.14)
        routes=visuals/f'{sid}_routes.png';fig.savefig(routes,dpi=160);plt.close(fig)
        pictures.append(dict(sample_id=sid,image=str(image_path),routes=str(routes),fixed_frame=8,
            selection='all 13 diagnostic windows, E50 cap, no quality-based frame selection'))
    return groups,pictures


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=evaluation.ROOT)
    p.add_argument('--output',type=Path)
    p.add_argument('--wait',action='store_true')
    args=p.parse_args();args.output=args.output or args.root/'report'
    if not os.environ.get('TMUX'):
        raise RuntimeError('sender CPU report requires tmux')
    from demo.chunk_enhancement_experiment import Run
    run=Run(SimpleNamespace(output=args.output,command='sender_report',max_hours=96));run.thread.start()
    try:
        while not (args.root/'evaluation/complete.json').exists():
            if not args.wait:raise ValueError('sender evaluation incomplete')
            run.check();run.update(phase='waiting_for_sender_evaluation',holds_GPU_mutex=False);run.stop.wait(30)
        done=args.output/'complete.json'
        if done.exists():
            value=read(done);verify_artifacts(args.output,value['artifacts'])
            if value['code_sha256']!=digest(__file__):raise ValueError('sender report code changed')
            for path,sha in value['inputs'].items():
                if digest(path)!=sha:raise ValueError('sender report input changed')
            print('SENDER_REPORT_VERIFIED_READ_ONLY',flush=True);return
        protocol,rows,baselines,inputs=collect(args.root)
        groups,pictures=figures(protocol,rows,baselines,args.output)
        save(args.output/'summary.json',dict(complete=True,rows=rows,baselines=baselines,
            group_means=groups,pictures=pictures,independent_system_test=False,
            semantic_supervision=False,on_policy_adaptation=False,
            rate_scope='equal E byte caps do not guarantee equal realized rates; all container bytes charged',
            timing_scope='fresh receiver time; R_s allocation of authenticated candidates is NOT full source encoding time',
            baseline_scope='UF and whole-frame G are authenticated historical results; whole-frame G used a different seed',
            sender_selection='no automatic winner or on-policy promotion from these 13 windows'))
        names=['summary.json','rd_senders.png']+[str(Path(pic[k]).relative_to(args.output))
            for pic in pictures for k in ('image','routes')]
        save(done,dict(complete=True,code_sha256=digest(__file__),inputs=inputs,
            artifacts={n:digest(args.output/n) for n in names}))
        print('SENDER_REPORT_COMPLETE',flush=True)
    finally:
        run.log_resources();run.stop.set();run.thread.join(timeout=3);run.lock.close()


if __name__=='__main__':main()
