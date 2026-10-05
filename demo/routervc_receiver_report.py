"""CPU-only fixed-E Rg report: authentic measured curves, images and decisions."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

from demo import routervc_receiver_evaluate as evaluation
from demo.routervc_fullview_probe import read, digest, save, immutable, verify_artifacts
from demo.routervc_visual_report import save_plot

METRICS = evaluation.METRICS


def mean(rows, keys):
    if not rows:
        raise ValueError('empty receiver report group')
    return {key:sum(r[key] for r in rows)/len(rows) for key in keys}


def collect(root):
    protocol=read(root/'evaluation/protocol.json')
    verify_artifacts(root/'evaluation',read(root/'evaluation/complete.json')['artifacts'])
    if protocol['code'] != evaluation.code_hashes():
        raise ValueError('fixed-E evaluator code changed')
    if evaluation.completed_models(Path(protocol['training'])) != protocol['models']:
        raise ValueError('receiver training selection changed')
    rows, baseline, inputs = [], [], {}
    for entry in protocol['sources']:
        sample=entry['sample'];sid=sample['sample_id']
        item=protocol['inputs'][sid]
        if digest(item['source_path']) != item['source_sha256']:
            raise ValueError('report source changed')
        inputs[item['source_path']]=item['source_sha256']
        for point in protocol['points']:
            folder=root/'evaluation/samples'/sid/point['name']
            value=evaluation.check_point(folder,protocol,sample)
            ref,old,encoded=evaluation.reference(protocol,sid,point['ratio'],verify=False)
            report=value['decode']; runtime=report['generation_runtime']
            inputs[str(ref/'result.json')]=digest(ref/'result.json')
            inputs[str(folder/'result.json')]=digest(folder/'result.json')
            rows.append(dict(sample_id=sid,dataset=sample['dataset'],point=point['name'],
                arm=point['arm'],ratio=point['ratio'],bytes=value['bytes'],bpp=value['bpp'],
                **{k:value['quality'][k] for k in METRICS},Goff=value['same_wire_G_off_quality'],
                receiver_seconds=report['seconds'],policy_seconds=report['policy_seconds'],
                G_seconds=runtime['seconds_model_load_excluded'] if runtime else 0.,
                G_peak_GiB=max((w['runtime']['peak_cuda_allocated_bytes'] for w in runtime['windows']),default=0)/2**30 if runtime else 0.,
                worker_peak_GiB=report['peak_cuda_allocated_bytes']/2**30,
                E_indices=encoded['plan']['selected_indices'],G_indices=report['route']['indices'],
                E_packet_bytes=report['packet_bytes'],header_bytes=report['generation_control_bytes'],
                reused=value['reused'],fixed_E_reference=str(ref),output_folder=str(ref if value['reused'] else folder),
                noise_pairing=value.get('noise_pairing')))
        names=[f'uf_qp{q}' for q in (8,16,24,32,40,48,56)]+['wholeframe_g_one_roi']
        for name in names:
            base=(evaluation.REVISION/'visual_uf_rate_extension' if name in ('uf_qp40','uf_qp48','uf_qp56')
                  else evaluation.REVISION/'visual_evaluation_recovered')
            folder=base/'samples'/sid/name
            value=read(folder/'result.json');verify_artifacts(folder,value['artifacts'])
            if value['protocol_sha256'] != digest(base/'protocol.json') or value['sample_id'] != sid:
                raise ValueError('UF/full-G reference binding changed')
            stream=folder/('stream.acsg' if name=='wholeframe_g_one_roi' else 'stream.bin')
            if value['bytes'] != stream.stat().st_size:
                raise ValueError('baseline measured on-disk bytes changed')
            inputs[str(folder/'result.json')]=digest(folder/'result.json')
            baseline.append(dict(sample_id=sid,dataset=sample['dataset'],point=name,
                bytes=value['bytes'],bpp=value['bpp'],**{k:value['quality'][k] for k in METRICS},
                reused=True,reference=str(folder),scope='authenticated historical native-UF/full-G, not fresh timing'))
    return protocol,rows,baseline,inputs


def aggregate(rows, baselines):
    groups={}
    for dataset in ('REDS','UVG','all_13_descriptive'):
        chosen=[r for r in rows if dataset=='all_13_descriptive' or r['dataset']==dataset]
        refs=[r for r in baselines if dataset=='all_13_descriptive' or r['dataset']==dataset]
        group={}
        for name in dict.fromkeys(r['point'] for r in chosen):
            values=[r for r in chosen if r['point']==name]
            group[name]=dict(windows=len(values),**mean(values,('bytes','bpp',*METRICS,
                'receiver_seconds','policy_seconds','G_seconds','G_peak_GiB','worker_peak_GiB')))
        for name in dict.fromkeys(r['point'] for r in refs):
            values=[r for r in refs if r['point']==name]
            group[name]=dict(windows=len(values),**mean(values,('bytes','bpp',*METRICS)))
        groups[dataset]=group
    pairs=[]
    for row in rows:
        if row['arm']=='shared':continue
        old=next(r for r in rows if r['sample_id']==row['sample_id'] and r['ratio']==row['ratio'] and r['arm']=='shared')
        if row['E_indices'] != old['E_indices'] or row['E_packet_bytes'] != old['E_packet_bytes']:
            raise ValueError('fixed-E report compared different packet selections')
        pairs.append(dict(sample_id=row['sample_id'],dataset=row['dataset'],arm=row['arm'],ratio=row['ratio'],
            delta={k:row[k]-old[k] for k in ('bytes','bpp',*METRICS,'receiver_seconds','G_peak_GiB','worker_peak_GiB')},
            changed_G=row['G_indices']!=old['G_indices']))
    paired={dataset:{arm:{f'e{ratio:g}':dict(windows=len(values),
        lower_lpips=sum(v['delta']['lpips_alex']<0 for v in values),
        changed_G=sum(v['changed_G'] for v in values),
        mean_delta=mean([v['delta'] for v in values],('bytes','bpp',*METRICS,'receiver_seconds','G_peak_GiB','worker_peak_GiB')))
        for ratio in evaluation.RATIOS for values in [[p for p in pairs if p['arm']==arm and p['ratio']==ratio
            and (dataset=='all_13_descriptive' or p['dataset']==dataset)]]}
        for arm in evaluation.ARMS} for dataset in groups}
    return groups,dict(rows=pairs,groups=paired,scope='same measured E payload; any wrapper-size delta charged; no BD-rate',
                      dependence='three operating points share each video; 13 diagnostic windows, not independent full tests')


def figures(groups, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8.4))
    for i,dataset in enumerate(('REDS','UVG')):
        values=groups[dataset]
        curves=[('Native DCVC-UF','#333333','o',[values[f'uf_qp{q}'] for q in (8,16,24,32,40,48,56)]),
                ('Old shared Router / fixed E','#777777','s',[values[f'shared_e{r:g}_g8'] for r in evaluation.RATIOS]),
                ('Independent Rg / core','#276fbf','o',[values[f'core_e{r:g}_g8'] for r in evaluation.RATIOS]),
                ('Independent Rg / halo','#ae4b82','^',[values[f'halo_e{r:g}_g8'] for r in evaluation.RATIOS]),
                ('Base + whole-frame G','#23905a','*',[values['wholeframe_g_one_roi']])]
        for j,key in enumerate(METRICS[:2]):
            ax=axes[i,j]
            for title,color,marker,points in curves:
                ax.plot([p['bpp'] for p in points],[p[key] for p in points],color=color,marker=marker,
                        linestyle='-' if len(points)>1 else 'None',label=title)
            ax.set_xscale('log');ax.grid(alpha=.2,which='both');ax.set_xlabel('Actual complete-stream bpp')
            ax.set_ylabel('LPIPS (lower)' if j==0 else 'PSNR dB (higher)')
            ax.set_title('REDS / 6 resized full views' if dataset=='REDS' else 'UVG / 7 existing crops')
    fig.suptitle('Receiver-only adaptation: identical E packets, frozen UF/E/G')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=3,bbox_to_anchor=(.5,.045),frameon=False)
    fig.text(.5,.012,'Lines join measured means. Historical controls reused; changed routes do not universally share pixelwise diffusion noise.',ha='center',fontsize=8)
    fig.subplots_adjust(top=.92,bottom=.20,hspace=.32,wspace=.24)
    save_plot(fig,output/'rd_fixed_E.png')


def visuals(protocol, rows, output):
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    output.mkdir(exist_ok=True);records=[]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    colors=('#dddddd','#90c2e7','#f1c072','#bb9cce');labels=('B','E','G','EG')
    for sid in dict.fromkeys(r['sample_id'] for r in rows):
        points=[next(r for r in rows if r['sample_id']==sid and r['arm']==arm and r['ratio']==.5)
                for arm in ('shared','core','halo')]
        with np.load(protocol['inputs'][sid]['source_path'],allow_pickle=False) as data:
            source=data['source'][8].copy()
        with np.load(Path(points[0]['output_folder'])/'fresh/reconstruction.npz',allow_pickle=False) as data:
            images=[('Source',source,None),('Base B',data['base'][8].copy(),None),
                    ('Received Y / G off',data['enhanced'][8].copy(),points[0]['Goff']['lpips_alex']),
                    ('Old shared Router',data['reconstruction'][8].copy(),points[0]['lpips_alex'])]
        for title,row in zip(('Independent Rg / core','Independent Rg / halo'),points[1:]):
            with np.load(Path(row['output_folder'])/'fresh/reconstruction.npz',allow_pickle=False) as data:
                images.append((title,data['reconstruction'][8].copy(),row['lpips_alex']))
        width=300;height=round(source.shape[0]*width/source.shape[1])
        canvas=Image.new('RGB',(width*6,height+90),'white');draw=ImageDraw.Draw(canvas)
        draw.text((8,5),sid+' / E50 fixed packets, G8 / fixed frame 8 / display resized',fill='black',font=font)
        for i,(title,pixels,lpips) in enumerate(images):
            draw.text((i*width+5,30),title,fill='black',font=font)
            if lpips is not None:draw.text((i*width+5,53),f'LPIPS {lpips:.4f}',fill='black',font=font)
            canvas.paste(Image.fromarray(pixels).resize((width,height),Image.Resampling.LANCZOS),(i*width,90))
        image_path=output/f'{sid}.png';canvas.save(image_path)
        fig,axes=plt.subplots(1,3,figsize=(10.4,4.3));states=[]
        for ax,row,title in zip(axes,points,('Old shared Router','Independent Rg / core','Independent Rg / halo')):
            values=[int(i in row['E_indices'])+2*int(i in row['G_indices']) for i in range(16)];states.append(values)
            ax.imshow(np.array(values).reshape(4,4),cmap=ListedColormap(colors),vmin=0,vmax=3)
            for i,value in enumerate(values):ax.text(i%4,i//4,f'{i:02d}\n{labels[value]}',ha='center',va='center')
            ax.set_xticks([]);ax.set_yticks([]);ax.set_title(f'{title}\nLPIPS {row["lpips_alex"]:.4f}')
        fig.suptitle(sid+' / same E locations; only receiver G decisions change',fontsize=10)
        fig.legend(handles=[Patch(facecolor=c,label=n) for c,n in zip(colors,labels)],loc='lower center',ncol=4,frameon=False)
        fig.text(.5,.09,'B: neither | E: enhance only | G: generate only | EG: both. No region mask is transmitted.',ha='center',fontsize=8)
        fig.subplots_adjust(top=.80,bottom=.20,wspace=.16)
        route_path=output/f'{sid}_routes.png';save_plot(fig,route_path)
        records.append(dict(sample_id=sid,image=str(image_path),image_sha256=digest(image_path),
            routes=str(route_path),routes_sha256=digest(route_path),states=states,fixed_frame=8,
            selection='all 13 windows, fixed E50 budget; not cherry-picked by quality'))
    return records


def build(root, output):
    protocol,rows,baselines,inputs=collect(root)
    groups,comparisons=aggregate(rows,baselines)
    figures(groups,output);pictures=visuals(protocol,rows,output/'fixed_visuals')
    preferred=min(evaluation.ARMS,key=lambda a:(protocol['models']['arms'][a]['validation']['regret'],
                                               protocol['models']['arms'][a]['validation']['loss']))
    result=dict(complete=True,rows=rows,baselines=baselines,group_means=groups,comparisons=comparisons,
        pictures=pictures,validation_preferred_receiver=preferred,
        selection_scope='TRAIN-held-out validation ranks candidates; 13 real-stream diagnostic windows inform discussion, not automatic promotion',
        sender_training_complete=False,independent_system_test=False,semantic_supervision=False,
        memory_scope='G maximum over all calls; worker maximum includes resets; neither implies consumer-device compatibility',
        timing_scope='new fresh workers with loads versus authenticated historical fresh shared workers; no sender time claim',
        noise_scope=protocol['noise_scope'])
    save(output/'summary.json',result)
    return result,inputs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=evaluation.ROOT)
    p.add_argument('--output',type=Path,default=evaluation.ROOT/'report')
    p.add_argument('--wait',action='store_true');args=p.parse_args()
    if not os.environ.get('TMUX'):raise RuntimeError('receiver report requires tmux')
    from demo.chunk_enhancement_experiment import Run
    run=Run(SimpleNamespace(output=args.output,command='receiver_report',max_hours=48));run.thread.start()
    try:
        while not (args.root/'evaluation/complete.json').exists():
            if not args.wait:raise ValueError('receiver evaluation incomplete')
            run.check();run.update(phase='waiting_for_evaluation');run.stop.wait(30)
        done=args.output/'complete.json'
        if done.exists():
            value=read(done);verify_artifacts(args.output,value['artifacts'])
            if value['code_sha256']!=digest(__file__):raise ValueError('receiver report source changed')
            for path,sha in value['inputs'].items():
                if digest(path)!=sha:raise ValueError('receiver report input changed')
            print('RECEIVER_REPORT_VERIFIED_READ_ONLY',flush=True);return
        result,inputs=build(args.root,args.output)
        for path in (args.root/'evaluation/complete.json',args.root/'evaluation/protocol.json',
                     args.root/'formal/router/complete.json'):
            inputs[str(path)]=digest(path)
        immutable(args.output/'inputs.json',inputs)
        names=['summary.json','inputs.json','rd_fixed_E.png']
        names += [str(Path(item[key]).relative_to(args.output)) for item in result['pictures'] for key in ('image','routes')]
        save(done,dict(complete=True,code_sha256=digest(__file__),inputs=inputs,
                       artifacts={name:digest(args.output/name) for name in names}))
    finally:
        run.log_resources();run.stop.set();run.thread.join(timeout=3);run.lock.close()


if __name__=='__main__':main()
