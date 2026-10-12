"""Actual small-width RD, budget replay, fixed frames and resource figures."""
from collections import Counter
from pathlib import Path
import json
import numpy as np
from demo.routervc_fullview_probe import read,save
from tools.cooperative_review import source_rgb
from tools.cooperative_sender_review_report import pyplot
from tools.plot_latent_diagnostics import comparison_image


def report(root,protocol,check=lambda:None):
    from tools.precision_pilot import point_folder,bank_folder,RATIOS,verify_point
    points=[]
    for row in protocol['rows']:
        for width in protocol['widths']:
            for cap in RATIOS:
                point=read(point_folder(root,row,width,cap)/'point.json');verify_point(point);points.append(point)
    evaluations=[p for p in points if p['cohort']=='internal_evaluation']
    lookup={(p['sample_id'],f'w{p["width"]}_e{round(p["cap"]*100):03d}'):p for p in evaluations}
    groups=[]
    for cohort in ('calibration','internal_evaluation'):
        for dataset in ('REDS','UVG'):
            for width in protocol['widths']:
                for cap in RATIOS:
                    rows=[p for p in points if (p['cohort'],p['dataset'],p['width'],p['cap'])==(cohort,dataset,width,cap)]
                    groups.append(dict(cohort=cohort,dataset=dataset,width=width,cap=cap,count=len(rows),
                        bpp=float(np.mean([p['bpp'] for p in rows])),base_bytes=float(np.mean([p['base_bytes'] for p in rows])),
                        quality={k:float(np.mean([p['quality'][k] for p in rows])) for k in ('lpips_alex','psnr_db','temporal_delta_mae')},
                        received_quality={k:float(np.mean([p['received_quality'][k] for p in rows])) for k in ('lpips_alex','psnr_db')},
                        base_quality={k:float(np.mean([p['base_quality'][k] for p in rows])) for k in ('lpips_alex','psnr_db')},
                        seconds=float(np.mean([p['fresh_seconds'] for p in rows])),
                        peak_GiB=max(p['peak_GiB'] for p in rows),reserved_GiB=max(p['reserved_GiB'] for p in rows)))
    replay=[]
    for row in protocol['rows']:
        if row['cohort']!='internal_evaluation':continue
        decision=read(root/'decisions'/f'{row["sample_id"]}.json')
        for level,d in enumerate(decision['decisions']):
            feasible=[p for p in evaluations if p['sample_id']==row['sample_id'] and p['actual_bytes']<=d['budget_bytes']]
            oracle=min(feasible,key=lambda p:(p['quality']['lpips_alex'],p['actual_bytes'])) if feasible else None
            keys={'automatic':d['automatic']['key'],'base_fraction':d['base_fraction'],
                  **{'fixed'+w:key for w,key in d['fixed'].items()},
                  'offline_grid_oracle':None if oracle is None else f'w{oracle["width"]}_e{round(oracle["cap"]*100):03d}'}
            scores={}
            for method,key in keys.items():
                p=lookup[row['sample_id'],key] if key is not None else None
                scores[method]=None if p is None else dict(key=key,width=p['width'],cap=p['cap'],
                    bpp=p['bpp'],actual_bytes=p['actual_bytes'],quality=p['quality'])
            replay.append(dict(sample_id=row['sample_id'],dataset=row['dataset'],level=level,
                budget_bytes=d['budget_bytes'],scores=scores,oracle_uses_ground_truth=True))
    plt=pyplot();colors={3:'#2474b5',5:'#df8824',7:'#298d62'}; names=[]
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for i,dataset in enumerate(('REDS','UVG')):
        for j,(metric,title) in enumerate((('lpips_alex','LPIPS (lower)'),('psnr_db','PSNR (higher)'))):
            ax=axes[i,j]
            for w in protocol['widths']:
                rows=sorted([g for g in groups if g['cohort']=='internal_evaluation' and g['dataset']==dataset and g['width']==w],key=lambda g:g['cap'])
                x=[g['bpp'] for g in rows]
                ax.plot(x,[g['quality'][metric] for g in rows],'-o',color=colors[w],label=f'w{w} + local G8')
                ax.plot(x,[g['received_quality'][metric] for g in rows],'--',alpha=.6,color=colors[w],label=f'w{w}, G off')
                for xx,g in zip(x,rows):ax.annotate(str(round(g['cap']*100)),(xx,g['quality'][metric]),fontsize=7)
            ax.set_title(dataset+(' resized full views' if dataset=='REDS' else ' HoneyBee crops'))
            ax.set_xlabel('Actual complete-file bits/pixel');ax.set_ylabel(title);ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('P3 internal evaluation | 3 REDS windows + 3 HoneyBee crops\nFixed width3-trained routers; labels: E byte cap %, NOT equal rates')
    fig.savefig(root/'width_rd.png',dpi=150);plt.close(fig);names.append('width_rd.png')
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    methods=('fixed3','fixed5','fixed7','base_fraction','automatic','offline_grid_oracle')
    budget_summary=[]
    for method in methods:
        rates=[];quality=[];coverage=[]
        for level in range(6):
            rows=[r for r in replay if r['level']==level]
            ok=[r['scores'][method] for r in rows if r['scores'][method] is not None]
            coverage.append(len(ok)/len(rows))
            # Quality/rate only on common delivered rows across ALL methods.
            common=[r for r in rows if all(r['scores'][m] is not None for m in methods)]
            rates.append(float(np.mean([r['scores'][method]['bpp'] for r in common])) if common else np.nan)
            quality.append(float(np.mean([r['scores'][method]['quality']['lpips_alex'] for r in common])) if common else np.nan)
            budget_summary.append(dict(method=method,level=level,delivered=len(ok),total=len(rows),common=len(common),
                bpp=None if not common else rates[-1],lpips=None if not common else quality[-1]))
        style='--' if method=='offline_grid_oracle' else '-'
        axes[0].plot(range(6),coverage,style+'o',label=method)
        axes[1].plot(range(6),quality,style+'o',label=method)
        axes[2].plot(rates,quality,style+'o',label=method)
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=7)
    axes[0].set(xlabel='Budget level (same bytes per window)',ylabel='Delivered fraction',ylim=(-.05,1.05))
    axes[1].set(xlabel='Budget level',ylabel='LPIPS, common delivered windows')
    axes[2].set(xlabel='Actual bpp, common delivered windows',ylabel='LPIPS')
    fig.suptitle('Train-only table vs fixed widths | sparse four-cap grid\nHindsight oracle is a reference, NOT the deployed selector; missing B is not a quality point')
    fig.savefig(root/'budget_replay.png',dpi=150);plt.close(fig);names.append('budget_replay.png')
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for i,(dataset,color) in enumerate((('REDS','#2474b5'),('UVG','#df8824'))):
        rows=[p for p in evaluations if p['dataset']==dataset]
        for j,(field,title,reduce) in enumerate((('fresh_seconds','Fresh decode seconds (mean)',np.mean),
                    ('peak_GiB','CUDA allocated GiB (max)',max),('reserved_GiB','CUDA reserved GiB (max)',max))):
            axes[j].bar(np.arange(3)+i*.35,[reduce([p[field] for p in rows if p['width']==w]) for w in (3,5,7)],width=.35,label=dataset,color=color)
            axes[j].set_xticks(np.arange(3)+.175,['width3','width5','width7']);axes[j].set_title(title);axes[j].legend();axes[j].grid(axis='y',alpha=.2)
    fig.suptitle('Receiver only; excludes sender preparation, saving and CPU quality scoring')
    fig.savefig(root/'resources.png',dpi=150);plt.close(fig);names.append('resources.png')
    galleries=[]
    for row in protocol['rows']:
        if row['cohort']!='internal_evaluation':continue
        check();source=source_rgb(row); base_panels=[('Source',source[8])]; final_panels=[('Source',source[8])]
        decision=read(root/'decisions'/f'{row["sample_id"]}.json')['decisions'][3]
        for w in protocol['widths']:
            p=lookup[row['sample_id'],f'w{w}_e000']
            with np.load(Path(p['receive'])/'pixels.npz',allow_pickle=False) as z:
                base_panels.append((f'B w{w}: {p["base_bytes"]} bytes',z['base'][8].copy()))
            key=decision['fixed'][str(w)]
            if key is not None:
                p=lookup[row['sample_id'],key]
                with np.load(Path(p['receive'])/'pixels.npz',allow_pickle=False) as z:
                    final_panels.append((f'w{w}: {p["actual_bytes"]}B LP {p["quality"]["lpips_alex"]:.3f}',z['reconstruction'][8].copy()))
        folder=root/'figures';folder.mkdir(exist_ok=True)
        for suffix,panels,title in (('base',base_panels,'Actual B clarity, G off'),
             ('final',final_panels,f'Same byte ceiling {decision["budget_bytes"]}; automatic: {decision["automatic"]["key"]}')):
            name=f'figures/{row["sample_id"]}_{suffix}.png'
            comparison_image(panels,root/name,row['sample_id']+' | frame 8 | '+title);names.append(name)
            galleries.append(dict(sample_id=row['sample_id'],kind=suffix,path=name))
    beats=[json.loads(x) for x in (root/'queue/heartbeat.jsonl').read_text().splitlines()] if (root/'queue/heartbeat.jsonl').exists() else []
    device_peaks=[float(b['gpu'].split(',')[2]) for b in beats if isinstance(b.get('gpu'),str) and b['gpu'].count(',')==4]
    preparation=[]
    for row in protocol['rows']:
        preparation.append(dict(sample_id=row['sample_id'],dataset=row['dataset'],
            three_width_encode_seconds=sum(read(bank_folder(root,row,w)/'encoded.json')['seconds'] for w in (3,5,7)),
            three_width_order_seconds=sum(read(bank_folder(root,row,w)/'order.json')['seconds'] for w in (3,5,7))))
    summary=dict(complete=True,groups=groups,budget_replay=replay,budget_summary=budget_summary,galleries=galleries,
        preparation=preparation,logical_points=len(points),new_decodes=sum(not p['reused_identical_stream'] for p in points),
        sampled_device_peak_MiB=max(device_peaks) if device_peaks else None,
        queue_seconds=beats[-1]['elapsed_seconds'] if beats else None,final_disks=beats[-1]['disks'] if beats else None,
        limitation='14 development windows, evaluation reuses model-selection sequences; zero-shot multi-width, four-cap grid; no full UVG or realtime claim',
        no_evaluation_quality_in_selector=True,no_optimization=True)
    save(root/'summary.json',summary);names.append('summary.json');return names
