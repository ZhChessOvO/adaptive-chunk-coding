"""Automatic bounded P2 review figures; scientific prose stays in Notion."""
from pathlib import Path
import numpy as np

from demo.routervc_fullview_probe import read, save, digest, verify_artifacts
from tools.latent_boundary_report import EVALUATION


def aggregate(points):
    groups=[]
    for cohort in ('diagnostic','grouped_validation'):
        for dataset in ('REDS','UVG'):
            subset=[p for p in points if p['cohort']==cohort and p['dataset']==dataset]
            budgets=(0.,.25,.5,1.) if cohort=='diagnostic' else (None,)
            for cap in budgets:
                rows=[p for p in subset if cap is None or p['cap']==cap]
                if not rows:raise ValueError('missing cohort/dataset/budget')
                arms={}
                for arm in ('old_policy','cooperative'):
                    scores=[p['scores'][arm] for p in rows]
                    arms[arm]=dict(samples=len(rows),
                        **{k:float(np.mean([s['quality'][k] for s in scores]))
                           for k in ('lpips_alex','psnr_db','temporal_delta_mae')},
                        **{k:float(np.mean([s[k] for s in scores])) for k in
                           ('bpp','actual_bytes','fresh_seconds','G_calls')},
                        max_GiB=max(s['peak_GiB'] for s in scores),
                        max_reserved_GiB=max(s['reserved_GiB'] for s in scores))
                differences=[p['scores']['cooperative']['quality']['lpips_alex']-
                             p['scores']['old_policy']['quality']['lpips_alex'] for p in rows]
                groups.append(dict(cohort=cohort,dataset=dataset,cap=cap,arms=arms,
                    delta_lpips=float(np.mean(differences)),
                    improved=int(np.sum(np.array(differences)<-1e-8)),
                    worsened=int(np.sum(np.array(differences)>1e-8))))
    return groups


def training_plot(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from routervc.cooperation.data import ROOT
    history=read(ROOT/'router/history.json')
    # Inspect the stored schema explicitly: no interpolation/missing epochs.
    epochs=history['epochs']
    fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
    for dataset in ('REDS','UVG'):
        ax.plot([e['epoch'] for e in epochs],
                [e['validation']['groups'][dataset]['regret'] for e in epochs],label=dataset)
    best=read(ROOT/'router/complete.json')['best']['epoch']
    ax.axvline(best,linestyle=':',color='k',label=f'Preselected epoch {best}')
    ax.set_xlabel('Epoch');ax.set_ylabel('Measured one-step regret (lower better)')
    ax.set_title('Training diagnostic only: NOT actual final-picture LPIPS or RD')
    ax.legend();ax.grid(alpha=.2);fig.savefig(root/'training_diagnostic.png',dpi=160);plt.close(fig)


def report(root,protocol,points,check=lambda:None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from tools.cooperative_review import location,source_rgb
    from tools.plot_latent_diagnostics import comparison_image
    if len(points)!=64 or len({(p['cohort'],p['sample_id'],p['cap']) for p in points})!=64:
        raise ValueError('incomplete paired review')
    prior=read(EVALUATION/'summary.json')
    if digest(EVALUATION/'summary.json')!=protocol['previous_summary']:raise ValueError('old summary changed')
    # Original native/G-off/feather points are reused, not silently re-measured.
    for p in prior['points']:
        if p['arm'] not in ('source','source_Goff','UF'):continue
        check();suffix=f'UF_q{p["budget"]}' if p['arm']=='UF' else f'{p["arm"]}_e{round(p["budget"]*100):03d}'
        folder=EVALUATION/'samples'/p['sample_id']/suffix/'receive'
        if digest(folder/'complete.json')!=p['receipt_sha256']:raise ValueError('old point receipt drift')
        done=read(folder/'complete.json');verify_artifacts(folder,done['artifacts'])
        if done['actual_bytes']!=p['actual_bytes']:raise ValueError('old bytes drift')
    groups=aggregate(points)
    save(root/'summary.json',dict(complete=True,groups=groups,points=points,
        scope=protocol,old_curves=prior['groups'],
        caveats=['Validation selected checkpoint; no new independent benchmark',
                 'Old policy converted to common profile; exact selection checked for all 64 states',
                 'Fresh timing includes codec/model load and fusion, excludes saving and metric computation',
                 'Boundary scores follow own selected edges; descriptive, not matched-edge causal estimates',
                 'No new sender optimization; no adaptive width; no semantic protection training']))
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for i,dataset in enumerate(('REDS','UVG')):
        rows=[g for g in groups if g['dataset']==dataset and g['cohort']=='diagnostic']
        for j,metric in enumerate(('lpips_alex','psnr_db')):
            ax=axes[i,j]
            for arm,label,style in [('old_policy','Old Rg + multiband','o--'),
                                    ('cooperative','P2 Rg + multiband','s-')]:
                ax.plot([r['arms'][arm]['bpp'] for r in rows],[r['arms'][arm][metric] for r in rows],style,label=label)
            for arm,label,style in [('UF','DCVC-UF (I32/P sweep)','k.-'),
                                    ('source','Old Rg + feather',':'),('source_Goff','B+E; no G','x--')]:
                values=list(prior['groups'][dataset][arm].values())
                ax.plot([v['bpp'] for v in values],[v[metric] for v in values],style,label=label)
            ax.set_title('REDS resized full view (6)' if dataset=='REDS' else 'UVG existing crops (7)')
            ax.set_xlabel('Actual whole-stream bpp');ax.set_ylabel(metric);ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Fixed sender / width3 / frozen G | 13 reused diagnostic windows, not full test sets')
    fig.savefig(root/'paired_rd.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for ax,cohort in zip(axes,('diagnostic','grouped_validation')):
        for dataset in ('REDS','UVG'):
            rows=[p for p in points if p['cohort']==cohort and p['dataset']==dataset]
            ax.scatter([p['scores']['old_policy']['quality']['lpips_alex'] for p in rows],
                       [p['scores']['cooperative']['quality']['lpips_alex'] for p in rows],label=dataset)
        lo,hi=ax.get_xlim();ax.plot([lo,hi],[lo,hi],'k:',linewidth=1)
        ax.set_title(cohort+' | below diagonal = improved')
        ax.set_xlabel('Old Rg + multiband LPIPS');ax.set_ylabel('P2 Rg + multiband LPIPS');ax.legend()
    fig.savefig(root/'paired_lpips.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,key,title in zip(axes,('peak_GiB','fresh_seconds'),('Peak CUDA allocated GiB','Mean fresh decode seconds')):
        labels=[];values=[]
        for dataset in ('REDS','UVG'):
            for arm in ('old_policy','cooperative'):
                scores=[p['scores'][arm][key] for p in points if p['cohort']=='diagnostic' and p['dataset']==dataset]
                values.append(max(scores) if key=='peak_GiB' else np.mean(scores));labels.append(dataset+'\n'+arm)
        ax.bar(range(4),values);ax.set_xticks(range(4),labels,fontsize=8);ax.set_ylabel(title)
        ax.set_ylim(0,max(values)*1.15)
        for i,v in enumerate(values):ax.text(i,v,f'{v:.2f}',ha='center',va='bottom')
    fig.suptitle('G <= 8 | process allocation, not device total | old policy uses common decoder profile')
    fig.savefig(root/'resources.png',dpi=160);plt.close(fig)
    names=['summary.json','paired_rd.png','paired_lpips.png','resources.png']
    for job in protocol['jobs']:
        if job['cohort']=='diagnostic' and job['cap']!=.5:continue
        check();folder=location(root,job);point=read(folder/'pair.json');source=source_rgb(job['row'])
        with np.load(folder/'old_policy/receive/pixels.npz') as z:received=z['enhanced'][8].copy()
        panels=[('Source',source[8]),('Received B+E (Y)',received)]
        if job['cohort']=='diagnostic':
            native=min([p for p in prior['points'] if p['sample_id']==job['row']['sample_id'] and p['arm']=='UF'],
                       key=lambda p:abs(p['bpp']-point['scores']['cooperative']['bpp']))
            with np.load(EVALUATION/'samples'/job['row']['sample_id']/f'UF_q{native["budget"]}'/'receive/pixels.npz') as z:
                panels.append((f'UF nearest: {native["bpp"]:.4f} bpp',z['reconstruction'][8].copy()))
        for arm in ('old_policy','cooperative'):
            score=point['scores'][arm]
            with np.load(folder/arm/'receive/pixels.npz') as z:
                panels.append((f'{arm} | LPIPS {score["quality"]["lpips_alex"]:.4f}',z['reconstruction'][8].copy()))
        comparison_image(panels,folder/'fixed_frame.png',job['row']['sample_id']+
                         ' | frame 9 and fixed center crop | nearest UF is NOT exact matched rate')
        fig,axs=plt.subplots(1,2,figsize=(7,3),constrained_layout=True)
        for ax,arm in zip(axs,('old_policy','cooperative')):
            mask=np.zeros(16);mask[point['scores'][arm]['selected']]=1
            ax.imshow(mask.reshape(4,4),vmin=0,vmax=1,cmap='Blues');ax.set_title(arm);ax.set_xticks([]);ax.set_yticks([])
        fig.suptitle('Local G choices; NOT transmitted masks')
        fig.savefig(folder/'local_G_choices.png',dpi=140);plt.close(fig)
        names += [str((folder/n).relative_to(root)) for n in ('fixed_frame.png','local_G_choices.png')]
    training_plot(root);names.append('training_diagnostic.png')
    return names
