"""Compact, CPU-only plots of saved weighted-LoRA gradient observations."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from demo.routervc_fullview_probe import read, digest, immutable, verify_artifacts
from demo.scalable_codec import atomic_json
from demo.scalable_experiment import now
from tools.g_objective_diagnostic import DEFAULT
from tools.texture_context_report import plotting


def aggregate(cases):
    if not cases: raise ValueError('no cases')
    names=cases[0]['gradient']['names'];index=names.index('lpips')
    output=[]
    for name in names:
        j=names.index(name)
        cosines=[r['gradient']['cosine'][j][index] for r in cases]
        ratios=[r['gradient']['norms'][name]/r['gradient']['norms']['lpips'] for r in cases]
        output.append(dict(name=name,count=len(cases),negative_lpips_cosine=sum(x<0 for x in cosines),
            mean_lpips_cosine=float(np.mean(cosines)),median_norm_over_lpips=float(np.median(ratios)),
            min_lpips_cosine=min(cosines),max_lpips_cosine=max(cosines),
            mean_weighted_loss=float(np.mean([r['values'][name]['weighted'] for r in cases]))))
    return output


def main(args):
    if not os.environ.get('TMUX'):raise RuntimeError('tmux required')
    root=args.root; output=root/'report';output.mkdir(exist_ok=True)
    done=read(root/'complete.json');verify_artifacts(root,done['artifacts'])
    immutable(output/'protocol.json',dict(summary=digest(root/'summary.json'),code=digest(Path(__file__))))
    if (output/'complete.json').exists():
        verify_artifacts(output,read(output/'complete.json')['artifacts']);print('REPORT_VERIFIED');return
    cases=read(root/'summary.json')['cases'];summary=aggregate(cases)
    groups={d:aggregate([r for r in cases if r['case']['dataset']==d]) for d in ('REDS','UVG')}
    names=cases[0]['gradient']['names'];plt=plotting()
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    ax=axes[0];ax.barh(names,[r['median_norm_over_lpips'] for r in summary],color='#16838b')
    ax.axvline(1.,color='black',linewidth=.8);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    ax.set_title('Weighted parameter gradient / LPIPS\nmedian norm ratio; not the scalar loss value')
    colors=['#d47730' if r['mean_lpips_cosine']<0 else '#16838b' for r in summary]
    ax=axes[1];ax.barh(names,[r['mean_lpips_cosine'] for r in summary],color=colors)
    ax.axvline(0,color='black',linewidth=.8);ax.invert_yaxis();ax.set_xlim(-1,1);ax.grid(axis='x',alpha=.2)
    ax.set_title('Mean cosine with weighted LPIPS gradient\nnegative: local first-order conflict, NOT quality evidence')
    fig.suptitle('Fixed mixed-view G / 8 predetermined training draws / no model updates')
    fig.savefig(output/'objectives.png',dpi=145);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,6),layout='constrained')
    lpips=names.index('lpips')
    matrix=np.array([[r['gradient']['cosine'][names.index(n)][lpips] for n in names] for r in cases])
    m=ax.imshow(matrix,vmin=-1,vmax=1,cmap='RdBu');fig.colorbar(m,ax=ax)
    ax.set_xticks(range(len(names)),names,rotation=35,ha='right')
    ax.set_yticks(range(len(cases)),[r['case']['sample_id']+' / '+str(r['case']['crop'][2]) for r in cases])
    ax.set_title('Each training draw: cosine with LPIPS / preserve both supporting and contrary cases')
    fig.savefig(output/'per_case.png',dpi=145);plt.close(fig)
    atomic_json(output/'summary.json',dict(aggregate=summary,by_dataset=groups,
        max_combined_backward_relative_error=max(r['combined_backward_relative_error'] for r in cases),
        exact_repeats=sum(r['exact_repeat'] for r in cases),seconds=sum(r['seconds'] for r in cases),
        peak_cuda_allocated_bytes=max(r['peak_cuda_allocated_bytes'] for r in cases),training=False,
        caution='Only a local first-order diagnostic. Verify any loss choice by paired actual training and images.'))
    names=['protocol.json','summary.json','objectives.png','per_case.png']
    atomic_json(output/'complete.json',dict(complete=True,utc=now(),artifacts={n:digest(output/n) for n in names}))
    print(json.dumps(read(output/'summary.json')),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=DEFAULT);main(p.parse_args())
