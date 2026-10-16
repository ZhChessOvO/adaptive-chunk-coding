"""Compact figures and evidence for the small offline G-selection check."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time
import numpy as np

from demo.routervc_fullview_probe import read,digest,verify_artifacts,immutable
from demo.scalable_codec import atomic_json
from tools.g_selection_probe import DEFAULT,prepare
from tools.texture_ablation_detail import resource_figure
from tools.texture_context_report import plotting


def compact(point):
    old=point['quality']['old']['lpips_alex'];best=point['best_finite_choice']
    new=point['quality'][best]['lpips_alex'];legacy=point['legacy_quality']['lpips_alex']
    gap=old-legacy
    return dict(sample=point['sample_id'],cohort=point['cohort'],dataset=point['dataset'],
        legacy=legacy,candidate_old=old,candidate_best=new,recovered=old-new,
        fraction_of_old_gap_recovered=(old-new)/gap if gap>0 else None,
        old_regions=point['choices']['old'],new_regions=point['choices'][best],best_policy=best,
        replaced_regions=len(set(point['choices']['old'])-set(point['choices'][best])),
        all_positive_regions=sum(v>0 for v in point['single_region_gains']))


def main(args):
    if not os.environ.get('TMUX'):raise RuntimeError('tmux required')
    root=args.output
    while not (root/'complete.json').exists():
        if not args.wait:raise RuntimeError('probe incomplete')
        if (root/'last_failure.json').exists():raise RuntimeError('probe failure; inspect first')
        time.sleep(5)
    with (root/'selection_probe.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX)
    prepare(root);verify_artifacts(root,read(root/'complete.json')['artifacts'])
    verify_artifacts(root/'generation',read(root/'generation/complete.json')['artifacts'])
    dest=root/'report';dest.mkdir(exist_ok=True)
    if (dest/'complete.json').exists():verify_artifacts(dest,read(dest/'complete.json')['artifacts']);return
    source=read(root/'summary.json');rows=[compact(p) for p in source['cases']]
    plt=plotting();fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    names=['REDS000','Beauty','REDS005','REDS020','ShakeNDry','HoneyBee*'];x=np.arange(len(rows))
    for j,(key,label,color) in enumerate((('legacy','Legacy G / frozen Router','#607d8b'),
            ('candidate_old','Candidate G / frozen Router','#d28d36'),
            ('candidate_best','Candidate G / GT-assisted finite choices','#238c8a'))):
        axes[0].bar(x+(j-1)*.25,[r[key] for r in rows],.25,label=label,color=color)
    axes[0].set_xticks(x,names,rotation=25);axes[0].set_ylabel('Whole 17-frame LPIPS / lower is better')
    axes[0].legend(fontsize=8);axes[0].grid(axis='y',alpha=.2)
    matrix=np.array([p['single_region_gains'] for p in source['cases']])
    im=axes[1].imshow(matrix,cmap='YlGnBu',aspect='auto');fig.colorbar(im,ax=axes[1],label='Single-region LPIPS gain / 5 fixed frames')
    axes[1].set_yticks(range(6),names);axes[1].set_xticks(range(16));axes[1].set_xlabel('Region index')
    for j,r in enumerate(rows):
        for i in r['old_regions']:axes[1].plot(i-.16,j,'o',ms=4,mfc='none',mec='white')
        for i in r['new_regions']:axes[1].plot(i+.16,j,'+',ms=5,color='#e46a24')
    axes[1].set_title('White circle: frozen Router / orange +: diagnostic choice',fontsize=9)
    fig.suptitle('Same candidate G and B/E bytes / at most 8 G regions\nOffline GT-assisted diagnosis, NOT a deployed Router or exhaustive optimum; * internal crop')
    fig.savefig(dest/'selection.png',dpi=145);plt.close(fig)
    resource=resource_figure(root,dest)
    memory=[json.loads(s) for s in (root/'memory.jsonl').read_text().splitlines()]
    atomic_json(dest/'compact.json',dict(cases=rows,resources=resource,last_memory=memory[-1],
        scope='finite offline GT-assisted diagnosis; not optimal or general benchmark'))
    immutable(dest/'complete.json',dict(complete=True,source=digest(root/'complete.json'),
        artifacts={n:digest(dest/n) for n in ('selection.png','resources.png','compact.json')}))
    print(json.dumps(rows),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--wait',action='store_true');main(p.parse_args())
