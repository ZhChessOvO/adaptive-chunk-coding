"""Readable detail contact sheets and measured resource plots after evaluation."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from demo.routervc_fullview_probe import read, digest, verify_artifacts
from demo.scalable_codec import atomic_json
from tools.texture_ablation import DEFAULT
from tools.texture_ablation_report import panels


def resource_figure(root,output):
    import matplotlib.pyplot as plt
    events=[json.loads(s) for s in (root/'heartbeat.jsonl').read_text().splitlines()]
    xs=np.array([r['elapsed_seconds'] for r in events])/60
    memory=[]
    for row in events:
        line=row['gpu'].splitlines()[0];memory.append(float(line.split(',')[2])/1024)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    axes[0].plot(xs,memory,color='#008b8b');axes[0].set_ylabel('GPU device memory / GiB')
    axes[0].set_title('Device memory, sampled every 30s')
    for mount in ('/root','/root/autodl-tmp','/root/autodl-fs'):
        axes[1].plot(xs,[r['disks'][mount]['used_percent'] for r in events],label=mount)
    axes[1].axhline(80,color='#b91c1c',linestyle='--',label='80% stop guard')
    axes[1].set_ylim(0,85);axes[1].set_ylabel('Disk used / %');axes[1].legend(fontsize=8)
    axes[1].set_title('All three mounts; no dataset downloads')
    for ax in axes:ax.grid(alpha=.25);ax.set_xlabel('Minutes since queue launch')
    fig.savefig(output/'resources.png',dpi=160);plt.close(fig)
    return dict(sampled_gpu_peak_gib=max(memory),queue_minutes=float(xs[-1]),
                last_disks=events[-1]['disks'])


def main(args):
    if not os.environ.get('TMUX'):raise RuntimeError('tmux required for report queue')
    while not (args.root/'complete.json').exists():
        if not args.wait:raise RuntimeError('ablation not complete')
        time.sleep(5)
    evaluation=args.root/'evaluation'
    done=read(evaluation/'complete.json');verify_artifacts(evaluation,done['artifacts'])
    summary=read(evaluation/'summary.json');output=args.root/'presentation';output.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    artifacts={}
    for item in summary['cases']:
        case=item['case']
        with np.load(item['reference_inputs']) as z:source=z['source']
        values=panels(case,source,{r['name']:r for r in item['results']})
        fig,axes=plt.subplots(2,3,figsize=(10,8),layout='constrained')
        for ax,(label,pixels,result) in zip(axes.flat,values):
            h,w=pixels.shape[1:3];ch,cw=min(96,h),min(96,w)
            detail=pixels[8,(h-ch)//2:(h+ch)//2,(w-cw)//2:(w+cw)//2]
            ax.imshow(detail,interpolation='nearest');ax.axis('off')
            if result:
                label+=f"\nWhole-core LPIPS {result['quality']['lpips_alex']:.4f}"
                label+=f" | phase16 {result['error_phase16']['phase_rms_u8']:.3f}"
            ax.set_title(label,fontsize=10)
        fig.suptitle(case['sample_id']+'\nSame 96px crop / frame 9 / nearest display zoom / NO image filtering',fontsize=11)
        name=case['sample_id']+'_detail.png';fig.savefig(output/name,dpi=150);plt.close(fig)
        artifacts[name]=digest(output/name)
    resource=resource_figure(args.root,output);artifacts['resources.png']=digest(output/'resources.png')
    atomic_json(output/'complete.json',dict(complete=True,evaluation_receipt=digest(evaluation/'complete.json'),
        code=digest(Path(__file__)),resource=resource,artifacts=artifacts))
    print(json.dumps(dict(complete=True,figures=len(artifacts),resource=resource)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=DEFAULT);p.add_argument('--wait',action='store_true')
    main(p.parse_args())
