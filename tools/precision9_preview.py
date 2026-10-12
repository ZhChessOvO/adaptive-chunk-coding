"""Publishable startup B clarity panels, separate from formal review figures.

No GPU, inference or tuning. Read only completed source-free startup decodes.
Every image uses the same fixed frame/central crop, not a quality-chosen frame.
"""
import argparse
import os
from pathlib import Path
import numpy as np
from demo.routervc_fullview_probe import read,save,digest,verify_artifacts
from tools.cooperative_review import source_rgb
from tools.plot_latent_diagnostics import comparison_image


def make(root):
    protocol=read(root/'protocol.json');result={}
    output=root/'startup_previews';output.mkdir(exist_ok=True)
    for dataset in ('REDS','UVG'):
        row=next(r for r in protocol['rows'] if r['cohort']=='calibration' and r['dataset']==dataset)
        places={w:(root if w==9 else Path(protocol['parent_root']))/'startup'/row['sample_id']/f'w{w}'/'base'/'receive' for w in (3,5,7,9)}
        if not all((p/'complete.json').exists() for p in places.values()):continue
        bindings={str(p/'complete.json'):digest(p/'complete.json') for p in places.values()}
        name=dataset.lower()+'_base_widths.png';record=output/(dataset.lower()+'.json')
        if record.exists():
            prior=read(record);verify_artifacts(output,prior['artifacts'])
            if prior['inputs']!=bindings:raise ValueError('startup preview inputs changed')
            result[dataset]=prior;continue
        panels=[('Original',source_rgb(row)[8])];details=[]
        for w,p in places.items():
            done=read(p/'complete.json');verify_artifacts(p,done['artifacts'])
            if not done['generation_disabled'] or done['output_hash']!=done['base_hash']:
                raise ValueError('preview must be genuine fresh B-only pixels')
            with np.load(p/'pixels.npz',allow_pickle=False) as z:
                panels.append((f'width{w}: {done["actual_bytes"]:,} bytes',z['reconstruction'][8].copy()))
            details.append(dict(width=w,bytes=done['actual_bytes'],bpp=done['bpp'],base_hash=done['base_hash']))
        comparison_image(panels,output/name,row['sample_id']+' | startup example, NOT aggregate RD | frame 8; fixed center zoom')
        prior=dict(sample_id=row['sample_id'],dataset=dataset,inputs=bindings,frame=8,
            preview_only=True,points=details,artifacts={name:digest(output/name)})
        save(record,prior);result[dataset]=prior
    print(result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/root/autodl-fs/DCVC/runs/routervc_precision9_20261012'))
    args=parser.parse_args()
    if not os.environ.get('TMUX'):raise RuntimeError('tmux required')
    make(args.root)
