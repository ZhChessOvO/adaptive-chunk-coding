"""Receiver-only pair merging with a bounded maximum processing-window area.

Geometry is a scheduling proxy, not a measured VRAM/FLOP/latency guarantee.
Only exact adjacent selected cores can merge. The original writeback feather
mask is retained separately; merged context does not authorize extra pixels.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from types import SimpleNamespace

from demo.routervc_fullview_probe import read, digest, save, immutable, verify_artifacts

ORIGINAL = Path('/root/autodl-fs/DCVC/runs/routervc_revision_20261003/visual_evaluation_recovered')
OUTPUT = Path('/root/autodl-fs/DCVC/runs/routervc_efficiency_20261004/G_geometry')


def crop(roi, shape, halo=64):
    height, width = shape
    x,y,w,h = roi
    if min(x,y) < 0 or min(w,h) <= 0 or x+w > width or y+h > height:
        raise ValueError('ROI outside valid image')
    x0,y0=max(0,x-halo),max(0,y-halo)
    return [x0,y0,min(width,x+w+halo)-x0,min(height,y+h+halo)-y0]


def area(box):
    return box[2]*box[3]


def adjacent_union(a,b):
    ax,ay,aw,ah=a;bx,by,bw,bh=b
    horizontal = ay==by and ah==bh and (ax+aw==bx or bx+bw==ax)
    vertical = ax==bx and aw==bw and (ay+ah==by or by+bh==ay)
    if not (horizontal or vertical):
        return None
    x,y=min(ax,bx),min(ay,by)
    return [x,y,max(ax+aw,bx+bw)-x,max(ay+ah,by+bh)-y]


def schedule(indices, rois, shape, *, area_multiplier=1.5, halo=64):
    if (len(set(indices))!=len(indices) or any(type(i) is not int or not 0<=i<len(rois) for i in indices)
            or not math.isfinite(area_multiplier) or area_multiplier < 1):
        raise ValueError('invalid selected cells or processing-area cap')
    groups = [dict(members=[i], core=list(rois[i]), seed_slot=slot) for slot,i in enumerate(indices)]
    old_areas = [area(crop(g['core'],shape,halo)) for g in groups]
    cap = int(max(old_areas,default=0)*area_multiplier)
    while True:
        options = []
        for i,a in enumerate(groups):
            if len(a['members'])!=1: continue
            for j in range(i+1,len(groups)):
                b=groups[j]
                if len(b['members'])!=1: continue
                box=adjacent_union(a['core'],b['core'])
                if box is None: continue
                merged=area(crop(box,shape,halo))
                saving=area(crop(a['core'],shape,halo))+area(crop(b['core'],shape,halo))-merged
                if merged<=cap and saving>0:
                    options.append((-saving,a['seed_slot'],b['seed_slot'],i,j,box))
        if not options: break
        _,_,_,i,j,box=min(options)
        a,b=groups[i],groups[j]
        groups[i]=dict(members=a['members']+b['members'],core=box,seed_slot=min(a['seed_slot'],b['seed_slot']))
        del groups[j]
    groups.sort(key=lambda g:g['seed_slot'])
    for g in groups:
        g['processing_crop']=crop(g['core'],shape,halo)
    new_areas=[area(g['processing_crop']) for g in groups]
    return dict(groups=groups, selected_indices=list(indices), original_calls=len(indices),
        merged_calls=len(groups), original_total_area=sum(old_areas), total_area=sum(new_areas),
        original_max_area=max(old_areas,default=0), max_area=max(new_areas,default=0),
        area_cap=cap, area_multiplier=area_multiplier,
        total_area_ratio=sum(new_areas)/sum(old_areas) if old_areas else 1.,
        max_area_ratio=max(new_areas)/max(old_areas) if old_areas else 1.,
        selected_core_area_unchanged=sum(area(rois[i]) for i in indices)==sum(area(g['core']) for g in groups))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ORIGINAL)
    p.add_argument('--output',type=Path,default=OUTPUT)
    args=p.parse_args(argv)
    if not os.environ.get('TMUX'):raise RuntimeError('formal geometry report requires tmux')
    if args.output.resolve().is_relative_to(args.root.resolve()):raise ValueError('keep old results unchanged')
    from demo.chunk_enhancement_experiment import Run
    run=Run(SimpleNamespace(output=args.output,command='G_geometry',max_hours=1));run.thread.start()
    try:
        old=read(args.root/'protocol.json')
        names=[f"samples/{e['sample']['sample_id']}/global_local_e0.5_g8/fresh/decode.json" for e in old['sources']]
        protocol=dict(code_sha256=digest(__file__), original_complete=digest(args.root/'complete.json'),
            source_artifacts={n:digest(args.root/n) for n in names},max_group_cells=2,area_caps=[1.,1.5],
            changed_pixels='none; CPU geometry only',measured_speedup=False,measured_memory=False)
        immutable(run.root/'protocol.json',protocol)
        if (run.root/'complete.json').exists():
            verify_artifacts(run.root,read(run.root/'complete.json')['artifacts'])
            print('G_GEOMETRY_VERIFIED_READ_ONLY',flush=True);return
        rows=[]
        for entry,name in zip(old['sources'],names):
            run.check(); d=read(args.root/name); route=d['route'];sample=entry['sample']
            size=sample['transform']['coded_size']
            for cap in (1.,1.5):
                plan=schedule(route['indices'],route['rois'],(size[1],size[0]),area_multiplier=cap)
                rows.append(dict(sample_id=sample['sample_id'],dataset=sample['dataset'],**plan))
        groups={}
        for dataset in ('REDS','UVG'):
            groups[dataset]={}
            for cap in (1.,1.5):
                selected=[r for r in rows if r['dataset']==dataset and r['area_multiplier']==cap]
                groups[dataset][str(cap)]={k:sum(r[k] for r in selected)/len(selected) for k in
                    ('original_calls','merged_calls','total_area_ratio','max_area_ratio')}
        save(run.root/'summary.json',dict(complete=True,windows=13,rows=rows,group_means=groups,
            measured_speedup=False,measured_memory=False,
            interpretation='cap is input area, not a hard GPU memory ceiling; quality/context/noise can change'))
        names=['protocol.json','summary.json']
        save(run.root/'complete.json',dict(complete=True,artifacts={n:digest(run.root/n) for n in names}))
        run.update(phase='complete',completed=13,total=13)
        print('G_GEOMETRY_COMPLETE',flush=True)
    finally:
        run.log_resources();run.stop.set();run.thread.join(timeout=3);run.lock.close()


if __name__=='__main__':main()
