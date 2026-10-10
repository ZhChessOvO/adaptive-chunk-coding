"""CPU boundary-centered LPIPS from completed controls or the learned review."""
import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from demo.chunk_enhancement_experiment import Run
from demo.routervc_fullview_probe import read, save, digest, immutable, verify_artifacts
from routervc.fusion.boundaries import edges, CATEGORIES
from routervc.fusion.metrics import boundary_lpips
from tools.latent_boundary_report import ROOT, EVALUATION


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--learned', action='store_true')
    args=p.parse_args()
    if not os.environ.get('TMUX'):raise RuntimeError('tmux required')
    run=Run(SimpleNamespace(output=ROOT/('p1_boundary_learned' if args.learned else 'p1_boundary_controls'),
                            command='perceptual',max_hours=4));run.thread.start()
    try:
        import lpips
        torch.set_num_threads(2)
        model=lpips.LPIPS(net='alex',verbose=False).eval().requires_grad_(False)
        rows=read(EVALUATION/'summary.json')['scope']['rows'];records=[]
        immutable(run.root/'protocol.json',dict(learned=args.learned,source_summary=digest(EVALUATION/'summary.json'),
            code={n:digest(Path(__file__).resolve().parents[1]/n) for n in
                  ('tools/fusion_boundary_quality.py','routervc/fusion/metrics.py')},crop=128,frames=[0,8,16]))
        for index,row in enumerate(rows):
            run.check();sid=row['sample_id'];folder=ROOT/'p1_controls/samples'/sid
            target=run.root/(sid+'.json')
            if target.exists():records.append(read(target));continue
            verify_artifacts(folder,read(folder/'complete.json')['artifacts'])
            verify_artifacts(folder/'controls',read(folder/'controls/complete.json')['artifacts'])
            with np.load(row['source_path']) as z:source=z['source']
            with np.load(folder/'current.npz') as z:current=z['pixels']
            with np.load(folder/'controls/controls.npz') as z:overlap,multiband=z['overlap'],z['multiband']
            variants=dict(current=current,overlap=overlap,multiband=multiband)
            if args.learned:
                fresh=ROOT/'p1_evaluation/samples'/sid
                verify_artifacts(fresh,read(fresh/'complete.json')['artifacts'])
                with np.load(fresh/'learned.npz') as z:variants['learned']=z['pixels']
            if digest(Path(row['source_path']))!=row['source_sha256']:raise ValueError('source changed')
            regions=read(folder/'received.json')['detail']['received_regions']
            generated=read(folder/'complete.json')['generated']
            result=boundary_lpips(source,variants,edges(source.shape,regions,generated),model,check=run.check)
            result.update(sample_id=sid,dataset=row['dataset'])
            save(target,result);records.append(result)
            run.update(phase='boundary_perceptual',completed=index+1,total=len(rows),sample=sid)
        groups={}
        for dataset in ('REDS','UVG'):
            subset=[r for r in records if r['dataset']==dataset]
            groups[dataset]={c:{v:float(np.mean([r['metrics'][c][v] for r in subset if r['metrics'][c][v] is not None]))
                               for v in subset[0]['metrics'][c]} for c in CATEGORIES}
        save(run.root/'summary.json',dict(complete=True,groups=groups,records=records))
        save(run.root/'complete.json',dict(complete=True,artifacts={n:digest(run.root/n) for n in ('protocol.json','summary.json')}))
        run.update(phase='complete');run.log_resources()
    finally:run.stop.set();run.thread.join();run.lock.close()


if __name__=='__main__':main()
