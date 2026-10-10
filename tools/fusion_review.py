"""After the bounded fit: paired cached review plus source-free fresh checks."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

from demo.chunk_enhancement_experiment import Run
from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
from demo.routervc_fullview_probe import read, save, digest, verify_artifacts
from routervc.latent.sender_data import RECEIVER
from tools.latent_boundary_report import ROOT


def execute(run,name,module,args,distributed=False):
    command=[sys.executable]
    if distributed:command+=['-m','torch.distributed.run','--standalone','--nproc-per-node=1']
    command+=['-m',module,*map(str,args)]
    with (run.root/(name+'.log')).open('a') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while child.poll() is None:run.check();time.sleep(.5)
            if child.returncode:raise RuntimeError(f'{name} failed; see {run.root}/{name}.log')
        finally:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit',type=Path,default=ROOT/'p1_fit')
    p.add_argument('--output',type=Path,default=ROOT/'p1_evaluation')
    a=p.parse_args()
    if not os.environ.get('TMUX'):raise RuntimeError('tmux required')
    run=Run(SimpleNamespace(output=a.output/'queue',command='review',max_hours=48));run.thread.start()
    try:
        while not (a.fit/'complete.json').exists():
            if (a.fit/'last_failure.json').exists():raise RuntimeError('fusion fitting needs attention')
            run.check();run.update(phase='waiting_for_fusion_fit');time.sleep(5)
        verify_artifacts(a.fit,read(a.fit/'complete.json')['artifacts'])
        checkpoint=a.fit/'best.pt'
        with exclusive_native_evaluation(run):
            if not (a.output/'offline.complete.json').exists():
                run.update(phase='paired_offline_review')
                execute(run,'offline','tools.fusion_offline',['--output',a.output,'--checkpoint',checkpoint])
            verify_artifacts(a.output,read(a.output/'offline.complete.json')['artifacts'])
            summary=read(a.output/'summary.json');receipts=[]
            selected=[next(r for r in summary['results'] if r['dataset']==d) for d in ('REDS','UVG')]
            for i,row in enumerate(selected):
                folder=a.output/'samples'/row['sample_id']
                for name,mode,disabled in [('current','current',False),('multiband','multiband',False),
                        ('learned','learned',False),('repeat','learned',False),('Goff','learned',True)]:
                    run.update(phase='fresh_sourcefree_receiver',sample=row['sample_id'],variant=name)
                    dest=folder/'fresh'/name;wire=folder/f'{mode}.rvlf'
                    if not (dest/'complete.json').exists():
                        args=['--stream',wire,'--output',dest,
                              '--receiver','/nonexistent/Goff.pt' if disabled else RECEIVER,
                              '--checkpoint','/nonexistent/Foff.pt' if disabled else checkpoint]
                        if disabled:args+=['--disable-generation']
                        execute(run,f'fresh_{i}_{name}','tools.fusion_decode',args,True)
                    receipt=read(dest/'complete.json');verify_artifacts(dest,receipt['artifacts'])
                    if (receipt['actual_bytes']!=row['actual_bytes'] or receipt['base_hash']!=row['base_hash']
                            or receipt['enhanced_hash']!=row['enhanced_hash'] or receipt['additional_mask_bytes']!=0
                            or receipt['source_frames_read'] or receipt['sender_router_loaded']):
                        raise ValueError('fresh bytes/sourcefree/B/Y binding failed')
                    if disabled:
                        expected=row['enhanced_hash']
                        if receipt['fusion_model_loaded'] or receipt['generated']:raise ValueError('Goff loaded F/G')
                    elif mode=='learned':expected=row['output_hash']
                    elif mode=='current':
                        expected=read(ROOT/'p1_controls/samples'/row['sample_id']/'complete.json')['output_hash']
                    else:
                        import numpy as np
                        from demo.scalable_format import frame_hash
                        with np.load(ROOT/'p1_controls/samples'/row['sample_id']/'controls/controls.npz') as z:
                            expected=frame_hash(z[mode])
                    if receipt['output_hash']!=expected:raise ValueError('fresh fused pixels differ from paired cache')
                    receipts.append(dict(sample_id=row['sample_id'],variant=name,receipt=digest(dest/'complete.json')))
            save(a.output/'fresh_audit.json',dict(complete=True,receipts=receipts,points=len(receipts),
                sourcefree_pixels_exact=True,repeats_exact=True,Goff_requires_no_Rg_G_F_assets=True,
                additional_header_bytes=77,additional_mask_bytes=0))
            save(a.output/'complete.json',dict(complete=True,paired_views=13,fresh_checks=len(receipts),
                checkpoint_sha256=digest(checkpoint),artifacts={n:digest(a.output/n) for n in
                    ('protocol.json','summary.json','offline.complete.json','fresh_audit.json')}))
            run.update(phase='fusion_review_complete');run.log_resources()
    except BaseException as error:
        save(run.root/'last_failure.json',dict(error=repr(error),progress=run.progress));raise
    finally:run.stop.set();run.thread.join();run.lock.close()


if __name__=='__main__':main()
