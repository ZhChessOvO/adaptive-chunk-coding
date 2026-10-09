"""Guarded single-GPU new-latent R_s smoke, fixed teacher, paired fit and verify."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from demo import routervc_sender_train as fit
from demo.routervc_fullview_probe import read,digest,save,immutable,verify_artifacts
from demo.chunk_enhancement_experiment import Run
from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
from demo.routervc_mixed_queue import exact
from routervc.latent import routing,sender_data as data
from tools.latent_router_queue import execute as receive_worker


def execute(run,name,args,distributed=False):
    command=[sys.executable]
    if distributed:command+=['-m','torch.distributed.run','--standalone','--nproc-per-node=1']
    command+=['-m','tools.latent_sender_worker',*map(str,args)]
    path=run.root/(name+'.log')
    with path.open('a') as log:
        run.update(phase=name,log=str(path))
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while child.poll() is None:
                run.check();time.sleep(.5)
            if child.returncode:raise RuntimeError(f'{name} failed ({child.returncode}); see {path}')
        finally:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait()


def verify_labels(root,cache,protocol,run):
    samples=data.Samples(root,cache,protocol,run.check,run.update)
    labels=fit.verify_labels(root,protocol,samples,run.check)
    for row in protocol['rows']:
        run.check();folder=root/'samples'/row['sample_id']
        batch=samples.get(row)
        bank=Path(row['bank_path']).read_bytes()
        for state,plan in enumerate(batch['plans']):
            parent=read(folder/f's{state}/parent/result.json')
            for name,selected in [('parent',plan['selected']),
                *((f'add{i}',plan['selected']+[i]) for i in plan['candidates'])]:
                dest=folder/f's{state}'/name;result=read(dest/'result.json')
                verify_artifacts(dest,result['artifacts'])
                wire=(dest/'stream.rvlrg').read_bytes()
                inner,config,_=routing.parse(wire)
                if (inner!=routing.subset(bank,selected) or result['total_bytes']!=len(wire)
                        or result['stream_sha256']!=digest(dest/'stream.rvlrg')
                        or config['receiver_sha256']!=data.RECEIVER_SHA
                        or result['binding']['protocol']!=digest(root/'protocol.json')
                        or result['binding']['selected']!=selected or result['mask_bytes']!=0):
                    raise ValueError('measured final state binding/bytes changed')
                if name!='parent':
                    region=int(name[3:])
                    gain=parent['quality']['lpips_alex']-result['quality']['lpips_alex']
                    if (np.float32(gain)!=batch['targets']['value'][state,region].item()
                            or result['total_bytes']-parent['total_bytes']!=batch['packet_bytes'][state,region]):
                        raise ValueError('cached target differs from actual final marginal')
    samples.release()
    return labels


def smoke_checks(run,root,cache,protocol):
    hashes=[]
    for index,row in enumerate(protocol['rows']):
        folder=root/'samples'/row['sample_id'];plans=read(folder/'plans.json')['plans']
        chosen=plans[1]['candidates'][0]
        for name,relative,disabled in [('empty','s0/parent',False),('partial','s1/parent',False),
                                     ('child',f's1/add{chosen}',False),('repeat',f's1/add{chosen}',False),
                                     ('off',f's1/add{chosen}',True)]:
            teacher=read(folder/relative/'result.json');dest=root/'fresh'/row['sample_id']/name
            args=['decode','--stream',folder/relative/'stream.rvlrg','--output',dest,
                  '--receiver','/nonexistent/Goff.pt' if disabled else data.RECEIVER]
            if disabled:args+=['--disable-generation']
            if not (dest/'complete.json').exists():
                receive_worker(run,f'fresh_{index}_{name}',args,distributed=True)
            receipt=read(dest/'complete.json');verify_artifacts(dest,receipt['artifacts'])
            if (receipt['base_hash']!=teacher['base_hash'] or receipt['enhanced_hash']!=teacher['received_hash']
                    or receipt['actual_bytes']!=teacher['total_bytes']
                    or receipt['output_hash']!=teacher['received_hash' if disabled else 'output_hash']):
                raise ValueError('fresh actual G/Y output differs from teacher')
            hashes.append(digest(dest/'complete.json'))
        for arm in fit.ARMS:
            output=root/'planned'/row['sample_id']/arm
            checkpoint=root/'resumed'/arm/'best.pt'
            if not (output/'complete.json').exists():
                execute(run,f'plan_{index}_{arm}', ['plan','--root',root,'--cache',cache,'--output',output,
                    '--sample-id',row['sample_id'],'--checkpoint',checkpoint])
            done=read(output/'complete.json');verify_artifacts(output,done['artifacts'])
            for point in (done['points'][0],done['points'][2]):
                dest=output/('fresh_'+point['file'][:-6])
                if not (dest/'complete.json').exists():
                    receive_worker(run,f'planned_fresh_{index}_{arm}_{point["file"]}',
                        ['decode','--stream',output/point['file'],'--output',dest,'--receiver',data.RECEIVER],
                        distributed=True)
                receipt=read(dest/'complete.json');verify_artifacts(dest,receipt['artifacts'])
                if receipt['actual_bytes']!=point['actual_bytes'] or receipt['base_hash']!=teacher['base_hash']:
                    raise ValueError('planned stream bytes/B differ')
                hashes.append(digest(dest/'complete.json'))
    save(root/'fresh_audit.json',dict(complete=True,fresh_points=len(hashes),receipts=hashes,
        teacher_fresh_pixels_exact=True,repeats_exact=True,G_off_no_G_or_Rg_assets=True,
        actual_mixed_Y=True,sourcefree_receiver=True,masks_transmitted=False,
        planned_prefix_pairs=12,both_sender_arms=True))


def stage(run,args,smoke):
    root=args.output/('smoke' if smoke else 'formal');cache=args.cache/('smoke' if smoke else 'formal')
    protocol=data.make_protocol(smoke);immutable(root/'protocol.json',protocol)
    if (root/'complete.json').exists():
        verify(root,cache,run);return
    if not smoke:
        tested=read(args.output/'smoke/protocol.json');verify_artifacts(args.output/'smoke',read(args.output/'smoke/complete.json')['artifacts'])
        for key in ('code','teacher','receiver_profile','packet_profile'):
            if protocol[key]!=tested[key]:raise ValueError('formal differs from smoke: '+key)
    common=['--root',root,'--cache',cache,'--max-hours',args.max_hours]
    if smoke and not (root/'label_resume_checked.json').exists():
        stop=root/'label_stop'
        if not (stop/'intentional_stop.json').exists():
            execute(run,'smoke_label_stop',['labels',*common,'--output',stop,'--stop-renderings',1],True)
        prior={str(p.relative_to(root)):digest(p) for p in (root/'samples').glob('*/s*/*/result.json')}
        execute(run,'smoke_labels',['labels',*common,'--output',root/'label_worker'],True)
        if not prior or any(digest(root/n)!=h for n,h in prior.items()):
            raise ValueError('teacher did not resume its measured states exactly')
        save(root/'label_resume_checked.json',dict(complete=True,preserved_states=prior))
    elif not (root/'labels.complete.json').exists():
        execute(run,'formal_labels' if not smoke else 'smoke_labels',
                ['labels',*common,'--output',root/'label_worker'],True)
    verify_labels(root,cache,protocol,run)
    if smoke:
        if not (root/'resume_checked.json').exists():
            if not (root/'resumed/resume.pt').exists():
                execute(run,'smoke_stop',['fit',*common,'--output',root/'resumed','--stop-after',1])
            execute(run,'smoke_resume',['fit',*common,'--output',root/'resumed'])
            execute(run,'smoke_direct',['fit',*common,'--output',root/'direct'])
            exact(torch.load(root/'resumed/resume.pt',map_location='cpu',weights_only=True),
                  torch.load(root/'direct/resume.pt',map_location='cpu',weights_only=True))
            save(root/'resume_checked.json',dict(complete=True,models_optimizers_rng_best_exact=True))
        smoke_checks(run,root,cache,protocol)
        names=['protocol.json','labels.complete.json','train_scale.json','resumed/complete.json',
               'direct/complete.json','resume_checked.json','label_resume_checked.json','fresh_audit.json']
    else:
        execute(run,'formal_fit',['fit',*common,'--output',root/'router'])
        names=['protocol.json','labels.complete.json','train_scale.json','router/complete.json']
    save(root/'complete.json',dict(complete=True,smoke=smoke,new_latent_sender=True,
        evaluation_pending=not smoke,artifacts={n:digest(root/n) for n in names}))


def verify(root,cache,run):
    done=read(root/'complete.json');verify_artifacts(root,done['artifacts'])
    protocol=read(root/'protocol.json')
    for name,h in protocol['code'].items():
        if digest(data.REPO/name)!=h:raise ValueError('completed sender source changed')
    labels=verify_labels(root,cache,protocol,run)
    for folder in ('resumed','direct') if protocol['smoke'] else ('router',):
        fit.verify_training(root/folder,protocol,labels_binding=labels)
    print('LATENT_SENDER_VERIFIED',root,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('smoke','train','all','verify'))
    p.add_argument('--output',type=Path,default=data.ROOT)
    p.add_argument('--cache',type=Path,default=data.CACHE)
    p.add_argument('--max-hours',type=float,default=48.)
    args=p.parse_args()
    if not os.environ.get('TMUX'):raise RuntimeError('tmux required')
    run=Run(SimpleNamespace(output=args.output/'queue',command='latent_sender',max_hours=args.max_hours))
    run.thread.start()
    try:
        if args.command=='verify':
            for name in ('smoke','formal'):verify(args.output/name,args.cache/name,run)
        else:
            with exclusive_native_evaluation(run):
                if args.command in ('smoke','all'):stage(run,args,True)
                if args.command in ('train','all'):stage(run,args,False)
                run.update(phase='latent_sender_stage_complete')
    except BaseException as error:
        save(run.root/'last_failure.json',dict(error=repr(error),progress=run.progress));raise
    finally:
        run.log_resources();run.stop.set();run.thread.join(timeout=3);run.lock.close()


if __name__=='__main__':main()
