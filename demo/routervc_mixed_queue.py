"""Single-A800 tmux queue: smoke, then training only; evaluation is separate."""
import argparse
import os
from pathlib import Path
from types import SimpleNamespace

from demo.routervc_mixed_train import ROOT, CACHE, make_protocol
from demo.routervc_fullview_probe import read, digest, save, immutable, verify_artifacts
from demo.chunk_enhancement_experiment import Run
from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
from demo.conditioned_generation_pipeline import execute


def exact(a,b):
    import torch
    if torch.is_tensor(a): torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        if a.keys() != b.keys(): raise ValueError('resume tree keys differ')
        for key in a: exact(a[key],b[key])
    elif isinstance(a,(tuple,list)):
        if len(a) != len(b): raise ValueError('resume tree lengths differ')
        for x,y in zip(a,b): exact(x,y)
    elif a != b: raise ValueError(f'resume scalar differs: {a} versus {b}')


def fresh_checks(run, data_root, protocol):
    import numpy as np
    from demo.routervc_mixed_data import ENHANCEMENT, ADAPTER
    from demo.routervc_mixedview_teacher import crop
    from demo.online_eg_eval_core import noise_pair
    from demo.routervc_encode import compose_candidates
    from demo.routervc_mixed_router import selection
    records = {}
    for row in protocol['rows']:
        with np.load(row['reconstruction_path'],allow_pickle=False) as data:
            base,enhanced = data['base'].copy(),data['enhanced'].copy()
        rois = [r['roi'] for r in read(row['path'])['regions']]
        # Boundary/interior, selected/unselected locations arise naturally from
        # deterministic patterns. Fresh checks cover both partial E levels.
        for j,i in ((0,0),(1,5)):
            cell = data_root/'samples'/row['sample_id']/f'mix{j}/cell_{i:02d}'
            expected = read(cell/'result.json')
            for off in (False,True) if j == 0 else (False,):
                out = data_root/'fresh_checks'/row['sample_id']/f'mix{j}_r{i}_off{int(off)}'
                out.mkdir(parents=True,exist_ok=True); done=out/'checked.json'
                if done.exists():
                    record=read(done);verify_artifacts(out,record['artifacts'])
                    if record['stream'] != digest(cell/'teacher.acsg'): raise ValueError('fresh stream changed')
                else:
                    if not (out/'decode.json').exists():
                        execute(run,'fresh_'+row['sample_id']+f'_{j}_{i}_{off}','online_eg_decode.py',
                            ['--stream',cell/'teacher.acsg','--output',out,'--enhancement',ENHANCEMENT,
                             '--adapter',out/'intentionally_absent_G.pt' if off else ADAPTER,
                             *(['--disable-generation'] if off else [])],distributed=True)
                    report=read(out/'decode.json')
                    if (report['source_frames_read'] or not report['outside_generate_exact']
                            or report['total_bytes'] != expected['total_bytes']
                            or report['generation_input_hash'] != expected['generation_input_hash']):
                        raise ValueError('fresh mixed condition differs from actual partial E decode')
                    with np.load(out/'reconstruction.npz',allow_pickle=False) as data: actual=data['reconstruction'].copy()
                    if off:
                        mixed=compose_candidates(base,enhanced,selection(row['sample_id'],j),rois)
                        np.testing.assert_array_equal(actual,mixed)
                        if report['generation_assets_validated']: raise ValueError('G-off loaded G assets')
                    else:
                        if report['output_hash'] != expected['output_hash']: raise ValueError('fresh mixed generation differs')
                        with np.load(cell/'generated.npz',allow_pickle=False) as data:
                            np.testing.assert_array_equal(crop(actual,rois[i]),data['generated'])
                        noise_pair(report,dict(generation_runtime=expected['runtime']),same_condition=True)
                    record=dict(complete=True,stream=digest(cell/'teacher.acsg'),G_off=off,
                        source_free=True,pixels_exact=True,
                        artifacts={n:digest(out/n) for n in ('decode.json','reconstruction.npz')})
                    save(done,record)
                records[str(done.relative_to(data_root))]=digest(done)
    immutable(data_root/'fresh_audit.json',dict(complete=True,checks=records,
        mixed_subset_decode_exact=True,persistent_G_matches_fresh=True,G_off_without_G_assets=True))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('smoke','train','verify'))
    p.add_argument('--output',type=Path,default=ROOT);p.add_argument('--cache',type=Path,default=CACHE)
    p.add_argument('--max-hours',type=float,default=24.)
    args=p.parse_args()
    if not os.environ.get('TMUX'): raise RuntimeError('run this resumable queue inside tmux')
    import math
    if not math.isfinite(args.max_hours) or args.max_hours<=0: raise ValueError('positive finite max-hours required')
    run=Run(SimpleNamespace(output=args.output/'queue',command='mixed_router',max_hours=args.max_hours))
    run.thread.start();os.environ['ROUTERVC_MIXED_PARENT']=str(os.getpid())
    try:
        smoke=args.command=='smoke';data_root=args.output/('smoke' if smoke else 'formal')
        protocol=make_protocol(smoke);immutable(data_root/'protocol.json',protocol)
        if args.command=='verify':
            execute(run,'verify_only','routervc_mixed_train.py',
                ['--root',data_root,'--cache',args.cache/'formal','--output',data_root/'router','--verify-only'])
            return
        with exclusive_native_evaluation(run):
            if smoke:
                def worker(name,stop=0):
                    execute(run,name,'routervc_mixed_train.py',
                        ['--root',data_root,'--cache',args.cache/'smoke','--output',data_root/name,
                         '--stop-after',stop],distributed=True)
                if not (data_root/'resume_checked.json').exists():
                    worker('resumed',1)
                    saved={str(f.relative_to(data_root)):digest(f) for f in (data_root/'samples').rglob('*') if f.is_file()}
                    worker('resumed');worker('direct')
                    for name,value in saved.items():
                        if digest(data_root/name)!=value: raise ValueError('resume rewrote completed labels')
                    import torch
                    a=torch.load(data_root/'resumed/resume.pt',weights_only=True,map_location='cpu')
                    b=torch.load(data_root/'direct/resume.pt',weights_only=True,map_location='cpu')
                    exact(a,b)
                    # Every arm really updated, and public B/Y-only reload matches.
                    from demo.routervc_mixed_router import ARMS, load_model, predict
                    initial=torch.load(protocol['initial_model'],weights_only=True,map_location='cpu')['state_dict']
                    row=protocol['rows'][0]
                    with np_load(row['reconstruction_path']) as pixels:
                        base,enhanced=pixels['base'].copy(),pixels['enhanced'].copy()
                    import numpy as np
                    from demo.routervc_mixed_router import selection,build_inputs
                    from demo.routervc_visual_router import grid_rois
                    from demo.routervc_encode import compose_candidates
                    selected=selection(row['sample_id'],0);coverage=np.zeros(16,np.float32);coverage[selected]=1
                    received=compose_candidates(base,enhanced,selected,grid_rois(*base.shape[1:3]))
                    for arm in ARMS:
                        if all(torch.equal(v,initial[k]) for k,v in a['models'][arm].items()):
                            raise ValueError('Router did not actually optimize')
                        model,payload=load_model(data_root/'resumed'/arm/'best.pt')
                        with torch.no_grad():
                            expected=model(build_inputs(base,received,coverage,halo=payload['input_halo']))['gains']
                        torch.testing.assert_close(predict(model,payload,base,received,coverage),expected,rtol=0,atol=0)
                    save(data_root/'resume_checked.json',dict(complete=True,all_arms=True,
                        models_exact=True,optimizers_exact=True,best_selection_exact=True,
                        public_input_reload_exact=True,completed_labels_preserved=True))
                fresh_checks(run,data_root,protocol)
                immutable(data_root/'complete.json',dict(complete=True,protocol=digest(data_root/'protocol.json'),
                    artifacts={n:digest(data_root/n) for n in ('resume_checked.json','fresh_audit.json','resumed/complete.json','direct/complete.json')}))
                run.update(phase='smoke_complete')
            else:
                smoke_root=args.output/'smoke';tested=read(smoke_root/'protocol.json')
                checked=read(smoke_root/'complete.json');verify_artifacts(smoke_root,checked['artifacts'])
                if checked['protocol']!=digest(smoke_root/'protocol.json') or not checked['complete']:
                    raise ValueError('smoke not complete')
                for key in ('code','initial_sha256','scale_sha256','teacher_profile','enhancement','adapter'):
                    if tested[key]!=protocol[key]: raise ValueError('formal configuration differs from tested code/models')
                execute(run,'formal_training','routervc_mixed_train.py',
                    ['--root',data_root,'--cache',args.cache/'formal','--output',data_root/'router'],distributed=True)
                immutable(args.output/'complete.json',dict(complete=True,
                    protocol=digest(data_root/'protocol.json'),router=digest(data_root/'router/complete.json'),
                    labels=digest(data_root/'labels.complete.json'),evaluation_pending=True))
                run.update(phase='training_complete_evaluation_pending')
    except BaseException as error:
        save(run.root/'last_failure.json',dict(error=repr(error),progress=run.progress));raise
    finally:
        run.log_resources();run.stop.set();run.thread.join(timeout=3);run.lock.close()


def np_load(path):
    import numpy as np
    return np.load(path,allow_pickle=False)


if __name__=='__main__':main()
