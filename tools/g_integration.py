"""Small, resumable frozen B/E/Rg/fusion evaluation; no training or promotion."""
import argparse
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

from demo.chunk_enhancement_codec import atomic_torch, configure_torch
from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
from demo.routervc_fullview_probe import read, digest, immutable, verify_artifacts
from demo.routervc_mixed_queue import exact
from demo.scalable_codec import atomic_bytes, atomic_json
from demo.scalable_format import frame_hash
from demo.scalable_experiment import now, quality
from routervc.generation import integration as g
from routervc.generation.texture_data import REPO
from routervc.latent import generation
from routervc.fusion import boundaries
from routervc.fusion.blend import geometry
from tools.cooperative_review import source_rgb
from tools.texture_clean import CleanRun
from tools.texture_repair import execute

DEFAULT=Path('/root/autodl-fs/DCVC/runs/routervc_g_integration_20261016')
PRIOR=Path('/root/autodl-fs/DCVC/runs/routervc_cooperative_sender_review_20261011')
ARMS=('legacy','initial','control')
CASES=('reds-val-000-f000-n17-fullview','uvg-beauty-f000-historical-evaluation-crop',
       'reds-val-005-f000-n17-fullview','reds-val-020-f000-n17-fullview',
       'uvg-shakendry-f000-historical-evaluation-crop','uvg-honeybee-f160-left')
MODEL_PATHS=dict(legacy=generation.ADAPTER,
    initial=Path('/root/autodl-fs/DCVC/runs/routervc_texture_context_20261015/training/formal/mixed/adapter.pt'),
    control=Path('/root/autodl-fs/DCVC/runs/routervc_g_objectives_20261015/trial/formal/control/adapter.pt'))
CODE=('routervc/generation/integration.py','tools/g_integration.py','tools/g_integration_decode.py',
      'tools/run_g_integration.sh','tools/test_g_integration.py')


def checked(path, sha):
    if digest(path)!=sha: raise ValueError('bound file changed: '+str(path))


def prepare(root):
    if (root/'protocol.json').exists(): return check_protocol(root)
    old=read(PRIOR/'protocol.json')
    for name,sha in old['code'].items(): checked(REPO/name,sha)
    if old['receiver_profile']!=g.stream.identity(): raise ValueError('historical profile changed')
    models={}; legacy=torch.load(MODEL_PATHS['legacy'],weights_only=True,map_location='cpu')
    for arm,path in MODEL_PATHS.items():
        original_sha=digest(path)
        if arm=='legacy':
            if original_sha!=old['teacher']['G_assets']['lora']: raise ValueError('legacy changed')
            exported=path
        else:
            if read(path.parent/'complete.json')['adapter_sha256']!=original_sha:
                raise ValueError('candidate not the completed training checkpoint')
            candidate=torch.load(path,weights_only=True,map_location='cpu')
            bundle=g.compatible_bundle(candidate,legacy); exported=root/'models'/arm/'adapter.pt'
            if not exported.exists(): atomic_torch(exported,bundle)
            exact(torch.load(exported,weights_only=True,map_location='cpu'),bundle)
        models[arm]=dict(path=str(exported),sha256=digest(exported),original_path=str(path),
            original_sha256=original_sha,assets=generation.assets(exported))
    if models['legacy']['assets']!=old['teacher']['G_assets']: raise ValueError('shared G assets changed')
    jobs=[]; inputs={str(PRIOR/'protocol.json'):digest(PRIOR/'protocol.json')}
    for sid in CASES:
        job=copy.deepcopy(next(j for j in old['jobs'] if j['row']['sample_id']==sid and j['cap']==.5))
        point_path=PRIOR/job['cohort']/sid/'e050/source/point.json'; point=read(point_path)
        receipt_path=Path(point['receive'])/'complete.json'; receipt=read(receipt_path)
        wire=Path(point['wire']); checked(wire,point['stream_sha256'])
        if not point['complete'] or not receipt['complete']: raise ValueError('old point incomplete')
        for path in (point_path,receipt_path,wire): inputs[str(path)]=digest(path)
        job.update(point=point,receipt=receipt)
        jobs.append(job)
    config=dict(format='frozen_chain_G_v1',created=now(),jobs=jobs,models=models,inputs=inputs,
        teacher=old['teacher'],profile=g.identity(),code={**old['code'],**{n:digest(REPO/n) for n in CODE}},
        no_training=True,no_promotion=True,no_masks=True,scope='six previously used E50 windows, not full benchmark')
    immutable(root/'protocol.json',config)
    return config


def check_protocol(root):
    p=read(root/'protocol.json')
    for name,sha in p['code'].items(): checked(REPO/name,sha)
    for path,sha in p['inputs'].items(): checked(path,sha)
    for arm,spec in p['models'].items():
        checked(spec['path'],spec['sha256']); checked(spec['original_path'],spec['original_sha256'])
        if generation.assets(Path(spec['path']))!=spec['assets']: raise ValueError('G assets changed')
        if arm!='legacy':
            export=torch.load(spec['path'],weights_only=True,map_location='cpu')
            original=torch.load(spec['original_path'],weights_only=True,map_location='cpu')
            legacy=torch.load(p['models']['legacy']['path'],weights_only=True,map_location='cpu')
            exact(export,g.compatible_bundle(original,legacy))
    checked(p['teacher']['receiver']['path'],p['teacher']['receiver']['sha256'])
    if p['profile']!=g.identity(): raise ValueError('receiver code changed')
    return p


def conditions(report):
    return [(r['crop'],r['core'],[{k:c[k] for k in ('before_vae','after_vae','conditions',
        'diffusion_noise','effective_condition','before_diffusion','rgb_condition_unchanged')}
        for c in r['runtime']['condition_windows']]) for r in report['generation_runtime']]


def validate(report, historical, *, disabled=False, legacy=False):
    for key in ('base_hash','enhanced_hash','actual_bytes','bpp','header_bytes','mask_bytes'):
        if report[key]!=historical[key]: raise ValueError('fixed-chain contract changed: '+key)
    if (report['source_frames_read'] or report['sender_router_loaded'] or report['mask_bytes']!=0
            or report['generation_disabled']!=disabled or not report['sender_checkpoint_access_guard']):
        raise ValueError('source-free decoder contract failed')
    if disabled:
        if report['output_hash']!=report['enhanced_hash'] or report['generated'] or report['selected'] is not None:
            raise ValueError('G-off did not return received Y')
    else:
        if report['selected']!=historical['selected'] or report['generated']!=historical['generated']:
            raise ValueError('R_g decision changed')
        if conditions(report)!=conditions(historical): raise ValueError('G geometry/condition/noise changed')
        if legacy and report['output_hash']!=historical['output_hash']: raise ValueError('legacy pixel replay differs')


def fresh(run,root,config,job,arm,tag='main',stop=0):
    sid=job['row']['sample_id']; dest=root/'samples'/sid/arm/tag
    model=config['models'][arm]
    inner,old=g.stream.parse(Path(job['point']['wire']).read_bytes())
    wire=g.wrap(inner,old['receiver_sha256'],generation.asset_hash(model['assets']),old['max_g'],old['seed'])
    if len(wire)!=job['point']['actual_bytes'] or g.parse(wire)[0]!=inner:
        raise ValueError('B/E bytes or envelope size changed')
    path=dest/'stream.rvlg'; dest.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=wire: raise ValueError('stream changed')
    else: atomic_bytes(path,wire)
    disabled=tag=='off'
    immutable(dest/'request.json',dict(stream=digest(path),model=model['sha256'],disabled=disabled,
        receiver=config['teacher']['receiver'],worker=digest(REPO/'tools/g_integration_decode.py')))
    if not (dest/'complete.json').exists():
        options=['--stream',path,'--output',dest,'--receiver',config['teacher']['receiver']['path'],
                 '--adapter',model['path'],'--stop-regions',stop]
        if disabled: options.append('--disable-generation')
        execute(run,'_'.join((sid,arm,tag,str(stop))),'tools.g_integration_decode',options)
    if stop: return dest
    report=read(dest/'complete.json'); verify_artifacts(dest,report['artifacts'])
    if report['stream']!=digest(path): raise ValueError('stream receipt changed')
    validate(report,job['receipt'],disabled=disabled,legacy=arm=='legacy' and not disabled)
    with np.load(dest/'pixels.npz',allow_pickle=False) as z:
        base,received,result=(z[k].copy() for k in ('base','enhanced','reconstruction'))
    for key,pixels in (('base_hash',base),('enhanced_hash',received),('output_hash',result)):
        if frame_hash(pixels)!=report[key]: raise ValueError('pixel hash differs')
    support=geometry(received.shape,report['generated'])[0]
    np.testing.assert_array_equal(result[:,~support],received[:,~support])
    return dest


def score(root,job):
    from demo.stage_c_three_path_roi_probe import LPIPSAlex
    from tools.texture_repair_review import temporal_phase
    from demo.routervc_policy import grid_rois
    folder=root/'samples'/job['row']['sample_id']; target=folder/'scores.json'
    if target.exists():
        point=read(target); verify_artifacts(root,point['artifacts']); return point
    source=source_rgb(job['row']); metric=LPIPSAlex(True); metrics={}; receipts={}
    arrays={}
    for arm in ARMS:
        dest=folder/arm/'main'; report=read(dest/'complete.json'); verify_artifacts(dest,report['artifacts'])
        receipts[arm]=report
        with np.load(dest/'pixels.npz',allow_pickle=False) as z:
            arrays[arm]=z['reconstruction'].copy()
            if arm=='legacy': arrays['received']=z['enhanced'].copy()
    edges=boundaries.edges(source.shape,job['receipt']['detail']['received_regions'],job['receipt']['generated'])
    for arm,output in arrays.items():
        phase=[]
        for i in job['receipt']['generated']:
            x,y,w,h=grid_rois(*source.shape[1:3])[i]
            phase.append(temporal_phase(output[:,y:y+h,x:x+w],source[:,y:y+h,x:x+w]))
        metrics[arm]=dict(quality=quality(source,output,metric),
            G_core_phase={k:float(np.mean([r[k] for r in phase])) for k in phase[0] if k!='period'},
            boundaries={c:boundaries.aggregate([boundaries.measure(source,output,e)
                for e in edges if e['category']==c]) for c in boundaries.CATEGORIES})
    del metric
    files=[p for a in ARMS for p in (folder/a/'main').rglob('*') if p.is_file() and
           (p.name in ('pixels.npz','complete.json','raw.npz','request.json','binding.json','stream.rvlg'))]
    point=dict(sample_id=job['row']['sample_id'],cohort=job['cohort'],dataset=job['row']['dataset'],
        shape=list(source.shape),actual_bytes=job['point']['actual_bytes'],bpp=job['point']['bpp'],
        G_regions=job['receipt']['generated'],E_regions=job['receipt']['detail']['received_regions'],
        metrics=metrics,resources={a:{k:receipts[a][k] for k in
            ('seconds','peak_cuda_allocated_bytes','peak_cuda_reserved_bytes','resumed_regions')} for a in ARMS},
        artifacts={str(p.relative_to(root)):digest(p) for p in files})
    atomic_json(target,point); return point


def verify(root):
    cfg=check_protocol(root); done=read(root/'complete.json'); verify_artifacts(root,done['artifacts'])
    smoke=read(root/'smoke.json'); verify_artifacts(root,smoke['artifacts'])
    for job in cfg['jobs']:
        folder=root/'samples'/job['row']['sample_id']
        point=read(folder/'scores.json'); verify_artifacts(root,point['artifacts'])
        for arm in ARMS:
            dest=folder/arm/'main'; r=read(dest/'complete.json'); verify_artifacts(dest,r['artifacts'])
            validate(r,job['receipt'],legacy=arm=='legacy')
    return done


def main(args):
    if not os.environ.get('TMUX'): raise RuntimeError('tmux required')
    root=args.output
    if args.verify_only or (root/'complete.json').exists():
        print(json.dumps(verify(root)),flush=True); return
    run=CleanRun(SimpleNamespace(output=root,command='integration',max_hours=8))
    run.thread.start(); run.log_resources(); configure_torch()
    try:
        with exclusive_native_evaluation(run):
            cfg=prepare(root)
            if not (root/'smoke.json').exists():
                records={}; resumed={}
                for job in cfg['jobs'][:2]:
                    sid=job['row']['sample_id']
                    for arm in ARMS:
                        main_dest=fresh(run,root,cfg,job,arm)
                        repeat=root/'samples'/sid/arm/'repeat'
                        if arm=='control' and not (repeat/'intentional_stop.json').exists() and not (repeat/'complete.json').exists():
                            fresh(run,root,cfg,job,arm,'repeat',stop=2)
                        preserved={str(p):digest(p) for p in repeat.glob('g*/complete.json')}
                        repeated=fresh(run,root,cfg,job,arm,'repeat')
                        if read(repeated/'complete.json')['output_hash']!=read(main_dest/'complete.json')['output_hash']:
                            raise ValueError('actual model reload/restart output differs')
                        if any(digest(p)!=s for p,s in preserved.items()): raise ValueError('resume rewrote regions')
                        if arm=='control':
                            if not preserved or read(repeated/'complete.json')['resumed_regions']!=2:
                                raise ValueError('actual partial-region resume not exercised')
                            resumed[sid]=preserved
                        for dest in (main_dest,repeated):
                            records[str((dest/'complete.json').relative_to(root))]=digest(dest/'complete.json')
                    off=fresh(run,root,cfg,job,'legacy','off')
                    records[str((off/'complete.json').relative_to(root))]=digest(off/'complete.json')
                immutable(root/'smoke.json',dict(complete=True,actual_reload_repeats=6,
                    exact_legacy_historical_replays=2,partial_region_resumes=resumed,G_off_exact=2,artifacts=records))
            verify_artifacts(root,read(root/'smoke.json')['artifacts'])
            if args.smoke_only: return
            points=[]
            for job in cfg['jobs']:
                for arm in ARMS: fresh(run,root,cfg,job,arm)
                points.append(score(root,job))
                run.update(phase='scored',completed=len(points),total=6,sample=job['row']['sample_id'])
            atomic_json(root/'summary.json',dict(complete=True,cases=points,no_promotion=True))
            names=['protocol.json','smoke.json','summary.json']+[
                'samples/'+j['row']['sample_id']+'/scores.json' for j in cfg['jobs']]
            atomic_json(root/'complete.json',dict(complete=True,utc=now(),new_full_outputs=18,
                reload_repeats=6,no_training=True,production_changed=False,
                artifacts={n:digest(root/n) for n in names}))
            verify(root); run.update(phase='complete_review_required')
    except BaseException as error:
        atomic_json(root/'last_failure.json',dict(utc=now(),error=repr(error),progress=run.progress)); raise
    finally:
        run.log_resources(); run.stop.set(); run.thread.join(timeout=3); run.lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--smoke-only',action='store_true'); p.add_argument('--verify-only',action='store_true')
    main(p.parse_args())
