"""Old/new global/local Routers on identical q2 banks and byte budgets.

Reuses already encoded q2 banks, never the old q1 packet selections. Uses the
unchanged tiled G, no per-region masks, no test-selected checkpoint. Old E=0
points may be reused only when their complete RTVC bytes match exactly.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from demo.routervc_fullview_probe import read,digest,save,immutable,verify_artifacts
from demo.routervc_light_teacher import OUTPUT, ENHANCEMENT, ADAPTER
from demo.routervc_light_packets import bank_info,subset_bank,predict_utility

REVISION = Path('/root/autodl-fs/DCVC/runs/routervc_revision_20261003')
ORIGINAL = REVISION/'visual_evaluation_recovered'
Q2 = Path('/root/autodl-fs/DCVC/runs/routervc_efficiency_20261004/q2_probe')
ARMS = ('global_local','local')
RATIOS = (0.,.25,.5)


def point_plan():
    return [dict(name=f'{version}_{arm}_e{ratio:g}_g8',version=version,arm=arm,ratio=ratio,max_g=8)
            for version in ('old','new') for arm in ARMS for ratio in RATIOS]


def protocol(root,models):
    from demo.routervc_visual_evaluate import completed_models,code_hashes
    from demo import routervc_visual_format as fmt
    old = read(ORIGINAL/'protocol.json')
    if not read(Q2/'complete.json')['complete'] or old['code'] != code_hashes():
        raise ValueError('fixed q2/old evaluation is not valid')
    for record in (old['enhancement'],old['adapter']):
        if digest(record['path']) != record['sha256']:raise ValueError('frozen component changed')
    weights = dict(old=completed_models(REVISION/'visual_router'),new=models)
    if weights['old'] != old['models']:raise ValueError('old Router model changed')
    for arm in ARMS:
        config = read(root/'router'/arm/'config.json')
        if config['epochs'] != 120 or config['data']['labels'] != digest(root/'teacher/labels.json'):
            raise ValueError('new Router is not the approved q2 training endpoint')
    configs = {f'{version}_{arm}':fmt.make_config(Path(weights[version]['arms'][arm]['path']),ADAPTER,
                   max_g=8,boundary_lambda=0.,seed=20261003) for version in ('old','new') for arm in ARMS}
    inputs = {}
    for entry in old['sources']:
        sid = entry['sample']['sample_id']; folder=Q2/'samples'/sid
        complete = read(folder/'prepare.complete.json');verify_artifacts(folder,complete['artifacts'])
        source = ORIGINAL/'samples'/sid/'source.npz'
        if digest(source) != complete['source_sha256']:raise ValueError('old q2 source changed')
        inputs[sid] = dict(bank_path=str(folder/'bank_q2.acse'),bank_sha256=digest(folder/'bank_q2.acse'),
                          source_path=str(source),source_sha256=digest(source))
    return dict(schema='routervc-light-router-evaluation-v1',models=weights,configs=configs,
        sources=old['sources'],inputs=inputs,points=point_plan(),qstep=2.,
        code={n:digest(REPO/'demo'/n) for n in ('routervc_light_packets.py','routervc_light_evaluate.py',
              'routervc_light_decode.py','run_routervc_light_evaluate.sh')},policy=fmt.policy_identity(),
        training_complete=digest(root/'router/complete.json'),q2_complete=digest(Q2/'complete.json'),
        budget='same actual q2 bank E-byte fraction, not equal realized rate or old q1 selection',
        freeze_UF_E_G=True,semantic_supervision=False,explicit_masks_bytes=0,
        independent_system_test=False)


def prepare_worker(root,sid):
    import numpy as np
    import torch
    from demo.chunk_enhancement_codec import configure_torch,decode_enhancement,load_model
    from demo.chunk_enhancement_experiment import codec
    from demo.scalable_codec import atomic_npz
    from demo.scalable_format import frame_hash
    p=read(root/'protocol.json');item=p['inputs'][sid];dest=root/'samples'/sid
    dest.mkdir(parents=True,exist_ok=True)
    if digest(item['bank_path']) != item['bank_sha256']:raise ValueError('fixed q2 bank changed')
    bank=Path(item['bank_path']).read_bytes();configure_torch();began=time.monotonic()
    with torch.inference_mode():
        all_e,detail,base=decode_enhancement(load_model(ENHANCEMENT),ENHANCEMENT,codec(),bank,return_base=True)
    atomic_npz(dest/'candidates.npz',base=base,all_E=all_e)
    save(dest/'prepare.complete.json',dict(complete=True,protocol=digest(root/'protocol.json'),
        bank_sha256=item['bank_sha256'],base_hash=frame_hash(base),all_E_hash=frame_hash(all_e),
        source_frames_read=False,seconds=time.monotonic()-began,
        artifacts={'candidates.npz':digest(dest/'candidates.npz')}))


def encode_sample(root,sid):
    import numpy as np
    from demo import routervc_visual_format as fmt
    from demo.routervc_visual_policy import load_model,allocate,route
    from demo.routervc_encode import compose_candidates
    from demo.scalable_codec import atomic_bytes
    from demo.scalable_format import frame_hash,parse
    p=read(root/'protocol.json');sample_dir=root/'samples'/sid
    prepared=read(sample_dir/'prepare.complete.json');verify_artifacts(sample_dir,prepared['artifacts'])
    if prepared['protocol'] != digest(root/'protocol.json'):raise ValueError('prepared protocol changed')
    item=p['inputs'][sid]
    if digest(item['bank_path'])!=item['bank_sha256']:raise ValueError('q2 bank changed')
    bank=Path(item['bank_path']).read_bytes();info=bank_info(bank)
    with np.load(sample_dir/'candidates.npz',allow_pickle=False) as data:
        base,all_e=data['base'].copy(),data['all_E'].copy()
    if frame_hash(base)!=prepared['base_hash'] or frame_hash(all_e)!=prepared['all_E_hash']:
        raise ValueError('candidate pixels changed')
    for version in ('old','new'):
        for arm in ARMS:
            # Predictions once per model/window; true prefix ranking is budget-independent.
            points=[v for v in p['points'] if v['version']==version and v['arm']==arm]
            model=None;utility=prediction=None;previous=None
            for point in points:
                folder=sample_dir/point['name'];folder.mkdir(exist_ok=True);done=folder/'encode.json'
                if done.exists():
                    result=read(done);verify_artifacts(folder,result['artifacts'])
                    if result['protocol']!=digest(root/'protocol.json'):raise ValueError('sender binding changed')
                else:
                    if model is None:
                        model=load_model(p['models'][version]['arms'][arm]['path'])
                        utility,prediction=predict_utility(bank,base,all_e,model)
                    budget=int(sum(info['e_bytes'])*point['ratio'])
                    plan=allocate(utility,info['e_bytes'],budget,8,mode='prefix')
                    inner=subset_bank(bank,plan['selected_indices']);config=p['configs'][f'{version}_{arm}']
                    mixed=compose_candidates(base,all_e,plan['selected_indices'],info['rois'])
                    expected=route(base,mixed,parse(inner),config,model,expected_policy=fmt.policy_identity())
                    wire=fmt.wrap(inner,config);atomic_bytes(folder/'stream.rtvc',wire)
                    if len(inner)-info['parsed'].base_end!=plan['e_packet_bytes']:
                        raise ValueError('realized E bytes do not match allocation')
                    result=dict(complete=True,protocol=digest(root/'protocol.json'),point=point,qstep=2.,
                        plan=plan,utility=utility.tolist(),prediction=prediction.tolist(),config=config,
                        expected_base_hash=frame_hash(base),expected_mixed_hash=frame_hash(mixed),
                        expected_shared_route=expected,stream_sha256=digest(folder/'stream.rtvc'),
                        explicit_E_mask_bytes=0,explicit_G_map_bytes=0,protection_mask_bytes=0,
                        artifacts={'stream.rtvc':digest(folder/'stream.rtvc')})
                    save(done,result)
                wire=(folder/'stream.rtvc').read_bytes()
                if previous is not None and not wire.startswith(previous):raise ValueError('lost literal q2 prefix')
                previous=wire
    immutable(sample_dir/'encode.complete.json',dict(complete=True,protocol=digest(root/'protocol.json'),
        prefix_pairs=8,artifacts={f'{pt["name"]}/encode.json':digest(sample_dir/pt['name']/'encode.json') for pt in p['points']}))


def check_point(folder,protocol,sample):
    from demo.routervc_qstep_probe import validate_decode
    result=read(folder/'result.json');verify_artifacts(folder,result['artifacts'])
    if result['protocol']!=digest(folder.parents[2]/'protocol.json'):raise ValueError('point binding changed')
    if result.get('reused'):
        src=Path(result['reference']);old=read(src/'result.json');verify_artifacts(src,old['artifacts'])
        if (digest(src/'result.json')!=result['reference_sha256']
                or (folder/'stream.rtvc').read_bytes()!=(src/'stream.rtvc').read_bytes()):
            raise ValueError('E0 reuse differs from original wire/evidence')
        for key in ('bytes','bpp','quality','decode'):
            if result[key]!=old[key]:raise ValueError('reused E0 metric changed')
        encoded=read(folder/'encode.json')
        if (result['decode']['route']!=encoded['expected_shared_route']
                or result['decode']['base_hash']!=encoded['expected_base_hash']
                or result['decode']['generation_input_hash']!=encoded['expected_mixed_hash']):
            raise ValueError('reused E0 receiver differs from current sender expectation')
    elif validate_decode(folder,protocol,sample)!=result['decode']:
        raise ValueError('completed decode record changed')
    return result


def recovery_checks(run,p,entry,*,verify_only=False):
    import numpy as np
    from demo.conditioned_generation_pipeline import execute
    sid=entry['sample']['sample_id'];parent=run.root/'samples'/sid/'new_global_local_e0.5_g8'
    formal=read(parent/'fresh/decode.json');paths=[]
    for mode in ('repeat','G_off'):
        dest=run.root/'checks'/sid/mode;dest.mkdir(parents=True,exist_ok=True)
        done=dest/'checked.json'
        if done.exists():
            saved=read(done);verify_artifacts(dest,saved['artifacts'])
            if saved['reference']!=digest(parent/'result.json'):raise ValueError('recovery reference changed')
        else:
            if verify_only:raise ValueError('read-only replay cannot run checks')
            if not (dest/'decode.json').exists():
                argv=['--stream',parent/'stream.rtvc','--output',dest,'--enhancement',ENHANCEMENT,
                      '--adapter',ADAPTER if mode=='repeat' else dest/'intentionally_absent_adapter',
                      '--router',p['models']['new']['arms']['global_local']['path'] if mode=='repeat'
                      else dest/'intentionally_absent_router']
                if mode=='G_off':argv+=['--disable-generation']
                execute(run,f'{sid}_{mode}','routervc_light_decode.py',argv,distributed=mode=='repeat')
            report=read(dest/'decode.json')
            if (report['source_frames_read'] or not report['outside_generate_exact']
                    or report['stream_sha256']!=formal['stream_sha256']
                    or report['total_bytes']!=formal['total_bytes']
                    or report['base_hash']!=formal['base_hash']
                    or report['generation_input_hash']!=formal['generation_input_hash']):
                raise ValueError('fresh repeat/G-off stream mismatch')
            if mode=='repeat':
                if report['output_hash']!=formal['output_hash'] or report['route']!=formal['route']:
                    raise ValueError('fresh repeat pixels/route mismatch')
                from demo.online_eg_eval_core import noise_pair
                if formal['generation_runtime'] is not None:
                    noise_pair(report,formal,same_condition=True)
            elif (report['generation_executed'] or report['shared_router_used']
                  or report['generation_assets_validated'] or report['output_hash']!=formal['generation_input_hash']):
                raise ValueError('G-off loaded generation models or modified Y')
            with np.load(dest/'reconstruction.npz',allow_pickle=False) as a, \
                    np.load(parent/'fresh/reconstruction.npz',allow_pickle=False) as b:
                np.testing.assert_array_equal(a['reconstruction'],b['reconstruction' if mode=='repeat' else 'enhanced'])
            save(done,dict(complete=True,reference=digest(parent/'result.json'),mode=mode,
                artifacts={n:digest(dest/n) for n in ('decode.json','reconstruction.npz')}))
        paths.append(str(done.relative_to(run.root)))
    return paths


def evaluate(run,p):
    import numpy as np
    import torch
    from demo.conditioned_generation_pipeline import execute
    from demo.routervc_qstep_probe import validate_decode
    from demo.scalable_experiment import quality
    from demo.stage_c_three_path_roi_probe import LPIPSAlex
    completed=(run.root/'complete.json').exists();records=[];metric=None;checks=[]
    if completed:verify_artifacts(run.root,read(run.root/'complete.json')['artifacts'])
    for entry in p['sources']:
        sample=entry['sample'];sid=sample['sample_id'];dest=run.root/'samples'/sid;item=p['inputs'][sid]
        if not (dest/'prepare.complete.json').exists():
            if completed:raise ValueError('read-only evaluation may not prepare')
            execute(run,f'candidates_{sid}',Path(__file__).name,['prepare','--output',run.root,'--sample-id',sid])
        verify_artifacts(dest,read(dest/'prepare.complete.json')['artifacts'])
        if not (dest/'encode.complete.json').exists():
            if completed:raise ValueError('read-only evaluation may not encode')
            encode_sample(run.root,sid)
        verify_artifacts(dest,read(dest/'encode.complete.json')['artifacts'])
        source=None
        for point in p['points']:
            run.check();folder=dest/point['name']
            if (folder/'result.json').exists():
                result=check_point(folder,p,sample)
            else:
                if completed:raise ValueError('read-only evaluation may not infer or score')
                if point['version']=='old' and point['ratio']==0:
                    src=REVISION/'visual_zero_E/samples'/sid/f'{point["arm"]}_e0_g8'
                    old=read(src/'result.json');verify_artifacts(src,old['artifacts'])
                    if (folder/'stream.rtvc').read_bytes()!=(src/'stream.rtvc').read_bytes():
                        raise ValueError('old E0 reuse is not byte-identical')
                    result=dict(old,point=point,protocol=digest(run.root/'protocol.json'),reused=True,
                        reference=str(src),reference_sha256=digest(src/'result.json'),
                        artifacts={n:digest(folder/n) for n in ('stream.rtvc','encode.json')},
                        timing_scope='historical byte-identical E0 decode, not measured again')
                else:
                    out=folder/'fresh';out.mkdir(exist_ok=True)
                    if not (out/'decode.json').exists():
                        execute(run,f'decode_{sid}_{point["name"]}','routervc_light_decode.py',
                            ['--stream',folder/'stream.rtvc','--output',out,'--enhancement',ENHANCEMENT,
                             '--adapter',ADAPTER,'--router',p['models'][point['version']]['arms'][point['arm']]['path']],distributed=True)
                    decoded=validate_decode(folder,p,sample)
                    run.update(phase='CPU_quality',sample=sid,point=point['name'])
                    if metric is None:torch.set_num_threads(4);metric=LPIPSAlex(True)
                    if source is None:
                        if digest(item['source_path'])!=item['source_sha256']:raise ValueError('source changed')
                        with np.load(item['source_path'],allow_pickle=False) as data:source=data['source'].copy()
                    with np.load(out/'reconstruction.npz',allow_pickle=False) as data:
                        scores=quality(source,data['reconstruction'],metric)
                        e_scores=quality(source,data['enhanced'],metric)
                    size=(folder/'stream.rtvc').stat().st_size
                    result=dict(complete=True,protocol=digest(run.root/'protocol.json'),point=point,
                        sample_id=sid,dataset=sample['dataset'],qstep=2.,bytes=size,bpp=8*size/math.prod(source.shape[:3]),
                        quality=scores,same_wire_G_off_quality=e_scores,decode=decoded,reused=False,
                        artifacts={n:digest(folder/n) for n in ('stream.rtvc','encode.json','fresh/decode.json','fresh/reconstruction.npz')})
                save(folder/'result.json',result)
                check_point(folder,p,sample)
            records.append(result);run.update(phase='real_q2_evaluation',completed=len(records),total=156)
        if sid==next(e['sample']['sample_id'] for e in p['sources'] if e['sample']['dataset']==sample['dataset']):
            checks+=recovery_checks(run,p,entry,verify_only=completed)
    if not completed:
        names=['protocol.json']+checks+[f'samples/{r["sample_id"]}/{r["point"]["name"]}/result.json' for r in records]
        save(run.root/'summary.json',dict(complete=True,points=len(records),fresh_points=sum(not r['reused'] for r in records),
            reused_points=sum(r['reused'] for r in records),literal_prefix_pairs=104,repeat_Goff_checks=checks,
            rows=[dict(sample_id=r['sample_id'],dataset=next(e['sample']['dataset'] for e in p['sources'] if e['sample']['sample_id']==r['sample_id']),
                       point=r['point'],bytes=r['bytes'],bpp=r['bpp'],quality=r['quality'],
                       same_wire_G_off_quality=r.get('same_wire_G_off_quality'),reused=r['reused']) for r in records]))
        names.append('summary.json')
        save(run.root/'complete.json',dict(complete=True,artifacts={n:digest(run.root/n) for n in names}))
    print('LIGHT_EVALUATION_VERIFIED' if completed else 'LIGHT_EVALUATION_COMPLETE',flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('run','prepare'))
    p.add_argument('--training',type=Path,default=OUTPUT)
    p.add_argument('--output',type=Path,default=OUTPUT/'evaluation');p.add_argument('--sample-id')
    p.add_argument('--max-hours',type=float,default=12.);p.add_argument('--wait-hours',type=float,default=36.)
    args=p.parse_args(argv)
    if not os.environ.get('TMUX'):raise RuntimeError('requires tmux')
    if args.command=='prepare':
        if not os.environ.get('ROUTERVC_LIGHT_EVALUATION_PARENT'):raise RuntimeError('parent required')
        return prepare_worker(args.output,args.sample_id)
    from demo.chunk_enhancement_experiment import Run
    from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
    from demo.routervc_visual_evaluate import wait_for_models,PhaseDeadline
    run=Run(SimpleNamespace(output=args.output,command='light_evaluation',max_hours=args.max_hours+args.wait_hours))
    run.thread.start();deadline=PhaseDeadline(run.check);run.check=deadline.check
    try:
        deadline.begin('wait_for_training',args.wait_hours*3600)
        models=wait_for_models(run,args.training/'router')
        deadline.begin('evaluation',args.max_hours*3600)
        prot=protocol(args.training,models);immutable(run.root/'protocol.json',prot)
        os.environ['ROUTERVC_LIGHT_EVALUATION_PARENT']=str(os.getpid())
        with nullcontext() if (run.root/'complete.json').exists() else exclusive_native_evaluation(run):evaluate(run,prot)
    except BaseException as error:
        save(run.root/'last_failure.json',dict(error=repr(error),progress=run.progress));raise
    finally:
        run.log_resources();run.stop.set();run.thread.join(timeout=3);run.lock.close()


if __name__=='__main__':main()
