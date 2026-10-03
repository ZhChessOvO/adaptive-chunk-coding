"""Resumable real RouterVC routing evaluation on grouped development clips.

Candidate caches are validated reuse. The receiver always independently
decodes the actual wire; no isolated generated tiles are used as outputs.
"""
import argparse
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from demo.four_state_core import DEFAULT as TEACHER, ENHANCEMENT, ADAPTER, rows, verify, immutable_json
from demo.four_state_router import split
from demo.routervc_encode import select, subset_bank
from demo import routervc_format as fmt
from demo.routervc_decode import route
from demo.chunk_enhancement_experiment import Run, read
from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
from demo.conditioned_generation_pipeline import execute
from demo.scalable_codec import atomic_bytes, atomic_json, file_hash
from demo.scalable_format import frame_hash, parse
from demo.scalable_experiment import quality
from demo.stage_c_three_path_roi_probe import LPIPSAlex

DEFAULT=TEACHER.parent/'routervc_20261003'
ROUTER=TEACHER.parent/'a800_four_state_router_20261002'
CODE=('routervc_format.py','routervc_policy.py','routervc_decode.py','routervc_encode.py',
      'routervc_pipeline.py','run_routervc.sh')


def samples(smoke):
    data=torch.load(ROUTER/'data.pt',map_location='cpu',weights_only=True)
    _,valid=split(data['records']);selected=[]
    for domain in ('REDS','UVG'):
        selected.extend([data['records'][i]['sample_id'] for i in valid
                         if data['records'][i]['dataset']==domain][:(1 if smoke else 2)])
    lookup={r['sample_id']:r for r in rows()}
    return [lookup[s] for s in selected]


def prepare(root,smoke,run=None):
    selected=samples(smoke)
    protocol=dict(code={n:file_hash(REPO/'demo'/n) for n in CODE},teacher=file_hash(TEACHER/'complete.json'),
        router={a:file_hash(ROUTER/a/'model.pt') for a in ('context','local')},
        enhancement=file_hash(ENHANCEMENT),adapter=file_hash(ADAPTER),samples=[r['sample_id'] for r in selected],
        smoke=smoke,ratios=[.25,.5,.75],calls=[4,8],boundary_lambda=.004,
        role='Router grouped holdout development; E/G training data, UVG HoneyBee only',
        selection='fixed nested E packet ranking per G budget; shared G uses actual mixed Y',
        budgets='E packet bytes; constant RouterVC header charged separately; G calls not milliseconds')
    immutable_json(root/'protocol.json',protocol)
    common_config=fmt.make_config(ROUTER/'context/model.pt',ADAPTER)
    jobs=[]
    for row in selected:
        sid=row['sample_id'];folder=TEACHER/'encoded'/sid
        enc=read(folder/'complete.json');verify(folder,enc['artifacts'])
        er=read(TEACHER/'received'/sid/'E.json');verify(TEACHER/'received'/sid,er['artifacts'])
        bank=(folder/'packets.acse').read_bytes();parsed=parse(bank)
        with np.load(TEACHER/'received'/sid/'received_E.npz',allow_pickle=False) as cache:
            base,enhanced=cache['base'].copy(),cache['enhanced'].copy()
        assert frame_hash(base)==enc['expected_base_hash'] and frame_hash(enhanced)==enc['expected_E_hash']
        ebytes=sum(len(p.wire) for p in parsed.packets)
        methods=[(f'{a}_{v}',a,lam) for a in ('context','local') for v,lam in (('raw',0.),('smooth',.004))]
        if smoke:methods=[('context_raw','context',0.),('context_smooth','context',.004)]
        specs=[(m,a,lam,r,k) for m,a,lam in methods for k in ([4] if smoke else [4,8])
               for r in ([.5] if smoke else [.25,.5,.75])]
        specs += [('base','context',0.,0.,0)]
        specs += [('g_only','context',0.,0.,k) for k in ([4] if smoke else [4,8])]
        specs += [('e_only','context',0.,r,0) for r in ([.5] if smoke else [.25,.5,.75])]
        # One genuine prefix sequence even in smoke; keep all controls identical.
        if smoke:specs += [('context_smooth','context',.004,r,4) for r in (.25,.75)]
        streams={}
        for method,arm,lam,ratio,k in specs:
            if run is not None:run.check()
            target=root/sid/f'{method}_r{ratio:g}_g{k}'
            target.mkdir(parents=True,exist_ok=True)
            model=ROUTER/arm/'model.pt';budget=int(ebytes*ratio)
            began=time.monotonic()
            plan=select(bank,base,enhanced,model,budget,k,mode='prefix')
            for key in ('prediction_seconds','allocation_seconds'):plan.pop(key,None)
            chosen=plan['selected_indices']
            inner=subset_bank(bank,chosen)
            config=dict(common_config,router=file_hash(model),max_g=k,boundary_lambda=lam)
            wire=fmt.wrap(inner,config)
            assert len(inner)-parsed.base_end<=budget
            wp=target/'stream.rtvc'
            if wp.exists():assert wp.read_bytes()==wire
            else:atomic_bytes(wp,wire)
            mixed=base.copy()
            from demo.routervc_policy import grid_rois
            rois=grid_rois(base.shape[1],base.shape[2])
            for i in chosen:
                x,y,w,h=rois[i];mixed[:,y:y+h,x:x+w]=enhanced[:,y:y+h,x:x+w]
            predicted=route(base,mixed,parse(inner),config,model) if k else None
            identity=dict(sample_id=sid,dataset=row['sample']['dataset'],method=method,arm=arm,
                ratio=ratio,max_g=k,budget=budget,bytes=len(wire),stream_sha256=file_hash(wp),
                selected_E=chosen,source_path=row['pair_path'],source_hash=row['pair_hash'],
                folder=str(target),base_hash=frame_hash(base),enhanced_hash=frame_hash(mixed),
                expected_route=predicted,plan=plan,
                candidate_encode_seconds=enc['encoding_seconds'],candidate_cache_reused=True,
                sender_cost_scope='cached E encode only; original UF/feature extraction excluded from this evaluation')
            jp=target/'job.json'
            immutable_json(jp,identity)
            timing=target/'sender_timing.json'
            if not timing.exists():atomic_json(timing,dict(selection_and_simulation_seconds=time.monotonic()-began))
            jobs.append(identity);streams.setdefault((method,k),[]).append((ratio,wire))
        for key,versions in streams.items():
            ordered=sorted(versions)
            for (_,a),(_,b) in zip(ordered,ordered[1:]):
                assert b.startswith(a),f'not a genuine prefix {key}'
    immutable_json(root/'jobs.json',dict(jobs=jobs,real_prefixes=True))
    return jobs


def validate(folder,job):
    d=read(folder/'fresh/decode.json')
    assert d['stream_sha256']==job['stream_sha256'] and d['total_bytes']==job['bytes']
    assert not d['source_frames_read'] and d['base_reference_unchanged'] and d['outside_generate_exact']
    assert d['base_hash']==job['base_hash'] and d['generation_input_hash']==job['enhanced_hash']
    if job['expected_route'] is not None:assert d['route']==job['expected_route'],'sender/receiver route mismatch'
    with np.load(folder/'fresh/reconstruction.npz',allow_pickle=False) as p:
        assert frame_hash(p['reconstruction'])==d['output_hash']
        assert frame_hash(p['enhanced'])==job['enhanced_hash']
    return d


def evaluate(run,jobs,verify_only=False):
    metric=None;records=[]
    for index,job in enumerate(jobs):
        run.check();folder=Path(job['folder']);out=folder/'fresh';out.mkdir(exist_ok=True)
        if not (out/'decode.json').exists():
            if verify_only:raise RuntimeError('missing completed decoder')
            execute(run,f'decode_{index:03d}','routervc_decode.py',['--stream',folder/'stream.rtvc',
                '--output',out,'--enhancement',ENHANCEMENT,'--adapter',ADAPTER,
                '--router',ROUTER/job['arm']/'model.pt'],distributed=True)
        d=validate(folder,job)
        result=folder/'result.json'
        if result.exists():
            record=read(result);verify(folder,record['artifacts']);assert record['decode']==d
        else:
            if verify_only:raise RuntimeError('missing completed metrics')
            assert file_hash(Path(job['source_path']))==job['source_hash']
            with np.load(job['source_path'],allow_pickle=False) as p:source=p['source'].copy()
            with np.load(out/'reconstruction.npz',allow_pickle=False) as p:output=p['reconstruction'].copy()
            if metric is None:metric=LPIPSAlex(True)
            record=dict(job,quality=quality(source,output,metric),decode=d,
                artifacts={n:file_hash(folder/n) for n in ('stream.rtvc','fresh/decode.json','fresh/reconstruction.npz')})
            atomic_json(result,record)
        records.append(record);run.update(phase='verified' if verify_only else 'evaluating',completed=index+1,total=len(jobs))
    return records


def smoke_checks(run,jobs):
    root=run.root
    if (root/'smoke_audit.json').exists():return read(root/'smoke_audit.json')
    # Repeat one full G route in a genuinely new process. G-off uses absent G/router assets.
    job=next(j for j in jobs if j['method']=='context_smooth' and j['ratio']==.5)
    folder=Path(job['folder'])
    for name,extra in [('repeat',[]),('G_off',['--disable-generation'])]:
        out=root/'checks'/name;out.mkdir(parents=True,exist_ok=True)
        execute(run,name,'routervc_decode.py',['--stream',folder/'stream.rtvc','--output',out,
            '--enhancement',ENHANCEMENT,'--adapter',ADAPTER if name=='repeat' else '/missing/G.pt',
            '--router',ROUTER/'context/model.pt' if name=='repeat' else '/missing/router.pt',*extra],distributed=True)
        got=read(out/'decode.json');expected=job['enhanced_hash'] if name=='G_off' else read(folder/'fresh/decode.json')['output_hash']
        assert got['output_hash']==expected
        if name=='repeat':assert got['route']==job['expected_route']
        else:assert not got['generation_assets_validated'] and not got['shared_router_used']
    before={str(p):file_hash(p) for j in jobs for p in [Path(j['folder'])/'fresh/decode.json',Path(j['folder'])/'result.json']}
    evaluate(run,jobs,verify_only=True)
    assert all(file_hash(Path(p))==h for p,h in before.items())
    result=dict(complete=True,fresh_repeat_exact=True,missing_G_router_fallback=True,
        sender_receiver_routes_equal=True,real_prefixes=True,readonly_resume=True,
        preserved_results=len(before),protocol=file_hash(root/'protocol.json'))
    atomic_json(root/'smoke_audit.json',result);return result


def main(args):
    run=Run(args);run.thread.start();start=time.monotonic()
    try:
        with exclusive_native_evaluation(run):
            if args.command=='verify':
                jobs=read(args.output/'jobs.json')['jobs'];evaluate(run,jobs,verify_only=True)
                run.update(phase='verified');return
            jobs=prepare(args.output,args.command=='smoke',run)
            if args.command=='run':
                s=read(DEFAULT.with_name(DEFAULT.name+'_smoke')/'smoke_audit.json')
                assert s['complete'] and s['fresh_repeat_exact']
                sp=read(DEFAULT.with_name(DEFAULT.name+'_smoke')/'protocol.json')
                assert sp['code']==read(args.output/'protocol.json')['code']
            records=evaluate(run,jobs)
            if args.command=='smoke':smoke_checks(run,jobs)
            summary=dict(complete=True,records=records,role='component-training / Router-development clips; not independent test')
            immutable_json(args.output/'summary.json',summary)
            from demo.routervc_report import report
            report(args.output)
            if not (args.output/'complete.json').exists():
                atomic_json(args.output/'complete.json',dict(complete=True,points=len(records),
                    elapsed_seconds=time.monotonic()-start,summary=file_hash(args.output/'summary.json'),
                    protocol=file_hash(args.output/'protocol.json')))
            run.update(phase='complete',completed=len(records))
    except BaseException as error:
        atomic_json(args.output/'last_failure.json',dict(error=repr(error),phase=run.progress));raise
    finally:run.log_resources();run.stop.set();run.thread.join(timeout=3)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['smoke','run','verify'])
    p.add_argument('--output',type=Path);p.add_argument('--max-hours',type=float,default=12.)
    a=p.parse_args()
    if a.output is None:a.output=DEFAULT.with_name(DEFAULT.name+'_smoke') if a.command=='smoke' else DEFAULT
    main(a)
