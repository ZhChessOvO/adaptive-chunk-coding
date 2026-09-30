"""Exclusive tmux queue: smoke, paired frozen-generator training, evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from demo.chunk_enhancement_experiment import Run,read
from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
from demo.conditioned_generation_pipeline import execute
from demo.internal_condition_train import DEFAULT,INITIAL,CACHE
from demo.internal_condition_model import ConditionBranch,make_bundle,validate_bundle
from demo.internal_condition_decode import identities
from demo.conditioned_generation_evaluate import OLD,MODES
from demo import scalable_cooperation_format as fmt
from demo.patch_prefix_probe import load_frames,verify_artifacts
from demo.scalable_codec import atomic_json,atomic_bytes,file_hash
from demo.stage_c_seedvr2_lora_utils import atomic_torch_save


def variant(source,target,mode):
    value=torch.load(source,weights_only=True,map_location='cpu')
    value['branch_mode']=mode
    if target.exists():
        saved=torch.load(target,weights_only=True,map_location='cpu')
        assert validate_bundle(saved)==validate_bundle(value) and saved['metadata']==value['metadata']
        for section in ('state_dict','branch_state'):
            for k,v in value[section].items():
                torch.testing.assert_close(v,saved[section][k],rtol=0,atol=0)
    else: atomic_torch_save(target,value)
    return target


def assert_noise(a,b):
    aa=a['generation_runtime']['condition_windows'];bb=b['generation_runtime']['condition_windows']
    assert len(aa)==len(bb)>0
    for x,y in zip(aa,bb,strict=True):
        for k in ('before_vae','after_vae','before_diffusion','diffusion_noise','conditions'):
            assert x[k]==y[k], f'paired condition/noise mismatch: {k}'
        assert x['diffusion_noise']['dtype']=='torch.bfloat16'


def decode_point(root,row,name,mode,adapter,run,disabled=False):
    sid=row['sample']['sample_id'];oldname,directname=MODES[mode]
    original=OLD/sid/f'{oldname}.acsg'
    assert file_hash(original)==row['points'][oldname]['stream_sha256']
    control,inner,_,_=fmt.parse(original.read_bytes())
    control.update(identities(adapter));control['strength']=1.
    wire=fmt.wrap(inner,control)
    assert len(wire)==original.stat().st_size and fmt.parse(wire)[1]==inner
    dest=root/sid/f'{name}_{mode}';dest.mkdir(parents=True,exist_ok=True)
    stream=dest/'stream.acsg';complete=dest/'point.json'
    if complete.exists():
        record=read(complete);verify_artifacts(dest,record['artifacts'])
        assert stream.read_bytes()==wire and record['disabled']==disabled
        return dest,record['report']
    atomic_bytes(stream,wire)
    execute(run,f'{sid}_{name}_{mode}','internal_condition_decode.py',
        ['--stream',stream,'--output',dest,'--adapter',
         Path('/nonexistent/generator.pt') if disabled else adapter,
         *(['--disable-generation'] if disabled else [])],distributed=True)
    report=read(dest/'decode.json');pixels=load_frames(dest/'reconstruction.npz')
    enhanced=load_frames(OLD/sid/directname/'reconstruction.npz')
    verify_artifacts(OLD/sid/directname,row['points'][directname]['artifacts'])
    alpha=fmt.weights(enhanced.shape,control)
    np.testing.assert_array_equal(pixels[alpha==0],enhanced[alpha==0])
    assert report['total_bytes']==len(wire)==stream.stat().st_size
    assert report['feature_bytes_added']==0 and not report['source_frames_read']
    assert report['base_reference_unchanged'] and report['outside_generate_exact']
    if disabled:
        np.testing.assert_array_equal(pixels,enhanced)
        assert not report['generation_assets_validated']
    else:
        assert report['assets']==identities(adapter)
        runtime=report['generation_runtime']
        if runtime['branch_kind']=='internal':
            for obs in runtime['condition_windows']:
                assert obs['rgb_condition_unchanged'] and obs['internal_statistics']
                assert all(s['outside_coverage_exact'] for s in obs['internal_statistics'])
    atomic_json(complete,dict(report=report,disabled=disabled,
        artifacts={p.name:file_hash(p) for p in (stream,dest/'decode.json',dest/'reconstruction.npz')}))
    return dest,report


def train(run,name,mode,steps,stop=-1):
    dest=run.root/name.split('_attempt')[0]
    if (dest/'complete.json').exists():
        complete=read(dest/'complete.json');cfg=read(dest/'config.json')
        assert complete['steps']==steps and cfg['mode']==mode
        assert complete['adapter_sha256']==file_hash(dest/'adapter.pt')
        for f,h in cfg['code'].items():
            path=REPO/'demo'/f
            if file_hash(path)!=h:
                # The initial smoke's input/zero workers saw a removed EOF
                # blank line during import cleanup. No executable text changed.
                # Accept only that exact whitespace case for smoke, never for
                # formal training or resume configuration checks.
                assert run.progress['mode']=='smoke' and f=='internal_condition_train.py'
                assert hashlib.sha256(path.read_bytes().rstrip(b'\n')+b'\n').hexdigest()==h
        return
    execute(run,name,'internal_condition_train.py',
        ['--output',dest,'--mode',mode,'--steps',steps,'--stop-after',stop],distributed=True)


def assert_resume(a,b):
    for section in ('state_dict','branch_state'):
        for k,v in a['adapter'][section].items():
            torch.testing.assert_close(v,b['adapter'][section][k],rtol=0,atol=0)
    assert a['optimizer']['param_groups']==b['optimizer']['param_groups']
    for key,values in a['optimizer']['state'].items():
        for name,value in values.items():
            torch.testing.assert_close(value,b['optimizer']['state'][key][name],rtol=0,atol=0)


def smoke(run):
    root=run.root
    train(run,'internal_resume_attempt1','internal',6,3)
    train(run,'internal_resume_attempt2','internal',6)
    train(run,'internal_direct','internal',6)
    train(run,'input','input',6);train(run,'zero','zero',6)
    a=torch.load(root/'internal_resume/resume.pt',weights_only=True,map_location='cpu')
    b=torch.load(root/'internal_direct/resume.pt',weights_only=True,map_location='cpu')
    assert_resume(a,b)
    atomic_json(root/'resume_check.json',dict(exact=True,steps=6,optimizer_exact=True,
        source_note='input/zero smoke differed only by one EOF blank line; formal source pins remain exact'))
    cfg=read(root/'internal_direct/config.json')
    initial=torch.load(INITIAL,weights_only=True,map_location='cpu')
    adapters={'internal':root/'internal_resume/adapter.pt','input':root/'input/adapter.pt',
              'zero':root/'zero/adapter.pt'}
    for kind in ('input','internal'):
        path=root/f'initial_{kind}.pt'
        if not path.exists():
            torch.manual_seed(260930)
            atomic_torch_save(path,make_bundle(initial,ConditionBranch(kind),cfg,0))
        adapters[f'initial_{kind}']=path
    adapters['off']=variant(adapters['internal'],root/'off.pt','off')
    row=read(OLD/'summary.json')['results'][0]
    cases=[('off','full'),('initial_input','full'),('initial_internal','full'),
           ('internal','full'),('input','full'),('zero','full'),
           ('off','none'),('internal','none'),('input','none'),('zero','none'),
           ('repeat','full'),('G_off','full')]
    records={};directories={}
    for name,mode in cases:
        key='internal' if name in ('repeat','G_off') else name
        dest,report=decode_point(root/'decode_smoke',row,name,mode,adapters[key],run,name=='G_off')
        records[f'{name}_{mode}']=report;directories[f'{name}_{mode}']=dest
        if name!='G_off':
            assert_noise(records[f'off_{mode}'],report)
        expected=(f'off_{mode}' if name.startswith('initial_') or mode=='none' else
                  'internal_full' if name=='repeat' else None)
        if expected:
            np.testing.assert_array_equal(load_frames(dest/'reconstruction.npz'),
                load_frames(directories[expected]/'reconstruction.npz'))
    atomic_json(root/'decode_check.json',dict(complete=True,zero_initial_exact=True,
        no_E_exact=True,repeat_exact=True,G_off_without_weights_exact=True,results=records))


def pin_protocol(root,steps,smoke_root):
    assert read(smoke_root/'resume_check.json')['exact']
    smoke=read(smoke_root/'decode_check.json');assert smoke['complete']
    cfg=read(smoke_root/'internal_direct/config.json')
    for f,h in cfg['code'].items(): assert file_hash(REPO/'demo'/f)==h
    assert smoke['results']['internal_full']['assets']==identities(smoke_root/'internal_resume/adapter.pt')
    files=['internal_condition_model.py','internal_condition_train.py','internal_condition_decode.py',
           'internal_condition_pipeline.py','internal_condition_evaluate.py',
           'internal_condition_report.py','run_internal_condition.sh']
    protocol=dict(steps=steps,initial=file_hash(INITIAL),cache=file_hash(CACHE),
        code={f:file_hash(REPO/'demo'/f) for f in files},
        smoke_resume=file_hash(smoke_root/'resume_check.json'),
        smoke_decode=file_hash(smoke_root/'decode_check.json'))
    path=root/'queue_protocol.json'
    if path.exists(): assert read(path)==protocol,'pinned queue changed'
    else: atomic_json(path,protocol)


def main(args):
    run=Run(args);run.thread.start()
    try:
        with exclusive_native_evaluation(run):
            pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            if pids: raise RuntimeError(f'GPU busy: {pids}')
            if args.command=='smoke': smoke(run)
            if args.command in ('train','run'):
                pin_protocol(args.output,args.steps,DEFAULT.with_name(DEFAULT.name+'_smoke'))
                for mode in ('internal','zero','input'): train(run,mode,mode,args.steps)
                atomic_json(args.output/'training.complete.json',dict(steps=args.steps,utc=time.time()))
            if args.command in ('evaluate','run'):
                from demo.internal_condition_evaluate import evaluate
                evaluate(args.output,run)
            if args.command in ('report','run'):
                from demo.internal_condition_report import report
                report(args.output)
            run.update(phase='complete')
            atomic_json(args.output/f'{args.command}.complete.json',dict(complete=True,
                elapsed_seconds=time.monotonic()-run.started,code=file_hash(Path(__file__))))
    except BaseException as error:
        atomic_json(args.output/'last_failure.json',dict(error=repr(error),phase=run.progress));raise
    finally:
        run.log_resources();run.stop.set();run.thread.join(timeout=3)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['smoke','train','evaluate','report','run'])
    p.add_argument('--output',type=Path);p.add_argument('--steps',type=int,default=3000)
    p.add_argument('--max-hours',type=float,default=24.)
    args=p.parse_args()
    if args.output is None:
        args.output=DEFAULT.with_name(DEFAULT.name+'_smoke') if args.command=='smoke' else DEFAULT
    main(args)
