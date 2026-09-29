"""Single-GPU tmux queue for adapter-only training and paired content controls."""
import argparse
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
from demo.feature_interface_train import DEFAULT,INITIAL,CACHE,make_bundle
from demo.feature_interface_model import FeatureInterface
from demo.feature_interface_decode import identities
from demo.feature_condition_pipeline import DEFAULT as PREVIOUS
from demo.feature_condition_smoke import OLD
from demo import scalable_cooperation_format as fmt
from demo.patch_prefix_probe import load_frames
from demo.scalable_codec import atomic_json,atomic_bytes,file_hash
from demo.stage_c_seedvr2_lora_utils import atomic_torch_save


def variant(source,target,mode):
    value=torch.load(source,weights_only=True,map_location='cpu')
    value.update(interface_mode=mode,feature_enabled=mode!='off')
    if target.exists():
        saved=torch.load(target,weights_only=True,map_location='cpu')
        assert saved['interface_mode']==mode and saved['metadata']==value['metadata']
        for section in ('state_dict','feature_state'):
            for k,v in value[section].items():
                torch.testing.assert_close(v,saved[section][k],rtol=0,atol=0)
    else: atomic_torch_save(target,value)
    return target


def decode_smoke(root,run):
    old=read(OLD/'summary.json')['results'][0]; sid=old['sample']['sample_id']
    trained=root/'actual_resume/adapter.pt'
    initial=torch.load(INITIAL,weights_only=True,map_location='cpu')
    config=read(root/'actual_resume/config.json')
    zero=root/'zero_initial.pt'
    if not zero.exists():
        torch.manual_seed(260930)
        atomic_torch_save(zero,make_bundle(initial,FeatureInterface(),'actual',config,0))
    without=variant(trained,root/'ablations/without.pt','zero')
    shuffled=variant(trained,root/'ablations/shuffled.pt','shuffle')
    cases=[('zero_full',zero,'full',False),('actual_full',trained,'full',False),
        ('repeat_full',trained,'full',False),('actual_none',trained,'none',False),
        ('without_full',without,'full',False),('shuffle_full',shuffled,'full',False),
        ('coverage_full',root/'zero_smoke/adapter.pt','full',False),
        ('G_off',trained,'full',True)]
    records={}
    for name,adapter,prefix,disabled in cases:
        dest=root/'decode_smoke'/name; dest.mkdir(parents=True,exist_ok=True)
        oldname='cooperate_l05' if prefix=='full' else 'generate_l05'
        control,inner,_,_=fmt.parse((OLD/sid/f'{oldname}.acsg').read_bytes())
        control.update(identities(adapter)); control['strength']=1.
        wire=fmt.wrap(inner,control); stream=dest/'stream.acsg'
        assert len(wire)==(OLD/sid/f'{oldname}.acsg').stat().st_size
        atomic_bytes(stream,wire)
        execute(run,name,'feature_interface_decode.py',['--stream',stream,'--output',dest,
            '--adapter',Path('/nonexistent/generator.pt') if disabled else adapter,
            *(['--disable-generation'] if disabled else [])],distributed=True)
        pixels=load_frames(dest/'reconstruction.npz'); report=read(dest/'decode.json')
        if name=='zero_full' or prefix=='none':
            np.testing.assert_array_equal(pixels,load_frames(PREVIOUS/'evaluation'/sid/f'rgb_{prefix}/reconstruction.npz'))
        if name=='repeat_full':
            np.testing.assert_array_equal(pixels,load_frames(root/'decode_smoke/actual_full/reconstruction.npz'))
        if disabled:
            np.testing.assert_array_equal(pixels,load_frames(OLD/sid/'enhance_q1/reconstruction.npz'))
            assert not report['generation_assets_validated']
        else:
            for stats in report['generation_runtime']['condition_statistics']:
                assert stats['outside_coverage_exact']
        assert report['feature_bytes_added']==0 and not report['source_frames_read']
        assert report['total_bytes']==stream.stat().st_size
        records[name]=dict(stream_sha256=file_hash(stream),report=report)
    atomic_json(root/'decode_check.json',dict(complete=True,zero_initial_exact=True,
        no_E_exact=True,repeat_exact=True,G_off_without_weights_exact=True,results=records))


def train(run,name,mode,steps,stop=-1):
    dest=run.root/name.split('_attempt')[0]
    if (dest/'complete.json').exists():
        complete=read(dest/'complete.json'); cfg=read(dest/'config.json')
        assert complete['steps']==steps and cfg['mode']==mode
        assert complete['adapter_sha256']==file_hash(dest/'adapter.pt')
        return
    execute(run,name,'feature_interface_train.py',['--output',dest,'--mode',mode,
        '--steps',steps,'--stop-after',stop],distributed=True)


def pin_protocol(root,steps,smoke):
    config=read(smoke/'actual_resume/config.json')
    for name,sha in config['code'].items():
        assert file_hash(REPO/'demo'/name)==sha, 'smoke-tested training code changed'
    for case in read(smoke/'decode_check.json')['results'].values():
        assert case['report']['total_bytes'] > 0
    assert read(smoke/'decode_check.json')['results']['actual_full']['report']['assets']==identities(
        smoke/'actual_resume/adapter.pt'), 'smoke-tested receiver profile changed'
    code={name:file_hash(REPO/'demo'/name) for name in (
        'feature_interface_pipeline.py','feature_interface_train.py','feature_interface_model.py',
        'feature_interface_decode.py','feature_interface_evaluate.py','feature_interface_report.py',
        'feature_condition_model.py','feature_condition_train.py','feature_condition_decode.py',
        'conditioned_generation_pipeline.py','run_feature_interface.sh')}
    protocol=dict(version=1,steps=steps,initial=file_hash(INITIAL),cache=file_hash(CACHE),code=code,
        smoke_resume=file_hash(smoke/'resume_check.json'),smoke_decode=file_hash(smoke/'decode_check.json'),
        comparisons='actual vs zero; 38 fresh decodes',
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
    path=root/'queue_protocol.json'
    if path.exists():
        previous=read(path)
        assert {k:v for k,v in previous.items() if k!='git_commit'}=={
            k:v for k,v in protocol.items() if k!='git_commit'}, 'queue inputs changed'
    else: atomic_json(path,protocol)


def main(args):
    run=Run(args); run.thread.start()
    try:
        with exclusive_native_evaluation(run):
            pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            if pids: raise RuntimeError(f'GPU busy: {pids}')
            if args.command=='smoke':
                train(run,'actual_resume_attempt1','actual',6,3)
                train(run,'actual_resume_attempt2','actual',6)
                train(run,'actual_direct','actual',6)
                a=torch.load(args.output/'actual_resume/resume.pt',weights_only=True,map_location='cpu')
                b=torch.load(args.output/'actual_direct/resume.pt',weights_only=True,map_location='cpu')
                for section in ('state_dict','feature_state'):
                    for k,v in a['adapter'][section].items():
                        torch.testing.assert_close(v,b['adapter'][section][k],rtol=0,atol=0)
                assert a['optimizer']['param_groups']==b['optimizer']['param_groups']
                for k,v in a['optimizer']['state'].items():
                    for name,tensor in v.items():
                        torch.testing.assert_close(tensor,b['optimizer']['state'][k][name],rtol=0,atol=0)
                train(run,'zero_smoke','zero',6)
                atomic_json(args.output/'resume_check.json',dict(exact=True,frozen_lora_exact=True,
                    interface_exact=True,optimizer_exact=True,steps=6))
                decode_smoke(args.output,run)
            elif args.command in ('train','run'):
                smoke=DEFAULT.with_name(DEFAULT.name+'_smoke')
                assert read(smoke/'resume_check.json')['exact'] and read(smoke/'decode_check.json')['complete']
                pin_protocol(args.output,args.steps,smoke)
                for name in ('actual','zero'): train(run,name,name,args.steps)
                atomic_json(args.output/'training.complete.json',dict(steps=args.steps,utc=time.time()))
            if args.command in ('evaluate','run'):
                from demo.feature_interface_evaluate import evaluate
                evaluate(args.output,run)
            if args.command in ('report','run'):
                from demo.feature_interface_report import report
                report(args.output)
            run.update(phase='complete')
            atomic_json(args.output/f'{args.command}.complete.json',dict(command=args.command,
                code=file_hash(Path(__file__)),elapsed_seconds=time.monotonic()-run.started))
    except BaseException as error:
        atomic_json(args.output/'last_failure.json',dict(error=repr(error),phase=run.progress)); raise
    finally:
        run.log_resources(); run.stop.set(); run.thread.join(timeout=3)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['smoke','train','evaluate','report','run'])
    p.add_argument('--output',type=Path); p.add_argument('--steps',type=int,default=3000)
    p.add_argument('--max-hours',type=float,default=12.)
    args=p.parse_args()
    if args.output is None:
        args.output=DEFAULT.with_name(DEFAULT.name+'_smoke') if args.command=='smoke' else DEFAULT
    main(args)
