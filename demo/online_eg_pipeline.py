"""tmux smoke, exact resume and paired training; no automatic post-train evaluation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from demo.chunk_enhancement_experiment import Run,read
from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
from demo.conditioned_generation_pipeline import execute
from demo.online_eg_train import DEFAULT,INITIAL,PATCH,CODE
from demo.patch_prefix_probe import load_frames
from demo.scalable_codec import atomic_json,file_hash
from demo.scalable_format import frame_hash


def exact(a,b):
    if torch.is_tensor(a): torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: exact(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b,strict=True): exact(x,y)
    else: assert a==b


def train(run,name,mode,steps,stop=-1):
    dest=run.root/name.split('_attempt')[0]
    if (dest/'complete.json').exists():
        c=read(dest/'complete.json');cfg=read(dest/'config.json')
        assert c['steps']==steps and c['mode']==mode
        for key in ('adapter','enhancement'): assert c[key+'_sha256']==file_hash(dest/f'{key}.pt')
        for name,digest in cfg['code'].items(): assert file_hash(REPO/'demo'/name)==digest
        return
    execute(run,name,'online_eg_train.py',['--output',dest,'--mode',mode,'--steps',steps,'--stop-after',stop],distributed=True)


def audit(root,joint='joint',fixed='fixed',steps=3000):
    logs=[]
    for name,mode in ((joint,'joint'),(fixed,'fixed')):
        folder=root/name; cfg=read(folder/'config.json'); info=read(folder/'complete.json')
        assert info['steps']==steps and info['enhancement_changed']==(mode=='joint')
        for key in ('adapter','enhancement'): assert info[key+'_sha256']==file_hash(folder/f'{key}.pt')
        for filename,h in cfg['code'].items(): assert h==file_hash(REPO/'demo'/filename)
        rows=[json.loads(l) for l in (folder/'steps.jsonl').read_text().splitlines()]
        assert [r['step'] for r in rows]==list(range(1,steps+1))
        for row in rows:
            assert np.isfinite(row['loss']) and row['lora_gradient_norm']>0
            if mode=='joint' and row['condition']!='none':
                assert row['e_gradient_norm']>0 and row['g_to_e_rgb_gradient_norm']>0
            else: assert row['e_gradient_norm']==0
        logs.append(rows)
    for a,b in zip(*logs,strict=True):
        for k in ('step','sample','dataset','condition','crop','packet_ids','diffusion_noise_identity','learning_rate'):
            assert a[k]==b[k],f'non-paired {k}'
    assert all(v>0 for v in logs[0][0]['first_lpips_path_gradients'])
    result=dict(complete=True,steps=steps,paired=True,lpips_reaches_E_analysis_and_synthesis=True,
        all_no_e_steps_skip_e=True,initial_lora=file_hash(INITIAL),initial_e=file_hash(PATCH))
    atomic_json(root/'training_audit.json',result)
    return result


def stream_smoke(run):
    root=run.root/'streams'
    enhancement=run.root/'joint_resume/enhancement.pt';adapter=run.root/'joint_resume/adapter.pt'
    if not (root/'encode.json').exists():
        execute(run,'encode_smoke','online_eg_stream_smoke.py',
                ['--output',root,'--enhancement',enhancement,'--adapter',adapter])
    report=read(root/'encode.json')
    assert report['enhancement']==file_hash(enhancement) and report['adapter']==file_hash(adapter)
    results=[]
    # Two domains, all three prefixes, exact G-off plus optional-generation replay.
    for row in report['results']:
        dest=Path(row['stream']).parent/(row['mode']+'_Goff')
        execute(run,row['dataset']+'_'+row['mode']+'_Goff','online_eg_decode.py',
            ['--stream',row['stream'],'--output',dest,'--enhancement',enhancement,
             '--adapter','/nonexistent/generator.pt','--disable-generation'],distributed=True)
        d=read(dest/'decode.json');pixels=load_frames(dest/'reconstruction.npz')
        assert d['total_bytes']==row['bytes'] and d['source_frames_read'] is False
        assert d['generation_assets_validated'] is False
        assert frame_hash(pixels)==row['enhanced_hash']
        results.append(d)
    for mode in ('full','none'):
        outputs=[]
        for suffix in ('G','repeat'):
            dest=root/'REDS'/f'{mode}_{suffix}'
            execute(run,mode+'_'+suffix,'online_eg_decode.py',
                ['--stream',root/'REDS'/f'{mode}.acsg','--output',dest,
                 '--enhancement',enhancement,'--adapter',adapter],distributed=True)
            d=read(dest/'decode.json');assert d['source_frames_read'] is False and d['outside_generate_exact']
            outputs.append(load_frames(dest/'reconstruction.npz'));results.append(d)
        np.testing.assert_array_equal(*outputs)
    atomic_json(root/'decode_check.json',dict(complete=True,fresh_decodes=len(results),repeat_exact=True,
        no_e_generation_works=True,g_off_model_free_exact=True,real_bytes_checked=True,
        online_pixels_match_entropy_coding=True))


def main(args):
    run=Run(args);run.thread.start()
    try:
        with exclusive_native_evaluation(run):
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            if args.command=='smoke':
                train(run,'joint_resume_attempt1','joint',6,3)
                train(run,'joint_resume_attempt2','joint',6)
                train(run,'joint_direct','joint',6)
                train(run,'fixed','fixed',6)
                a=torch.load(run.root/'joint_resume/resume.pt',weights_only=True,map_location='cpu')
                b=torch.load(run.root/'joint_direct/resume.pt',weights_only=True,map_location='cpu')
                for k in ('adapter','enhancement','optimizer_g','optimizer_e','step','config'): exact(a[k],b[k])
                audit(run.root,joint='joint_resume',steps=6)
                stream_smoke(run)
                atomic_json(run.root/'smoke_check.json',dict(complete=True,exact_resume=True,
                    training=read(run.root/'training_audit.json'),decode=read(run.root/'streams/decode_check.json'),
                    code={n:file_hash(REPO/'demo'/n) for n in CODE+['online_eg_decode.py','online_eg_stream_smoke.py']}))
            else:
                smoke=DEFAULT.with_name(DEFAULT.name+'_smoke')
                check=read(smoke/'smoke_check.json');assert check['complete'] and check['exact_resume']
                for name,h in check['code'].items(): assert file_hash(REPO/'demo'/name)==h
                protocol=dict(steps=args.steps,arms=['joint','fixed'],smoke=file_hash(smoke/'smoke_check.json'),
                    code={n:file_hash(REPO/'demo'/n) for n in CODE+['online_eg_pipeline.py','run_online_eg.sh']},
                    initial_lora=file_hash(INITIAL),initial_e=file_hash(PATCH),evaluation='next session')
                if (run.root/'queue_protocol.json').exists(): assert read(run.root/'queue_protocol.json')==protocol
                atomic_json(run.root/'queue_protocol.json',protocol)
                for mode in ('joint','fixed'): train(run,mode,mode,args.steps)
                audit(run.root,steps=args.steps)
            run.update(phase='complete')
            atomic_json(run.root/f'{args.command}.complete.json',dict(complete=True,elapsed_seconds=time.monotonic()-run.started))
    except BaseException as e:
        atomic_json(run.root/'last_failure.json',dict(error=repr(e),phase=run.progress));raise
    finally:
        run.log_resources();run.stop.set();run.thread.join(timeout=3)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['smoke','train'])
    p.add_argument('--output',type=Path);p.add_argument('--steps',type=int,default=3000)
    p.add_argument('--max-hours',type=float,default=24.)
    args=p.parse_args()
    if not os.environ.get('TMUX'): p.error('Run inside tmux')
    if args.output is None: args.output=DEFAULT.with_name(DEFAULT.name+'_smoke') if args.command=='smoke' else DEFAULT
    main(args)
