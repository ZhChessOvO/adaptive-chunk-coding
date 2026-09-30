"""Resumable three-arm LoRA coadaptation queue, reusing the pinned receiver."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from demo.chunk_enhancement_experiment import Run, read
from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
from demo.conditioned_generation_pipeline import execute
from demo.joint_condition_train import DEFAULT, INITIAL, CACHE
from demo.internal_condition_pipeline import variant, assert_resume, decode_point, assert_noise, OLD
from demo.internal_condition_model import validate_bundle
from demo.patch_prefix_probe import load_frames
from demo.scalable_codec import atomic_json, file_hash

ARMS = ('internal', 'zero', 'rgb')


def audit(root, names=None, steps=3000):
    names = names or dict(zip(ARMS, ARMS))
    logs, configs, result = {}, {}, {}
    initial = torch.load(INITIAL, weights_only=True, map_location='cpu')
    for mode, name in names.items():
        directory = root / name
        cfg, info = read(directory/'config.json'), read(directory/'complete.json')
        bundle = torch.load(directory/'adapter.pt', weights_only=True, map_location='cpu')
        resume = torch.load(directory/'resume.pt', weights_only=True, map_location='cpu')
        milestone = torch.load(directory/'checkpoints'/f'adapter_{steps:06d}.pt', weights_only=True, map_location='cpu')
        assert info['steps'] == resume['step'] == steps
        assert cfg == resume['config'] == bundle['metadata']['config']
        assert cfg['initial_adapter'] == file_hash(INITIAL) and cfg['cache_hash'] == file_hash(CACHE)
        for f, h in cfg['code'].items(): assert file_hash(REPO/'demo'/f) == h
        assert info['adapter_sha256'] == file_hash(directory/'adapter.pt')
        assert validate_bundle(bundle) == ('internal', {'rgb':'off','internal':'actual','zero':'zero'}[mode])
        assert bundle['metadata']['frozen_lora'] is False and info['base_weights_frozen']
        assert info['lora_changed'] and any(not torch.equal(v, initial['state_dict'][k])
            for k, v in bundle['state_dict'].items())
        for section in ('state_dict','branch_state'):
            for k, value in bundle[section].items():
                assert torch.isfinite(value).all()
                torch.testing.assert_close(value, resume['adapter'][section][k], rtol=0, atol=0)
                torch.testing.assert_close(value, milestone[section][k], rtol=0, atol=0)
        rows = [json.loads(line) for line in (directory/'steps.jsonl').read_text().splitlines()]
        assert [r['step'] for r in rows] == list(range(1,steps+1))
        assert rows[0]['first_step_image_latent_gradient_norm'] > 0
        for r in rows:
            assert np.isfinite(r['loss']) and np.isfinite(r['lora_gradient_norm']) and r['lora_gradient_norm'] > 0
            if mode != 'rgb' and r['condition'] != 'none':
                assert np.isfinite(r['feature_gradient_norm']) and r['feature_gradient_norm'] > 0
            else: assert r['feature_gradient_norm'] == 0
        assert info['dataset_steps'] == dict(Counter(r['dataset'] for r in rows))
        assert info['condition_steps'] == dict(Counter(r['condition'] for r in rows))
        logs[mode], configs[mode], result[mode] = rows, cfg, info
    for mode in ('zero','rgb'):
        assert {k:v for k,v in configs[mode].items() if k != 'mode'} == {
            k:v for k,v in configs['internal'].items() if k != 'mode'}
        for a,b in zip(logs['internal'], logs[mode], strict=True):
            for k in ('step','sample','dataset','condition','crop','learning_rate',
                      'lora_learning_rate','raw_condition_identity','diffusion_noise_identity'):
                assert a[k] == b[k], f'Nonpaired {mode}: {k}'
    report = dict(complete=True, steps=steps, paired_samples_conditions_noise_exact=True,
                  lora_updated_all_arms=True, base_weights_frozen=True, training=result)
    atomic_json(root/'training_audit.json', report)
    return report


def train(run, name, mode, steps, stop=-1):
    dest = run.root / name.split('_attempt')[0]
    if (dest/'complete.json').exists():
        info, cfg = read(dest/'complete.json'), read(dest/'config.json')
        assert info['steps'] == steps and cfg['mode'] == mode
        assert info['adapter_sha256'] == file_hash(dest/'adapter.pt')
        for f,h in cfg['code'].items(): assert file_hash(REPO/'demo'/f) == h
        return
    execute(run, name, 'joint_condition_train.py',
        ['--output',dest,'--mode',mode,'--steps',steps,'--stop-after',stop], distributed=True)


def smoke(run):
    train(run,'internal_resume_attempt1','internal',6,3)
    train(run,'internal_resume_attempt2','internal',6)
    train(run,'internal_direct','internal',6)
    train(run,'zero','zero',6)
    train(run,'rgb','rgb',6)
    a = torch.load(run.root/'internal_resume/resume.pt', weights_only=True, map_location='cpu')
    b = torch.load(run.root/'internal_direct/resume.pt', weights_only=True, map_location='cpu')
    assert_resume(a,b)
    audit(run.root, dict(internal='internal_resume',zero='zero',rgb='rgb'),6)
    adapters = {k:run.root/( 'internal_resume' if k == 'internal' else k)/'adapter.pt' for k in ARMS}
    adapters['off'] = variant(adapters['internal'],run.root/'same_lora_off.pt','off')
    row = read(OLD/'summary.json')['results'][0]
    results, paths = {}, {}
    for name,mode in [('internal','full'),('zero','full'),('rgb','full'),
                      ('internal','none'),('off','none'),('G_off','full')]:
        key = 'internal' if name == 'G_off' else name
        dest, d = decode_point(run.root/'decode_smoke',row,name,mode,adapters[key],run,name=='G_off')
        results[name,mode], paths[name,mode] = d,dest
        if name != 'G_off': assert_noise(results['internal',mode],d)
    np.testing.assert_array_equal(load_frames(paths['internal','none']/'reconstruction.npz'),
                                  load_frames(paths['off','none']/'reconstruction.npz'))
    cfg = read(run.root/'internal_resume/config.json')
    atomic_json(run.root/'smoke_check.json',dict(complete=True, resume_lora_branch_optimizer_exact=True,
        paired_training=True, no_E_same_LoRA_exact=True, G_off_without_weights_exact=True,
        fresh_reloads=6, code=cfg['code'], adapters={k:file_hash(v) for k,v in adapters.items()}))


def pin(root, smoke_root, steps):
    check = read(smoke_root/'smoke_check.json')
    assert check['complete'] and check['resume_lora_branch_optimizer_exact']
    for f,h in check['code'].items(): assert file_hash(REPO/'demo'/f) == h
    files = ['joint_condition_train.py','joint_condition_pipeline.py','run_joint_condition.sh',
             'internal_condition_model.py','internal_condition_decode.py','internal_condition_pipeline.py']
    protocol = dict(steps=steps, arms=list(ARMS), initial=file_hash(INITIAL), cache=file_hash(CACHE),
        code={f:file_hash(REPO/'demo'/f) for f in files}, smoke=file_hash(smoke_root/'smoke_check.json'),
        evaluation='next session; training queue does not launch new evaluation recipes')
    path = root/'queue_protocol.json'
    if path.exists(): assert read(path) == protocol, 'Pinned queue changed'
    else: atomic_json(path,protocol)


def main(args):
    run = Run(args); run.thread.start()
    try:
        with exclusive_native_evaluation(run):
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            if args.command == 'smoke': smoke(run)
            else:
                pin(run.root, DEFAULT.with_name(DEFAULT.name+'_smoke'), args.steps)
                for mode in ARMS: train(run,mode,mode,args.steps)
                audit(run.root,steps=args.steps)
            run.update(phase='complete')
            atomic_json(run.root/f'{args.command}.complete.json',dict(complete=True,
                elapsed_seconds=time.monotonic()-run.started, code=file_hash(Path(__file__))))
    except BaseException as e:
        atomic_json(run.root/'last_failure.json',dict(error=repr(e),phase=run.progress))
        raise
    finally:
        run.log_resources(); run.stop.set(); run.thread.join(timeout=3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['smoke','train'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--max-hours', type=float, default=24.)
    args = parser.parse_args()
    if not os.environ.get('TMUX'): parser.error('Run inside tmux')
    if args.output is None:
        args.output = DEFAULT.with_name(DEFAULT.name+'_smoke') if args.command == 'smoke' else DEFAULT
    main(args)
