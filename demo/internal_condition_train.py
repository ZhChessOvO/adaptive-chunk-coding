"""Paired input/internal/zero prompts with a frozen RGB LoRA and generator."""
import argparse
from collections import Counter
import copy
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from demo.chunk_enhancement_experiment import read
from demo.feature_condition_pipeline import DEFAULT as PREVIOUS
from demo.feature_condition_cache import selected
from demo.feature_condition_model import conditioned_latent, FORMAT as FEATURE_FORMAT
from demo.feature_condition_train import differentiable_decode, image_terms, restore_step_log
from demo.feature_interface_model import condition_statistics
from demo.internal_condition_model import ConditionBranch, injection, make_bundle, FORMAT
from demo.condition_path_decode import tensor_identity
from demo.scalable_codec import atomic_json, file_hash
from demo.scalable_experiment import check_space, now, resources
from demo.scalable_generation_decode import ASSETS
from demo.stage_c_seedvr2_bridge import configure_runner
from demo.stage_c_seedvr2_lora_finetune import select_training_entry, append_jsonl
from demo.stage_c_seedvr2_lora_utils import load_lora_adapter, lora_state_dict, atomic_torch_save

INITIAL = PREVIOUS/'rgb/adapter.pt'
CACHE = PREVIOUS/'cache.json'
DEFAULT = Path('/root/autodl-fs/DCVC/runs/a800_internal_condition_20260930')


def learning_rate(step, steps):
    warmup = min(100, steps)
    if step <= warmup:
        return 1e-4 * step / warmup
    progress = (step-warmup) / max(1, steps-warmup)
    return 2e-5 + .5 * (1e-4-2e-5) * (1 + math.cos(math.pi*progress))


def crop_for_packets(rng, packets):
    """Select a token-aligned crop containing received P information, without GT."""
    positions = [(x*16, y*16, 256, 256) for y in range(17) for x in range(17)]
    rng.shuffle(positions)
    for crop in positions:
        x, y, w, h = crop
        if any(p['delta'] is not None and p['start'] < 17 and
               max(x,p['roi'][0]) < min(x+w,p['roi'][0]+p['roi'][2]) and
               max(y,p['roi'][1]) < min(y+h,p['roi'][1]+p['roi'][3]) for p in packets):
            return crop
    raise ValueError('no received P feature intersects the training window')


def assert_frozen(runner, initial):
    assert not any(p.requires_grad or p.grad is not None for p in runner.dit.parameters())
    assert not any(p.requires_grad or p.grad is not None for p in runner.vae.parameters())
    for key, tensor in lora_state_dict(runner.dit).items():
        torch.testing.assert_close(tensor, initial['state_dict'][key], rtol=0, atol=0)


def main(args):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    from demo.chunk_enhancement_codec import configure_torch
    configure_torch(); torch.use_deterministic_algorithms(True)
    assert torch.cuda.device_count() == 1
    cache = read(CACHE)
    assert cache['complete'] and len(cache['entries']) == 120
    for row in cache['entries']:
        for path, key in [('path','sha256'),('feature_path','feature_hash'),('pair_path','pair_hash')]:
            assert file_hash(Path(row[path])) == row[key]
    config = dict(version=1, mode=args.mode, steps=args.steps, seed=260930,
        initial_adapter=file_hash(INITIAL), cache_path=str(CACHE), cache_hash=file_hash(CACHE),
        assets={k:file_hash(v) for k,v in ASSETS.items()}, interface_format=FORMAT,
        code={name:file_hash(REPO/'demo'/name) for name in ('internal_condition_train.py','internal_condition_model.py',
            'feature_interface_model.py','feature_condition_model.py','feature_condition_train.py')},
        learning_rate=dict(peak=1e-4, warmup=min(100,args.steps), end=2e-5, schedule='cosine'),
        loss=dict(velocity=.25,clean_latent_l1=.025,latent_temporal=.05,
                  lpips=.5,low_frequency=.1,edge=.05,rgb_temporal=.1),
        uvg_probability=.25, conditions=['partial','full'], crop=256,
        no_E='exact zero branch checked; no meaningless optimizer updates',
        source_roles='90 REDS train + 30 UVG adaptation; no validation training',
        frozen=['UF','E','DiT','LoRA','VAE'], adapter_arithmetic='FP32 branch, BF16 posterior mean; input BF16 add or internal FP32 add/BF16 output',
        vae_temporal_slicing=False, initial_interface='paired internal weights, zero output; input legacy architecture',
        vae_condition='mean BF16; full-frame cached latent crops during training',
        insertion='before block 24 of 32; relative bounded hidden residual',
        noise_dtype='bfloat16')
    args.output.mkdir(parents=True,exist_ok=True)
    config_path = args.output/'config.json'
    if config_path.exists():
        assert read(config_path) == config, 'configuration changed; use a separate run'
    atomic_json(config_path, config)
    stop = [False]
    for sig in (signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    torch.manual_seed(config['seed'])
    runner, device = configure_runner(SimpleNamespace(upstream_root=REPO/'third_party/SeedVR2',
        dit_checkpoint=ASSETS['dit'],vae_checkpoint=ASSETS['vae'],
        positive_embedding=ASSETS['positive'],negative_embedding=ASSETS['negative'],
        lora_checkpoint=None,sample_steps=1,cfg_scale=1.,dit_dtype='bfloat16'))
    runner.dit.to(device=device,dtype=torch.bfloat16)
    load_lora_adapter(runner.dit,INITIAL,trainable=False,strength=1.)
    runner.dit.requires_grad_(False).eval()
    runner.vae.requires_grad_(False).eval()
    runner.vae.set_causal_slicing(split_size=None,memory_device='same')
    # Reset the small-network seed independently of upstream initialization.
    torch.manual_seed(config['seed'])
    kind = 'input' if args.mode == 'input' else 'internal'
    feature = ConditionBranch(kind, 'zero' if args.mode == 'zero' else 'actual').to(device).train()
    parameters = list(feature.parameters())
    parameter_count = sum(p.numel() for p in parameters)
    initial = torch.load(INITIAL,weights_only=True,map_location='cpu')
    assert_frozen(runner,initial)
    import lpips
    metric = lpips.LPIPS(net='alex',verbose=False).to(device).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(parameters,lr=1e-4,weight_decay=1e-4)
    resume_path = args.output/'resume.pt'; completed=0
    if resume_path.exists():
        resume = torch.load(resume_path,weights_only=True,map_location='cpu')
        assert resume['config'] == config
        for k,v in initial['state_dict'].items():
            torch.testing.assert_close(v,resume['adapter']['state_dict'][k],rtol=0,atol=0)
        feature.load_state_dict(resume['adapter']['branch_state'])
        optimizer.load_state_dict(resume['optimizer']); completed=resume['step']
    rows = restore_step_log(args.output/'steps.jsonl',completed)
    from models.dit_v2 import na
    positive = torch.load(ASSETS['positive'],weights_only=True,map_location=device).to(torch.bfloat16)
    text,text_shape = na.flatten([positive])
    torch.cuda.reset_peak_memory_stats(); started=time.monotonic()

    def bundle(step):
        assert_frozen(runner,initial)
        return make_bundle(initial,feature,config,step)

    def save(step):
        atomic_torch_save(resume_path,dict(step=step,config=config,optimizer=optimizer.state_dict(),
                                         adapter=bundle(step)))

    for step in range(completed+1,args.steps+1):
        try: check_space()
        except BaseException:
            save(step-1); raise
        if stop[0]:
            save(step-1); raise InterruptedError('atomic checkpoint saved')
        entry,rng = select_training_entry(cache['entries'],step,config['seed'],.25)
        mode = ('partial','full')[(step-1)%2]
        features = torch.load(entry['feature_path'],weights_only=True,map_location='cpu')
        packets = selected(features['packets'],features['prefix_ids'][mode])
        crop = crop_for_packets(rng,packets); left,top=crop[0]//8,crop[1]//8
        cached = torch.load(entry['path'],weights_only=True,map_location='cpu')
        clean = cached['clean'][:,top:top+32,left:left+32].contiguous().to(device)
        raw = cached[mode][:,top:top+32,left:left+32].contiguous().to(device)
        cond,side,coverage = feature(raw,packets,crop=crop)
        assert coverage.any() and side.requires_grad
        prompt = cond if kind == 'input' else side
        prompt.retain_grad()
        statistics = condition_statistics(raw,cond,side,coverage) if kind == 'input' else dict(
            condition_dtype=str(raw.dtype), rgb_condition_unchanged=torch.equal(raw,cond),
            coverage=float(coverage.mean()), outside_coverage_exact=True,
            side_rms=float(side.detach().square().mean().sqrt()))
        if step == 1 or step % 100 == 0 or step == args.steps:
            with torch.no_grad():
                no_E,empty,_ = feature(raw,[],crop=crop)
                torch.testing.assert_close(no_E,raw,rtol=0,atol=0)
                assert torch.count_nonzero(empty) == 0
        generator = torch.Generator(device=device).manual_seed(config['seed']+step*97409)
        noise = torch.randn(clean.shape,device=device,dtype=torch.bfloat16,generator=generator)
        condition = torch.cat([cond,torch.ones_like(cond[...,:1])],-1)
        flat_noise,shape = na.flatten([noise]); flat_condition,_=na.flatten([condition])
        for group in optimizer.param_groups: group['lr']=learning_rate(step,args.steps)
        optimizer.zero_grad(set_to_none=True); begin=time.monotonic()
        internal_records = []
        with injection(runner.dit,feature,side,coverage,internal_records), torch.autocast('cuda',dtype=torch.bfloat16):
            velocity = runner.dit(vid=torch.cat([flat_noise,flat_condition],-1),txt=text,
                vid_shape=shape,txt_shape=text_shape,timestep=torch.full((1,),1000.,device=device),
                disable_cache=False).vid_sample
            predicted=(flat_noise.float()-velocity.float()).reshape(clean.shape)
            terms=dict(velocity=F.mse_loss(velocity.float(),(noise-clean).reshape_as(velocity).float()),
                clean_latent_l1=F.l1_loss(predicted,clean.float()),
                latent_temporal=F.mse_loss(predicted[1:]-predicted[:-1],clean[1:].float()-clean[:-1].float()))
        with np.load(entry['pair_path']) as pair:
            target=pair['source'][:,top*8:top*8+256,left*8:left*8+256].copy()
        target=torch.from_numpy(target).permute(0,3,1,2).to(device).float()/127.5-1
        image=checkpoint(lambda z:differentiable_decode(runner,z),predicted,use_reentrant=False)
        assert image.shape == target.shape
        terms.update(image_terms(image,target,metric,step))
        image_gradient = None
        if step == 1:
            image_loss = sum(config['loss'][k]*terms[k] for k in
                             ('lpips','low_frequency','edge','rgb_temporal'))
            image_grad = torch.autograd.grad(image_loss,prompt,retain_graph=True)[0]
            image_gradient = float(image_grad.float().norm())
            assert image_gradient > 0 and np.isfinite(image_gradient)
            prompt.grad = None
            del image_loss,image_grad
        loss=sum(config['loss'][k]*v for k,v in terms.items())
        if not torch.isfinite(loss): raise RuntimeError('nonfinite interface loss')
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
        if not torch.isfinite(norm) or norm == 0: raise RuntimeError('invalid interface gradient')
        condition_grad=float(prompt.grad.float().norm())
        assert condition_grad > 0 and np.isfinite(condition_grad)
        assert not any(p.grad is not None for p in runner.dit.parameters())
        assert not any(p.grad is not None for p in runner.vae.parameters())
        optimizer.step(); torch.cuda.synchronize()
        record=dict(step=step,sample=entry['sample_id'],dataset=entry['dataset'],condition=mode,crop=crop,
            loss=float(loss.detach()),terms={k:float(v.detach()) for k,v in terms.items()},
            learning_rate=learning_rate(step,args.steps),gradient_norm=float(norm),
            condition_gradient_norm=condition_grad,condition_statistics=statistics,
            first_step_image_condition_gradient_norm=image_gradient,
            raw_condition_identity=tensor_identity(raw), diffusion_noise_identity=tensor_identity(noise),
            internal_statistics=internal_records,
            seconds=time.monotonic()-begin,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),utc=now())
        append_jsonl(args.output/'steps.jsonl',record); rows.append(record)
        print(json.dumps(record),flush=True)
        if step%25 == 0 or step in (args.steps,args.stop_after):
            save(step)
            atomic_json(args.output/'progress.json',dict(step=step,total=args.steps,mode=args.mode,
                attempt_seconds=time.monotonic()-started,resources=resources()))
        if step%1000 == 0 or step == args.steps:
            atomic_torch_save(args.output/'checkpoints'/f'adapter_{step:06d}.pt',bundle(step))
        del loss,terms,velocity,predicted,clean,cond,condition,flat_condition,noise,cached,image,target,side
        if step == args.stop_after and step < args.steps:
            torch.distributed.destroy_process_group(); return
    final=args.output/'adapter.pt'; atomic_torch_save(final,bundle(args.steps))
    atomic_json(args.output/'complete.json',dict(steps=args.steps,adapter_sha256=file_hash(final),
        trainable_parameters=parameter_count,frozen_lora_exact=True,
        dataset_steps=dict(Counter(r['dataset'] for r in rows)),
        condition_steps=dict(Counter(r['condition'] for r in rows)),
        total_step_seconds=sum(r['seconds'] for r in rows),
        peak_cuda_allocated_bytes=max(r['peak_cuda_allocated_bytes'] for r in rows),resources=resources()))
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['input','internal','zero'],required=True)
    p.add_argument('--steps',type=int,default=3000); p.add_argument('--stop-after',type=int,default=-1)
    args=p.parse_args()
    if args.steps < 1: p.error('positive steps required')
    main(args)

