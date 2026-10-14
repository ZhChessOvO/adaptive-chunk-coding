"""Isolated two-factor G-loss ablation; historical trainer stays immutable."""
import argparse
from collections import Counter
import copy
import json
import math
import os
from pathlib import Path
import random
import signal
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from demo.chunk_enhancement_codec import configure_torch
from demo.condition_path_decode import tensor_identity
from demo.conditioned_generation_train import differentiable_decode, restore_step_log
from demo.roi_condition_data import encode_roi, core_image_terms
from demo.roi_condition_train import assert_frozen
from demo.routervc_fullview_probe import read, immutable, digest
from demo.scalable_codec import atomic_json
from demo.scalable_experiment import check_space, now
from demo.stage_c_seedvr2_bridge import configure_runner
from demo.stage_c_seedvr2_lora_finetune import append_jsonl
from demo.stage_c_seedvr2_lora_utils import (load_lora_adapter, trainable_lora_parameters,
    load_lora_state_dict, lora_state_dict, atomic_torch_save)
from routervc.generation.texture_data import REPO, choose, preload
from routervc.generation.texture_losses import paired_lpips, phase_bias_loss


ARMS = ('control', 'repair', 'augmentation_only', 'phase_only')


def switches(arm):
    if arm not in ARMS:
        raise ValueError('unknown loss arm: '+arm)
    return arm in ('repair', 'augmentation_only'), arm in ('repair', 'phase_only')


def loss_terms(terms, image, target, metric, step, config, arm):
    """Same operation order as the historical trainer; only two switches differ."""
    augmentation, periodic = switches(arm)
    original_lpips = float(terms['lpips'].detach())
    transform = None
    if augmentation:
        terms['lpips'], transform = paired_lpips(image, target, metric, step, config['seed']+step*7919)
    phase = phase_bias_loss(image[...,64:-64,64:-64], target[...,64:-64,64:-64])
    loss = sum(config['loss'][k]*v for k,v in terms.items())
    if periodic:
        loss = loss+config['phase_weight']*phase
    return loss, phase, transform, original_lpips


def learning_rate(step, config):
    spec = config['learning_rate']; warm = spec['warmup']
    if step <= warm:
        return spec['peak']*step/warm
    t = (step-warm)/max(1, config['steps']-warm)
    return spec['end']+.5*(spec['peak']-spec['end'])*(1+math.cos(math.pi*t))


def train(args):
    configure_torch(); torch.use_deterministic_algorithms(True)
    if torch.cuda.device_count() != 1 or not os.environ.get('TMUX'):
        raise RuntimeError('one visible GPU inside tmux required')
    config = read(args.protocol)
    for name, expected in config['code'].items():
        if digest(REPO/name) != expected: raise ValueError('training source changed: '+name)
    for asset in config['assets'].values():
        if digest(asset['path']) != asset['sha256']: raise ValueError('model asset changed')
    if digest(config['initial']) != config['initial_sha256']: raise ValueError('initial LoRA changed')
    args.output.mkdir(parents=True, exist_ok=True)
    binding = dict(protocol=digest(args.protocol), arm=args.arm)
    immutable(args.output/'binding.json', binding)
    if (args.output/'complete.json').exists():
        done = read(args.output/'complete.json')
        if done['binding'] != binding or digest(args.output/'adapter.pt') != done['adapter_sha256']:
            raise ValueError('completed arm changed')
        print('ALREADY_COMPLETE', args.arm, flush=True); return
    stop = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_:stop.__setitem__(0, True))

    def progress(**kw):
        atomic_json(args.output/'progress.json', dict(arm=args.arm, utc=now(), **kw))
        print(json.dumps(dict(arm=args.arm, **kw)), flush=True)
        if stop[0]: raise InterruptedError('stopped at preparation boundary')

    samples = preload(config['rows'], progress)
    torch.manual_seed(config['seed'])
    assets = {k:v['path'] for k,v in config['assets'].items()}
    runner, device = configure_runner(SimpleNamespace(upstream_root=REPO/'third_party/SeedVR2',
        dit_checkpoint=Path(assets['dit']), vae_checkpoint=Path(assets['vae']),
        positive_embedding=Path(assets['positive']), negative_embedding=Path(assets['negative']),
        lora_checkpoint=None, sample_steps=1, cfg_scale=1., dit_dtype='bfloat16'))
    runner.dit.to(device=device, dtype=torch.bfloat16)
    load_lora_adapter(runner.dit, config['initial'], trainable=True, strength=1.)
    runner.dit.train(); runner.vae.requires_grad_(False).eval()
    runner.vae.set_causal_slicing(split_size=None, memory_device='same')
    initial = torch.load(config['initial'], weights_only=True, map_location='cpu')
    parameters = trainable_lora_parameters(runner.dit)
    assert all(p.dtype == torch.float32 for p in parameters)
    assert_frozen(runner, initial)
    optimizer = torch.optim.AdamW(parameters, lr=1e-5, weight_decay=1e-4)
    import lpips
    metric = lpips.LPIPS(net='alex', verbose=False).to(device).eval().requires_grad_(False)
    from models.dit_v2 import na
    positive = torch.load(assets['positive'], weights_only=True, map_location=device).to(torch.bfloat16)
    text, text_shape = na.flatten([positive])
    random.seed(config['seed'])
    completed = 0
    if (args.output/'resume.pt').exists():
        saved = torch.load(args.output/'resume.pt', weights_only=True, map_location='cpu')
        if saved['binding'] != binding: raise ValueError('resume binding differs')
        load_lora_state_dict(runner.dit, saved['adapter']['state_dict'])
        optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['rng_cpu']); torch.cuda.set_rng_state_all(saved['rng_cuda'])
        random.setstate(saved['rng_python'])
        completed = saved['step']
    records = restore_step_log(args.output/'steps.jsonl', completed)
    torch.cuda.reset_peak_memory_stats()

    def bundle(step):
        result = copy.deepcopy(initial)
        result['state_dict'] = lora_state_dict(runner.dit)
        result.setdefault('metadata', {}).update(texture_loss_ablation=binding, step=step, frozen_lora=False)
        return result

    def save(step, milestone=False):
        assert_frozen(runner, initial)
        adapter = bundle(step)
        atomic_torch_save(args.output/'resume.pt', dict(binding=binding, step=step, adapter=adapter,
            optimizer=optimizer.state_dict(), rng_cpu=torch.get_rng_state(),
            rng_cuda=torch.cuda.get_rng_state_all(), rng_python=random.getstate()))
        if milestone:
            atomic_torch_save(args.output/'checkpoints'/f'adapter_{step:06d}.pt', adapter)

    try:
        for step in range(completed+1, config['steps']+1):
            check_space()
            if stop[0]: raise InterruptedError('checkpoint stop requested')
            began = time.monotonic()
            row, crop = choose(config['rows'], step, config['seed'])
            original, received = samples[row['sample_id']]
            x,y,w,h = crop
            target = torch.from_numpy(original[:,y:y+h,x:x+w].copy()).permute(0,3,1,2).to(device).float()/127.5-1
            # Always RGB-first native encoding; reset causal VAE history each time.
            clean = encode_roi(runner, original, crop, device).to(device)
            cond = encode_roi(runner, received, crop, device).to(device)
            torch.manual_seed(config['seed']+step*1000033)
            generator = torch.Generator(device=device).manual_seed(config['seed']+step*97409)
            noise = torch.randn(clean.shape, device=device, dtype=torch.bfloat16, generator=generator)
            flat_noise, shape = na.flatten([noise])
            flat_condition,_ = na.flatten([torch.cat([cond, torch.ones_like(cond[...,:1])], -1)])
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups: group['lr'] = learning_rate(step, config)
            def forward(condition):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    return runner.dit(vid=torch.cat([flat_noise,condition],-1), txt=text,
                        vid_shape=shape, txt_shape=text_shape,
                        timestep=torch.full((1,),1000.,device=device), disable_cache=False).vid_sample
            velocity = checkpoint(forward, flat_condition, use_reentrant=False)
            predicted = (flat_noise.float()-velocity.float()).reshape(clean.shape)
            terms = dict(velocity=F.mse_loss(velocity.float(),(noise-clean).reshape_as(velocity).float()),
                clean_latent_l1=F.l1_loss(predicted,clean.float()),
                latent_temporal=F.mse_loss(predicted[1:]-predicted[:-1],clean[1:].float()-clean[:-1].float()))
            image = checkpoint(lambda z:differentiable_decode(runner,z), predicted, use_reentrant=False)
            terms.update(core_image_terms(image,target,metric,step))
            loss, phase, transform, original_lpips = loss_terms(
                terms, image, target, metric, step, config, args.arm)
            if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
            path_checks = None
            if step == 1:
                path_checks = {}
                for name, value in [('perceptual',terms['lpips']),('periodic',phase)]:
                    gradient = torch.autograd.grad(value, parameters[-1], retain_graph=True)[0]
                    path_checks[name] = float(gradient.float().norm())
                if not all(np.isfinite(v) and v > 0 for v in path_checks.values()):
                    raise RuntimeError('loss has no finite LoRA gradient')
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True))
            if grad_norm <= 0: raise RuntimeError('zero LoRA gradient')
            assert_frozen(runner, initial)
            optimizer.step(); torch.cuda.synchronize()
            record = dict(step=step, arm=args.arm, sample=row['sample_id'], dataset=row['dataset'],
                E_cap=row['E_cap'], crop=crop, loss=float(loss.detach()),
                terms={k:float(v.detach()) for k,v in terms.items()},
                unaugmented_lpips=original_lpips, phase_error=float(phase.detach()), transform=transform,
                gradient_norm=grad_norm, first_loss_gradients=path_checks,
                learning_rate=learning_rate(step,config), condition=tensor_identity(cond), noise=tensor_identity(noise),
                seconds=time.monotonic()-began, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), utc=now())
            append_jsonl(args.output/'steps.jsonl', record); records.append(record); completed=step
            print(json.dumps(record),flush=True)
            if step % config['checkpoint_every'] == 0 or step in (config['steps'],args.stop_after):
                save(step, milestone=step % config['milestone_every'] == 0 or step==config['steps'])
                progress(phase='training', step=step, total=config['steps'])
            del loss,terms,phase,image,predicted,velocity,flat_condition,noise,cond,clean,target
            if step == args.stop_after and step < config['steps']: return
        atomic_torch_save(args.output/'adapter.pt', bundle(completed))
        atomic_json(args.output/'complete.json', dict(complete=True, binding=binding, steps=completed,
            adapter_sha256=digest(args.output/'adapter.pt'),
            dataset_steps=dict(Counter(r['dataset'] for r in records)),
            seconds=sum(r['seconds'] for r in records),
            peak_cuda_allocated_bytes=max(r['peak_cuda_allocated_bytes'] for r in records),
            trained_parameters=sum(p.numel() for p in parameters), production_model_changed=False))
        progress(phase='complete',step=completed,total=config['steps'])
    except BaseException:
        save(completed)
        raise
    finally:
        if torch.distributed.is_initialized(): torch.distributed.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--stop-after',type=int,default=0)
    train(p.parse_args())

