"""Paired RGB-only and decoded-feature-conditioned continuation.

Both arms start from the completed image-objective adapter and share every
sample, crop, noise and image loss. Original codec, E, DiT and VAE stay frozen.
"""
import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import random
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
from demo.scalable_codec import atomic_json, file_hash
from demo.scalable_generation_decode import ASSETS
from demo.stage_c_seedvr2_bridge import configure_runner
from demo.stage_c_seedvr2_lora_utils import (load_lora_adapter, lora_modules,
    trainable_lora_parameters, load_lora_state_dict, adapter_payload,
    atomic_torch_save, save_lora_adapter)
from demo.stage_c_seedvr2_lora_finetune import (select_training_entry,
    append_jsonl)
from demo.scalable_experiment import check_space, resources, now
from demo.feature_condition_cache import PREVIOUS, selected
from demo.feature_condition_model import FeatureCondition, conditioned_latent, FORMAT
INITIAL = PREVIOUS/"image/adapter.pt"


def restore_step_log(path, completed):
    """Power-loss recovery: ignore only an incomplete final JSON write.

    Checkpointed steps must still be present; corruption in an earlier or
    newline-terminated record is an error, not silently discarded evidence.
    """
    lines = path.read_text().splitlines(keepends=True) if path.exists() else []
    records = {}
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines)-1 and not line.endswith('\n'):
                break
            raise
        if int(record['step']) <= completed:
            if record['step'] in records:
                raise RuntimeError('duplicate checkpointed step record')
            records[record['step']] = record
    if sorted(records) != list(range(1,completed+1)):
        raise RuntimeError('missing checkpointed step record')
    ordered = [records[i] for i in range(1,completed+1)]
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix('.jsonl.tmp')
    temporary.write_text(''.join(json.dumps(r)+'\n' for r in ordered))
    os.replace(temporary,path)
    return ordered


def differentiable_decode(runner, latent):
    # Upstream runner.vae_decode is @no_grad; call its frozen VAE directly.
    # Disable temporal slicing: its history cache detaches tensors and has
    # side effects incompatible with checkpoint recomputation.
    value = latent.to(torch.bfloat16).permute(3, 0, 1, 2).unsqueeze(0)
    value = value / float(runner.config.vae.scaling_factor)
    value = value + float(runner.config.vae.get('shifting_factor', 0.))
    output = runner.vae.decode(value).sample
    if hasattr(runner.vae, 'postprocess'):
        output = runner.vae.postprocess(output)
    return output[0].permute(1, 0, 2, 3).float()


def image_terms(pred, target, metric, offset=0):
    # Omit 16 pixels around latent-crop boundaries. These are training-only
    # crops; final evaluation uses the actual receiver's context + feather.
    pred, target = pred[..., 16:-16, 16:-16], target[..., 16:-16, 16:-16]
    indices = list(range(offset % 4, pred.shape[0], 4))
    perceptual = metric(pred[indices].clamp(-1, 1), target[indices]).mean()
    low = F.l1_loss(F.avg_pool2d(pred, 8), F.avg_pool2d(target, 8))
    edge = (F.l1_loss(pred[..., 1:, :]-pred[..., :-1, :],
                      target[..., 1:, :]-target[..., :-1, :]) +
            F.l1_loss(pred[..., :, 1:]-pred[..., :, :-1],
                      target[..., :, 1:]-target[..., :, :-1])) / 2
    temporal = F.l1_loss(pred[1:]-pred[:-1], target[1:]-target[:-1])
    return dict(lpips=perceptual, low_frequency=low, edge=edge, rgb_temporal=temporal)


def main(args):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    from demo.chunk_enhancement_codec import configure_torch
    configure_torch()
    torch.use_deterministic_algorithms(True)
    if torch.cuda.device_count() != 1:
        raise RuntimeError('exactly one visible GPU required')
    cache = json.loads(args.cache.read_text())
    if not cache['complete'] or set(cache['modes']) != {'none', 'partial', 'full'}:
        raise RuntimeError('incomplete condition cache')
    for row in cache['entries']:
        if file_hash(Path(row['path'])) != row['sha256']:
            raise RuntimeError('latent input changed')
        if file_hash(Path(row['feature_path'])) != row['feature_hash']:
            raise RuntimeError('decoded feature cache changed')
        if file_hash(Path(row['pair_path'])) != row['pair_hash']:
            raise RuntimeError('RGB training target changed')
    config = dict(version=1, mode=args.mode, max_steps=args.steps, seed=260929,
        crop_size=256, learning_rate=2e-5, rank=8, alpha=8., last_n_blocks=8,
        initial_adapter=file_hash(INITIAL), initial_strength=1.,
        feature_format=FORMAT, feature_code=file_hash(REPO/'demo/feature_condition_model.py'),
        feature_cache_code=file_hash(REPO/'demo/feature_condition_cache.py'),
        uvg_probability=.25, condition_probabilities=[1/3]*3,
        cache_path=str(args.cache.resolve()), cache_hash=file_hash(args.cache),
        assets={k:file_hash(v) for k,v in ASSETS.items()},
        code=file_hash(Path(__file__)), source_roles='REDS train + UVG adaptation, no val',
        loss=dict(velocity=.25, clean_latent_l1=.025, latent_temporal=.05,
                  lpips=.5,
                  low_frequency=.1,
                  edge=.05,
                  rgb_temporal=.1),
        vae_temporal_slicing=False, latent_crop_border_ignored=16,
        deterministic_latent_posterior='mode', per_step_noise=True)
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output/'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError('training config changed; use a distinct experiment')
    atomic_json(config_path, config)
    stop = [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    torch.manual_seed(config['seed'])
    bridge = SimpleNamespace(upstream_root=REPO/'third_party/SeedVR2',
        dit_checkpoint=ASSETS['dit'], vae_checkpoint=ASSETS['vae'],
        positive_embedding=ASSETS['positive'], negative_embedding=ASSETS['negative'],
        lora_checkpoint=None, sample_steps=1, cfg_scale=1., dit_dtype='bfloat16')
    runner, device = configure_runner(bridge)
    runner.dit.to(device=device, dtype=torch.bfloat16)
    info = load_lora_adapter(runner.dit, INITIAL, trainable=True, strength=1.)
    parameters = trainable_lora_parameters(runner.dit)
    if sum(p.numel() for p in parameters) != 2822656:
        raise RuntimeError('unexpected trainable parameter set')
    feature = FeatureCondition().to(device)
    if args.mode == 'feature':
        parameters += list(feature.parameters())
    else:
        feature.requires_grad_(False)
    runner.vae.requires_grad_(False).eval()
    runner.vae.set_causal_slicing(split_size=None, memory_device='same')
    metric = None
    if True:
        import lpips
        metric = lpips.LPIPS(net='alex', verbose=False).to(device).eval().requires_grad_(False)
    else:
        runner.vae = None
        gc.collect(); torch.cuda.empty_cache()
    optimizer = torch.optim.AdamW(parameters, lr=config['learning_rate'], weight_decay=1e-4)
    completed = 0
    resume_path = args.output/'resume.pt'
    if resume_path.exists():
        resume = torch.load(resume_path, weights_only=True, map_location='cpu')
        if resume['config'] != config:
            raise RuntimeError('resume configuration mismatch')
        load_lora_state_dict(runner.dit, resume['adapter']['state_dict'])
        feature.load_state_dict(resume['adapter']['feature_state'])
        optimizer.load_state_dict(resume['optimizer'])
        completed = resume['step']
    rows = restore_step_log(args.output/'steps.jsonl', completed)
    if len(rows) != completed:
        raise RuntimeError('step log and checkpoint disagree')
    from models.dit_v2 import na
    positive = torch.load(ASSETS['positive'], weights_only=True, map_location=device).to(torch.bfloat16)
    text, text_shape = na.flatten([positive])
    runner.dit.train()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()

    def bundle(step):
        payload = adapter_payload(runner.dit, rank=8, alpha=8., last_n_blocks=8,
            metadata=dict(step=step, inference_strength=1.,
                          base_dit_sha256=config['assets']['dit'], config=config))
        payload.update(feature_format=FORMAT, feature_enabled=args.mode == 'feature',
            feature_state={k:v.detach().cpu() for k,v in feature.state_dict().items()})
        return payload

    def save(step):
        atomic_torch_save(resume_path, dict(step=step, config=config,
            optimizer=optimizer.state_dict(), adapter=bundle(step)))

    for step in range(completed+1, args.steps+1):
        try:
            check_space()
        except BaseException:
            save(step-1)
            raise
        if stop[0]:
            save(step-1)
            raise InterruptedError('checkpoint saved; resume same command')
        entry, rng = select_training_entry(cache['entries'], step, config['seed'], .25)
        mode = ('none', 'partial', 'full')[(step-1) % 3]
        cached = torch.load(entry['path'], weights_only=True, map_location='cpu')
        # Token-aligned crop; no latent flipping masquerading as exact RGB flipping.
        top, left = rng.randrange(17)*2, rng.randrange(17)*2
        clean = cached['clean'][:, top:top+32, left:left+32].contiguous().to(device)
        cond = cached[mode][:, top:top+32, left:left+32].contiguous().to(device)
        feature_cache = torch.load(entry['feature_path'], weights_only=True, map_location='cpu')
        packets = selected(feature_cache['packets'], feature_cache['prefix_ids'][mode])
        raw_cond = cond
        with torch.autocast('cuda', dtype=torch.bfloat16):
            if args.mode == 'feature':
                cond, side, coverage = conditioned_latent(feature, cond, packets,
                    crop=(left*8, top*8, 256, 256))
            else:
                side, coverage = torch.zeros_like(cond), torch.zeros_like(cond[..., :1])
        if mode == 'none' and torch.count_nonzero(side):
            raise RuntimeError('absent packet changed the feature condition')
        gen = torch.Generator(device=device).manual_seed(config['seed']+step*97409)
        noise = torch.randn(clean.shape, device=device, dtype=torch.bfloat16, generator=gen)
        condition = torch.cat([cond, torch.ones_like(cond[..., :1])], -1)
        flat_noise, shape = na.flatten([noise])
        flat_condition, _ = na.flatten([condition])
        optimizer.zero_grad(set_to_none=True)
        begin = time.monotonic()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            velocity = runner.dit(vid=torch.cat([flat_noise, flat_condition], -1),
                txt=text, vid_shape=shape, txt_shape=text_shape,
                timestep=torch.full((1,), 1000., device=device), disable_cache=False).vid_sample
            predicted = (flat_noise.float()-velocity.float()).reshape(clean.shape)
            terms = dict(velocity=F.mse_loss(velocity.float(), (noise-clean).reshape_as(velocity).float()),
                clean_latent_l1=F.l1_loss(predicted, clean.float()),
                latent_temporal=F.mse_loss(predicted[1:]-predicted[:-1], clean[1:].float()-clean[:-1].float()))
        rgb_grad = None
        if metric is not None:
            with np.load(entry['pair_path']) as pair:
                target = pair['source'][:, top*8:top*8+256, left*8:left*8+256].copy()
            target = torch.from_numpy(target).permute(0,3,1,2).to(device).float()/127.5-1
            # Non-reentrant checkpoint supports explicit autograd.grad validation.
            image = checkpoint(lambda z:differentiable_decode(runner,z), predicted,
                               use_reentrant=False)
            if image.shape != target.shape:
                raise RuntimeError(f'VAE temporal/shape mismatch: {image.shape} {target.shape}')
            terms.update(image_terms(image, target, metric, step))
            if step == 1:
                rgb_loss = sum(config['loss'][k]*terms[k] for k in
                               ('lpips','low_frequency','edge','rgb_temporal'))
                grad = torch.autograd.grad(rgb_loss, predicted, retain_graph=True)[0]
                rgb_grad = float(grad.norm())
                if not torch.isfinite(grad).all() or rgb_grad == 0:
                    raise RuntimeError('image loss does not backpropagate through VAE')
                with torch.no_grad():
                    native = runner.vae_decode([predicted.detach()])[0].permute(1,0,2,3).float()
                torch.testing.assert_close(native, image.detach(), rtol=0, atol=0)
                del native, grad, rgb_loss
        loss = sum(config['loss'][k]*v for k,v in terms.items())
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
        if not torch.isfinite(norm) or norm == 0:
            raise RuntimeError('invalid LoRA gradient')
        if any(p.grad is not None for p in runner.dit.parameters() if not p.requires_grad):
            raise RuntimeError('base DiT accumulated gradients')
        if runner.vae is not None and any(p.grad is not None for p in runner.vae.parameters()):
            raise RuntimeError('VAE accumulated gradients')
        feature_grad = float(torch.stack([p.grad.float().norm() for p in feature.parameters()
            if p.grad is not None]).norm()) if args.mode == 'feature' else 0.
        if args.mode == 'feature' and coverage.any() and feature_grad == 0:
            raise RuntimeError('received feature condition has no training gradient')
        optimizer.step()
        torch.cuda.synchronize()
        record = dict(step=step, sample=entry['sample_id'], dataset=entry['dataset'], condition=mode,
            crop=[left*8,top*8,256,256], loss=float(loss.detach()),
            terms={k:float(v.detach()) for k,v in terms.items()}, gradient_norm=float(norm),
            feature_gradient_norm=feature_grad, feature_side_rms=float(side.float().square().mean().sqrt().detach()),
            feature_coverage=float(coverage.mean()),
            image_to_latent_gradient=rgb_grad, seconds=time.monotonic()-begin,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), utc=now())
        append_jsonl(args.output/'steps.jsonl',record); rows.append(record)
        print(json.dumps(record), flush=True)
        if step % 25 == 0 or step == args.steps or step == args.stop_after:
            save(step)
            atomic_json(args.output/'progress.json', dict(step=step, total=args.steps,
                mode=args.mode, attempt_seconds=time.monotonic()-started, resources=resources()))
        if step in {250, 500, 1000, args.steps}:
            atomic_torch_save(args.output/'checkpoints'/f'adapter_{step:06d}.pt', bundle(step))
        del loss, terms, velocity, predicted, clean, cond, condition, noise, cached
        if metric is not None:
            del image, target
        if step == args.stop_after and step < args.steps:
            torch.distributed.destroy_process_group()
            return
    final = args.output/'adapter.pt'
    atomic_torch_save(final, bundle(args.steps))
    atomic_json(args.output/'complete.json', dict(steps=args.steps, adapter_sha256=file_hash(final),
        trainable_parameters=sum(p.numel() for p in parameters),
        dataset_steps=dict(Counter(r['dataset'] for r in rows)),
        condition_steps=dict(Counter(r['condition'] for r in rows)),
        mean_step_seconds=float(np.mean([r['seconds'] for r in rows])),
        peak_cuda_allocated_bytes=max(r['peak_cuda_allocated_bytes'] for r in rows),
        total_step_seconds=sum(r['seconds'] for r in rows), resources=resources()))
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['rgb','feature'], required=True)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--stop-after', type=int, default=-1)
    a = p.parse_args()
    if a.steps < 1:
        p.error('positive steps required')
    main(a)
