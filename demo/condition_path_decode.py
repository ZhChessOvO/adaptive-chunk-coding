"""Versioned fixed-weight VAE sample/mode receiver with paired-noise evidence.

The hashed model bundle selects the condition, not hidden decoder CLI state.
Both paths still call upstream VAE encode once (which consumes a posterior draw).
No cast, feature injection, diffusion sampler, or old receiver code is changed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from demo import feature_interface_decode as parent
from demo.feature_interface_model import validate_bundle
from demo.scalable_codec import file_hash

FORMAT = 'vae_condition_sample_or_mean_preserve_draw_dtype_v2'
PARENT_IDENTITIES = parent.identities
PARENT_RESTORE = parent.restore
UPSTREAM = ('projects/video_diffusion_sr/infer.py',
            'models/video_vae_v3/modules/attn_video_vae.py')


def validate_condition(bundle):
    validate_bundle(bundle)
    mode = bundle.get('vae_condition')
    if bundle.get('condition_path_format') != FORMAT or mode not in ('sample', 'mean'):
        raise ValueError('missing or invalid versioned VAE condition configuration')
    return mode


def tensor_identity(value):
    value = value.detach().contiguous().cpu()
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest())


def rng_identity():
    return dict(cpu=tensor_identity(torch.get_rng_state()),
                cuda=[tensor_identity(v) for v in torch.cuda.get_rng_state_all()])


def identities(adapter):
    mode = validate_condition(torch.load(adapter, weights_only=True, map_location='cpu'))
    result = PARENT_IDENTITIES(adapter)
    spec = dict(parent=result['profile'], receiver=file_hash(Path(__file__)),
                format=FORMAT, condition=mode,
                upstream={p:file_hash(REPO/'third_party/SeedVR2'/p) for p in UPSTREAM})
    result['profile'] = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    return result


def posterior_condition(result, mode):
    """Replace only values, not sample dtype/layout or random-number consumption.

    Upstream mode() can be BF16 while sample() promotes to FP32. Switching its
    use_sample flag would therefore also change randn_like diffusion noise.
    A same-layout/dtype destination retains the existing inference precision.
    """
    if mode == 'sample':
        return result
    if mode != 'mean':
        raise ValueError(mode)
    center = result.posterior.mode().squeeze(2)
    if center.shape != result.latent.shape:
        raise ValueError('posterior mean and sample shapes differ')
    latent = torch.empty_like(result.latent).copy_(center)
    return result._replace(latent=latent)


def observe_runner(runner, mode, records):
    """Always run upstream sampling; mean path substitutes only its values."""
    if mode not in ('sample', 'mean'):
        raise ValueError(mode)
    runner.config.vae.use_sample = True
    original_encode, original_condition = runner.vae_encode, runner.get_condition
    original_posterior = runner.vae.encode

    @torch.no_grad()
    def posterior(*args, **kwargs):
        return posterior_condition(original_posterior(*args, **kwargs), mode)

    runner.vae.encode = posterior

    @torch.no_grad()
    def encode(samples):
        before = rng_identity()
        latents = original_encode(samples)
        records.append(dict(before_vae=before, after_vae=rng_identity(),
                            conditions=[tensor_identity(z) for z in latents]))
        return latents

    @torch.no_grad()
    def condition(latent, latent_blur, task):
        if not records or 'diffusion_noise' in records[-1]:
            raise RuntimeError('expected exactly one VAE call per inference window')
        records[-1].update(diffusion_noise=tensor_identity(latent),
                           effective_condition=tensor_identity(latent_blur),
                           before_diffusion=rng_identity())
        return original_condition(latent=latent, latent_blur=latent_blur, task=task)

    runner.vae_encode, runner.get_condition = encode, condition


def restore(enhanced, control, adapter, packets):
    from demo import stage_c_a800_teacher as teacher
    mode = validate_condition(torch.load(adapter, weights_only=True, map_location='cpu'))
    original = teacher.PersistentSeedVR2
    records = []

    class Conditioned(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            observe_runner(self.runner, mode, records)

    with patch.object(teacher, 'PersistentSeedVR2', Conditioned):
        pixels, report = PARENT_RESTORE(enhanced, control, adapter, packets)
    if len(records) != len(report['windows']) or not records:
        raise RuntimeError('incomplete conditioning evidence')
    if any('diffusion_noise' not in r for r in records):
        raise RuntimeError('missing diffusion noise evidence')
    report.update(vae_condition=mode, condition_path_format=FORMAT, condition_windows=records,
                  cast_policy='unchanged: FP32 interface addition, DiT BF16 autocast')
    return pixels, report


def decode(args):
    with patch.object(parent, 'identities', identities), patch.object(parent, 'restore', restore):
        parent.decode(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stream', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--disable-generation', action='store_true')
    decode(parser.parse_args())
