"""Bounded, paired G-artifact diagnosis; no fitting or production-model changes.

Replays three saved receiver ROIs before blending. Checkpoints, input RGB,
geometry and seeds are fixed, and the current arm must reproduce saved pixels.
LoRA strengths and output-head ablations are diagnostics, not promoted models.
"""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from demo.chunk_enhancement_experiment import Run
from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
from demo.condition_path_decode import tensor_identity
from demo.routervc_fullview_probe import read, immutable, verify_artifacts
from demo.scalable_codec import atomic_json, atomic_npz, file_hash
from demo.scalable_experiment import quality
from demo.scalable_generation_decode import ASSETS
from demo.stage_c_a800_teacher import PersistentSeedVR2
from demo.stage_c_seedvr2_lora_utils import (
    lora_modules, load_lora_state_dict, set_lora_strength)
from demo.stage_c_three_path_roi_probe import LPIPSAlex

REPO = Path(__file__).resolve().parents[1]
RUNS = Path('/root/autodl-fs/DCVC/runs')
DEFAULT = RUNS/'routervc_texture_diagnostic_20261014'
CAPTURE = RUNS/'routervc_fusion_20261010/p1_controls'
CURRENT = RUNS/'a800_online_eg_20261002/joint/adapter.pt'
EARLY = Path('/root/autodl-tmp/DCVC/models/seedvr2/seedvr2_codec_lora_v1_1000step.pt')
CASES = (
    ('uvg-shakendry-f000-historical-evaluation-crop', 6),
    ('uvg-beauty-f000-historical-evaluation-crop', 6),
    ('reds-val-000-f000-n17-fullview', 9),
)
VARIANTS = (
    ('current', 'current', 1., 'all'),
    ('frozen', 'current', 0., 'all'),
    ('early_half', 'early', .5, 'all'),
    ('early_full', 'early', 1., 'all'),
    ('current_quarter', 'current', .25, 'all'),
    ('current_half', 'current', .5, 'all'),
    ('current_no_output_lora', 'current', 1., 'blocks'),
    ('current_output_lora_only', 'current', 1., 'output'),
    ('frozen_without_wrappers', 'current', 0., 'bypass'),
)


def phase_diagnostic(pixels, reference=None, period=16):
    """Fixed-phase high-pass bias, not a validated perceptual quality metric.

    Remove a 16px perimeter, spatially high-pass each frame, then average over
    complete period-sized cells and time. A stationary checker/grid survives;
    ordinary scene texture mostly cancels. Source controls remain necessary.
    """
    a = np.asarray(pixels, dtype=np.float32)
    if reference is not None:
        a = a - np.asarray(reference, dtype=np.float32)
    a = a[:, 16:-16, 16:-16]
    a = a - gaussian_filter(a, sigma=(0, 3, 3, 0))
    h, w = (a.shape[1]//period)*period, (a.shape[2]//period)*period
    if not h or not w:
        raise ValueError('insufficient interior for phase diagnostic')
    a = a[:, :h, :w]
    folded = a.reshape(a.shape[0], h//period, period, w//period, period, 3).mean((0, 1, 3))
    folded -= folded.mean((0, 1), keepdims=True)
    return dict(period=period, phase_rms_u8=float(np.sqrt(np.mean(folded**2))),
                highpass_rms_u8=float(np.sqrt(np.mean(a**2)))), folded


def cases():
    rows = {r['sample_id']: r for r in read(CAPTURE/'protocol.json')['rows']}
    result = []
    for sid, region in CASES:
        row = rows[sid]
        folder = CAPTURE/'samples'/sid
        received = read(folder/'received.json')
        verify_artifacts(folder, received['artifacts'])
        report = read(folder/f'g{region:02d}/result.json')
        verify_artifacts(folder/f'g{region:02d}', report['artifacts'])
        if file_hash(Path(row['source_path'])) != row['source_sha256']:
            raise ValueError('source changed')
        result.append(dict(row, region=region, crop=report['crop'], core=report['core'],
            folder=str(folder), seed=report['runtime']['windows'][0]['runtime']['seed'],
            received_sha256=file_hash(folder/'received.npz'),
            previous_raw_sha256=file_hash(folder/f'g{region:02d}/raw.npz'),
            previous_observation=report['runtime']['condition_windows'][0],
            role='previously used diagnostic; now used for artifact diagnosis'))
    return result


def core(pixels, case):
    x0, y0, _, _ = case['crop']; x, y, w, h = case['core']
    return pixels[:, y-y0:y-y0+h, x-x0:x-x0+w]


def run_one(model, frames, case, variant, states):
    name, checkpoint, strength, where = variant
    load_lora_state_dict(model.runner.dit, states[checkpoint])
    set_lora_strength(model.runner.dit, strength)
    modules = lora_modules(model.runner.dit)
    if where in ('blocks', 'output'):
        for key, module in modules.items():
            is_output = key == 'vid_out.proj'
            module.adapter_strength = float(is_output == (where == 'output'))
    runner = model.runner
    runner.config.vae.use_sample = False
    observations, latents = {}, []
    original_condition, original_decode = runner.get_condition, runner.vae_decode

    def observe_condition(latent, latent_blur, task):
        observations.update(condition=tensor_identity(latent_blur), noise=tensor_identity(latent))
        return original_condition(latent=latent, latent_blur=latent_blur, task=task)

    def observe_decode(values):
        latents.extend(z.detach().float().cpu().numpy() for z in values)
        return original_decode(values)

    with ExitStack() as stack:
        stack.enter_context(patch.object(runner, 'get_condition', observe_condition))
        stack.enter_context(patch.object(runner, 'vae_decode', observe_decode))
        if where == 'bypass':
            for module in modules.values():
                stack.enter_context(patch.object(module, 'forward', module.base.forward))
        with torch.no_grad():
            values, runtime = model.restore(list(frames), seed=case['seed'])
    if len(latents) != 1:
        raise ValueError('expected one decoded latent')
    return np.stack(values), latents[0], dict(runtime=runtime, **observations)


def draw(folder, case, source, received, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    variants = {r['name']: r for r in rows}
    names = ['source', 'received', 'frozen', 'early_half', 'current', 'current_half',
             'current_no_output_lora', 'current_output_lora_only']
    title = dict(source='Source', received='Received Y / no G', frozen='Original / no LoRA',
        early_half='First LoRA x 0.5', current='Current LoRA x 1', current_half='Current LoRA x 0.5',
        current_no_output_lora='Current: output LoRA OFF',
        current_output_lora_only='Current: output LoRA ONLY')
    fig, axs = plt.subplots(2, 4, figsize=(14, 8), layout='constrained')
    for ax, name in zip(axs.flat, names):
        if name in ('source', 'received'):
            value = core(source if name == 'source' else received, case)
        else:
            with np.load(folder/name/'output.npz') as z:
                value = core(z['pixels'], case)
        ax.imshow(value[8], interpolation='nearest'); ax.axis('off')
        label = title[name]
        if name in variants:
            q = variants[name]['quality']
            label += f"\nLPIPS {q['lpips_alex']:.4f} | PSNR {q['psnr_db']:.2f}"
        ax.set_title(label, fontsize=11)
    fig.suptitle(case['sample_id']+f" | region {case['region']} | frame 9 | BEFORE fusion", fontsize=12)
    fig.savefig(folder/'comparison.png', dpi=160); plt.close(fig)


def main(args):
    if not os.environ.get('TMUX'):
        raise RuntimeError('tmux required')
    run = Run(SimpleNamespace(output=args.output, command='diagnostic', max_hours=2))
    run.thread.start(); run.log_resources()
    try:
        selected = cases()
        protocol = dict(version=1, scope='paired raw-G diagnostic; no fitting, no new codec RD claim',
            cases=selected, variants=[list(v) for v in VARIANTS],
            current=dict(path=str(CURRENT), sha256=file_hash(CURRENT)),
            early=dict(path=str(EARLY), sha256=file_hash(EARLY)),
            assets={k:file_hash(v) for k, v in ASSETS.items()},
            code=file_hash(Path(__file__)), phase_metric='descriptive 16px phase bias, not quality')
        immutable(args.output/'protocol.json', protocol)
        if (args.output/'complete.json').exists():
            done = read(args.output/'complete.json'); verify_artifacts(args.output, done['artifacts'])
            for case in selected:
                for variant in VARIANTS:
                    p = args.output/case['sample_id']/variant[0]
                    verify_artifacts(p, read(p/'result.json')['artifacts'])
            print('Verified completed diagnostics without inference', flush=True)
            return
        completed, all_rows = 0, []
        with exclusive_native_evaluation(run):
            model, metric = None, None
            states = {k:torch.load(p, weights_only=True, map_location='cpu')['state_dict']
                      for k, p in [('current', CURRENT), ('early', EARLY)]}
            for case in selected:
                folder = args.output/case['sample_id']; folder.mkdir(exist_ok=True)
                old = Path(case['folder'])
                x, y, w, h = case['crop']
                with np.load(old/'received.npz') as z:
                    received = z['enhanced'][:, y:y+h, x:x+w].copy()
                with np.load(case['source_path']) as z:
                    source = z['source'][:, y:y+h, x:x+w].copy()
                atomic_npz(folder/'inputs.npz', source=source, received=received)
                rows, expected_pair = [], None
                for variant in VARIANTS:
                    run.check(); name = variant[0]; dest = folder/name; dest.mkdir(exist_ok=True)
                    if (dest/'result.json').exists():
                        row = read(dest/'result.json'); verify_artifacts(dest, row['artifacts'])
                    else:
                        if model is None:
                            model = PersistentSeedVR2(SimpleNamespace(upstream_root=REPO/'third_party/SeedVR2',
                                dit_checkpoint=ASSETS['dit'], vae_checkpoint=ASSETS['vae'],
                                positive_embedding=ASSETS['positive'], negative_embedding=ASSETS['negative'],
                                lora_checkpoint=CURRENT, lora_strength=1., sample_steps=1,
                                cfg_scale=1., dit_dtype='bfloat16'))
                            metric = LPIPSAlex(True)
                        started = time.monotonic()
                        pixels, latent, observation = run_one(model, received, case, variant, states)
                        current_exact = None
                        if name == 'current':
                            with np.load(old/f"g{case['region']:02d}/raw.npz") as z:
                                np.testing.assert_array_equal(pixels, z['pixels'])
                            prev = case['previous_observation']
                            assert observation['condition'] == prev['conditions'][0]
                            assert observation['noise'] == prev['diffusion_noise']
                            current_exact = True
                        if name == 'frozen_without_wrappers':
                            with np.load(folder/'frozen/output.npz') as z:
                                np.testing.assert_array_equal(pixels, z['pixels'])
                        q = quality(core(source, case), core(pixels, case), metric)
                        phase, folded = phase_diagnostic(core(pixels, case))
                        error_phase, _ = phase_diagnostic(core(pixels, case), core(source, case))
                        atomic_npz(dest/'output.npz', pixels=pixels, latent=latent, phase16=folded)
                        row = dict(name=name, quality=q, phase16=phase, error_phase16=error_phase,
                            observation=observation, current_replay_exact=current_exact,
                            frozen_bypass_exact=True if name == 'frozen_without_wrappers' else None,
                            seconds=time.monotonic()-started,
                            artifacts={'output.npz':file_hash(dest/'output.npz')})
                        atomic_json(dest/'result.json', row)
                    pair = {k:row['observation'][k] for k in ('condition', 'noise')}
                    if expected_pair is None: expected_pair = pair
                    if pair != expected_pair: raise ValueError('condition or diffusion noise changed')
                    rows.append(row); completed += 1
                    run.update(phase='paired_raw_G', completed=completed,
                        total=len(CASES)*len(VARIANTS), sample=case['sample_id'], variant=name)
                    print(json.dumps(dict(sample=case['sample_id'], variant=name, quality=row['quality'],
                                          phase16=row['phase16'])), flush=True)
                    if args.stop_after and completed >= args.stop_after:
                        return
                controls = {k:phase_diagnostic(core(v, case))[0]
                            for k, v in [('source', source), ('received', received)]}
                draw(folder, case, source, received, rows)
                result = dict(case=case, controls=controls, results=rows, paired_condition_noise=True)
                atomic_json(folder/'summary.json', result); all_rows.append(result)
        atomic_json(args.output/'summary.json', dict(cases=all_rows,
            conclusion_scope='diagnostic only; no automatic model replacement',
            current_replay_exact=all(r['results'][0]['current_replay_exact'] for r in all_rows)))
        names = ['protocol.json', 'summary.json']
        names += [f"{c['sample_id']}/{n}" for c in selected for n in ('comparison.png', 'summary.json', 'inputs.npz')]
        atomic_json(args.output/'complete.json', dict(complete=True, points=completed,
            seconds_this_process=time.monotonic()-run.started,
            artifacts={n:file_hash(args.output/n) for n in names}))
        run.update(phase='complete', completed=completed); run.log_resources()
    except BaseException as error:
        atomic_json(args.output/'last_failure.json', dict(error=repr(error), progress=run.progress))
        raise
    finally:
        if torch.distributed.is_initialized(): torch.distributed.destroy_process_group()
        run.stop.set(); run.thread.join(); run.lock.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=DEFAULT)
    p.add_argument('--stop-after', type=int, default=0)
    main(p.parse_args())
