"""Fresh, zero-contribution LoRA initialization and immutable paired protocol.

The old adapter supplies architecture/shapes ONLY. Every A is resampled and
every B is zero; no learned tensor or training metadata is inherited.
"""
import copy
import math
from pathlib import Path

import torch

from demo.routervc_fullview_probe import read, digest, immutable, verify_artifacts
from demo.scalable_codec import atomic_json
from demo.stage_c_seedvr2_lora_utils import atomic_torch_save
from routervc.generation.texture_data import CURRENT, DEFAULT as PREVIOUS, REPO, RUNS

DEFAULT = RUNS / 'routervc_texture_clean_20261014'
ARMS = ('control', 'phase_only')
MILESTONES = (100, 400, 1000, 2000)
INITIALIZATION_SEED = 2026101401
CODE = (
    'routervc/generation/texture_clean.py',
    'routervc/generation/texture_ablation.py',
    'tools/texture_clean.py', 'tools/texture_clean_evaluate.py',
    'tools/texture_clean_report.py', 'tools/run_texture_clean.sh',
    'tools/texture_ablation.py', 'tools/texture_repair_evaluate.py',
    'tools/texture_repair_review.py', 'tools/texture_diagnostic.py',
    'tools/texture_ablation_report.py', 'tools/texture_ablation_detail.py',
    'demo/stage_c_a800_teacher.py', 'demo/roi_condition_train.py',
    'demo/chunk_enhancement_experiment.py', 'tools/storage_guard.py',
)


def fresh_payload(template, seed=INITIALIZATION_SEED):
    keys = ('format_version', 'adapter_type', 'rank', 'alpha', 'last_n_blocks',
            'target_modules', 'trainable_parameter_count')
    payload = {k: copy.deepcopy(template[k]) for k in keys}
    generator = torch.Generator(device='cpu').manual_seed(seed)
    state = {}
    for name in template['target_modules']:
        for suffix in ('lora_a', 'lora_b'):
            key = name + '.' + suffix
            value = torch.empty(template['state_dict'][key].shape, dtype=torch.float32)
            if suffix == 'lora_a':
                torch.nn.init.kaiming_uniform_(value, a=math.sqrt(5), generator=generator)
            else:
                value.zero_()
            state[key] = value
    if set(state) != set(template['state_dict']):
        raise ValueError('unexpected adapter schema')
    if sum(t.numel() for t in state.values()) != payload['trainable_parameter_count']:
        raise ValueError('adapter parameter count differs')
    payload.update(state_dict=state, metadata=dict(
        initialization='fresh CPU Kaiming A, zero B; old checkpoint shapes only',
        initialization_seed=seed, inherited_learned_parameters=False))
    return payload


def initialize(root):
    from demo.routervc_mixed_queue import exact
    folder = root / 'initial'
    template = torch.load(CURRENT, map_location='cpu', weights_only=True)
    expected = fresh_payload(template)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'adapter.pt'
    if path.exists():
        exact(torch.load(path, map_location='cpu', weights_only=True), expected)
    else:
        atomic_torch_save(path, expected)
    receipt = dict(complete=True, adapter_sha256=digest(path), seed=INITIALIZATION_SEED,
                   parameters=expected['trainable_parameter_count'],
                   old_checkpoint_used_for_shapes_only=True, all_B_exact_zero=True,
                   all_A_fresh_nonzero=all(bool(t.any()) for k,t in expected['state_dict'].items()
                                           if k.endswith('lora_a')))
    immutable(folder / 'complete.json', receipt)
    return path


def protocol(root, initial, smoke=False):
    original = PREVIOUS / ('smoke_v2' if smoke else 'formal') / 'protocol.json'
    result = copy.deepcopy(read(original))
    for name, expected in result['code'].items():
        if digest(REPO/name) != expected:
            raise ValueError('historical bound source changed: ' + name)
    old_eval = PREVIOUS / 'evaluation'
    verify_artifacts(old_eval, read(old_eval/'complete.json')['artifacts'])
    result.update(format='routervc_clean_lora_v1', steps=6 if smoke else 2000,
        initial=str(initial), initial_sha256=digest(initial), arms=list(ARMS),
        initialization_seed=INITIALIZATION_SEED, phase_weight=1.,
        learning_rate=dict(peak=1e-5, end=2e-6, warmup=6 if smoke else 100),
        transform='none: original LPIPS view in both arms', milestone_every=100,
        evaluation_milestones=[6] if smoke else list(MILESTONES),
        reference_evaluation=str(old_eval),
        reference_evaluation_sha256=digest(old_eval/'complete.json'),
        prior_training_protocol=digest(original),
        production=dict(path=str(CURRENT), sha256=digest(CURRENT)),
        control_scope='fresh LoRA, not continuation; equal 2000-step original/periodic objectives',
        success_policy='visible grid absence matters; no metric-only automatic promotion')
    result['code'].update({name:digest(REPO/name) for name in CODE})
    immutable(root / 'protocol.json', result)
    return result


def verify_bindings(config):
    for name, expected in config['code'].items():
        if digest(REPO/name) != expected:
            raise ValueError('bound source changed: ' + name)
    for asset in config['assets'].values():
        if digest(asset['path']) != asset['sha256']:
            raise ValueError('backbone asset changed')
    for path, expected in ((config['initial'], config['initial_sha256']),
                           (config['production']['path'], config['production']['sha256'])):
        if digest(path) != expected:
            raise ValueError('initial or production adapter changed')
