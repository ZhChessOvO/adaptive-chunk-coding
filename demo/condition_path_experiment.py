"""Bounded, restartable 32-decode VAE condition comparison, without training."""
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
sys.path.insert(0, str(REPO))
from demo.chunk_enhancement_experiment import Run, read
from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
from demo.conditioned_generation_pipeline import execute
from demo.conditioned_generation_evaluate import OLD, MODES
from demo.condition_path_decode import FORMAT, identities, validate_condition
from demo.feature_interface_train import DEFAULT as INTERFACE
from demo.feature_condition_pipeline import DEFAULT as PREVIOUS
from demo.feature_condition_report import CLIPS
from demo import scalable_cooperation_format as fmt
from demo.patch_prefix_probe import load_frames, verify_artifacts
from demo.scalable_codec import atomic_bytes, atomic_json, file_hash
from demo.scalable_format import frame_hash
from demo.scalable_experiment import load_source, quality, resources
from demo.scalable_cooperation_experiment import region_metrics
from demo.stage_c_three_path_roi_probe import LPIPSAlex
from demo.stage_c_seedvr2_lora_utils import atomic_torch_save

DEFAULT = Path('/root/autodl-fs/DCVC/runs/a800_condition_path_20260929')
ARMS = ('rgb', 'zero', 'actual')
CONDITIONS = ('sample', 'mean')


def make_bundle(source, target, mode):
    value = torch.load(source, weights_only=True, map_location='cpu')
    value.update(condition_path_format=FORMAT, vae_condition=mode)
    if target.exists():
        saved = torch.load(target, weights_only=True, map_location='cpu')
        assert validate_condition(saved) == mode
        for key, section in value.items():
            if key in ('state_dict', 'feature_state'):
                assert section.keys() == saved[key].keys()
                for name, tensor in section.items():
                    torch.testing.assert_close(tensor, saved[key][name], rtol=0, atol=0)
            else:
                assert section == saved[key]
    else:
        atomic_torch_save(target, value)
    return target


def prepare(root):
    # Recheck every old code file pinned by the completed training queue.
    for name, sha in read(INTERFACE/'queue_protocol.json')['code'].items():
        assert file_hash(REPO/'demo'/name) == sha, f'pinned old file changed: {name}'
    assert read(INTERFACE/'evaluation/audit.json')['complete']
    sources = dict(rgb=INTERFACE/'evaluation/models/branch_off.pt',
                   zero=INTERFACE/'zero/adapter.pt', actual=INTERFACE/'actual/adapter.pt')
    initial = torch.load(PREVIOUS/'rgb/adapter.pt', weights_only=True, map_location='cpu')
    for arm, path in sources.items():
        bundle = torch.load(path, weights_only=True, map_location='cpu')
        for k, v in bundle['state_dict'].items():
            torch.testing.assert_close(v, initial['state_dict'][k], rtol=0, atol=0)
        if arm != 'rgb':
            assert file_hash(path) == read(INTERFACE/arm/'complete.json')['adapter_sha256']
    adapters = {(a,c):make_bundle(sources[a],root/'models'/f'{a}_{c}.pt',c)
                for a in ARMS for c in CONDITIONS}
    hashes = {k:identities(v) for k,v in adapters.items()}
    protocol = dict(version=1, sources={k:file_hash(v) for k,v in sources.items()},
        profiles={f'{a}_{c}':v for (a,c),v in hashes.items()},
        reference=file_hash(OLD/'summary.json'),
        old_results=file_hash(INTERFACE/'evaluation/summary.json'),
        initial_lora=file_hash(PREVIOUS/'rgb/adapter.pt'),
        code={n:file_hash(REPO/'demo'/n) for n in (
            'condition_path_decode.py','condition_path_experiment.py','run_condition_path.sh')},
        clips=list(CLIPS), role='Reused development clips; not independent generalization',
        design='24 full-E + 6 no-E first clip + 1 repeat + 1 G-off = 32 fresh decodes',
        cast_policy='unchanged FP32 interface, BF16 DiT', only_factor='VAE posterior sample vs mean')
    path = root/'protocol.json'
    if path.exists(): assert read(path) == protocol, 'resume protocol changed'
    else: atomic_json(path, protocol)
    return adapters, hashes, protocol


def destination(root, sid, arm, condition, prefix, check=''):
    return root/sid/f'{arm}_{condition}_{prefix}{"_"+check if check else ""}'


def validate_point(dest, result, row):
    verify_artifacts(dest, result['artifacts'])
    prefix, arm, condition = [result[k] for k in ('prefix','arm','condition')]
    oldname, directname = MODES[prefix]
    control, inner, _, _ = fmt.parse((dest/'stream.acsg').read_bytes())
    original, oldinner, _, _ = fmt.parse((OLD/result['sample_id']/f'{oldname}.acsg').read_bytes())
    assert inner == oldinner
    for k, v in original.items():
        if k not in ('profile','lora','strength'): assert control[k] == v
    assert control['strength'] == 1.
    d = read(dest/'decode.json')
    assert result['fresh_decode'] == d
    assert result['bytes'] == d['total_bytes'] == (dest/'stream.acsg').stat().st_size
    assert result['bytes'] == (OLD/result['sample_id']/f'{oldname}.acsg').stat().st_size
    assert sum(d[k] for k in ('base_bytes','container_header_bytes','packet_bytes',
                             'incomplete_tail_bytes','generation_control_bytes')) == d['total_bytes']
    assert not d['source_frames_read'] and d['feature_bytes_added'] == 0
    direct = OLD/result['sample_id']/directname
    verify_artifacts(direct, row['points'][directname]['artifacts'])
    enhanced, pixels = load_frames(direct/'reconstruction.npz'), load_frames(dest/'reconstruction.npz')
    assert d['output_hash'] == frame_hash(pixels) and d['generation_input_hash'] == frame_hash(enhanced)
    base = load_frames(OLD/result['sample_id']/'base/reconstruction.npz')
    assert d['base_hash'] == frame_hash(base)
    alpha = fmt.weights(pixels.shape,control)
    np.testing.assert_array_equal(pixels[alpha == 0],enhanced[alpha == 0])
    if result['check'] == 'G_off':
        np.testing.assert_array_equal(pixels,enhanced)
        assert not d['generation_assets_validated'] and not d['generation_executed']
    else:
        assert d['assets']['profile'] == control['profile'] and d['assets']['lora'] == control['lora']
        runtime = d['generation_runtime']
        assert runtime['vae_condition'] == condition
        assert len(runtime['condition_windows']) == len(runtime['windows']) > 0
        assert all(s['outside_coverage_exact'] for s in runtime['condition_statistics'])
        if condition == 'sample':
            if prefix == 'none': previous = PREVIOUS/'evaluation'/result['sample_id']/'rgb_none'
            else: previous = INTERFACE/'evaluation'/result['sample_id']/f'{"branch_off" if arm=="rgb" else arm}_full'
            verify_artifacts(previous, read(previous/'result.json')['artifacts'])
            np.testing.assert_array_equal(pixels,load_frames(previous/'reconstruction.npz'))
    return pixels


def assert_pair(sample, mean):
    assert sample['bytes'] == mean['bytes']
    assert sample['fresh_decode']['generation_input_hash'] == mean['fresh_decode']['generation_input_hash']
    a = sample['fresh_decode']['generation_runtime']['condition_windows']
    b = mean['fresh_decode']['generation_runtime']['condition_windows']
    for x, y in zip(a, b, strict=True):
        for key in ('before_vae','after_vae','before_diffusion','diffusion_noise'):
            assert x[key] == y[key], f'paired randomness differs: {key}'
        assert len(x['conditions']) == len(y['conditions']) == 1
        for key in ('shape','dtype'):
            assert x['conditions'][0][key] == y['conditions'][0][key]


def point(root,row,arm,condition,prefix,adapters,hashes,source,metric,run,check=''):
    sid = row['sample']['sample_id']
    dest = destination(root,sid,arm,condition,prefix,check); dest.mkdir(parents=True,exist_ok=True)
    oldname, directname = MODES[prefix]
    original = OLD/sid/f'{oldname}.acsg'
    assert file_hash(original) == row['points'][oldname]['stream_sha256']
    control, inner, _, _ = fmt.parse(original.read_bytes())
    control.update(hashes[(arm,condition)]); control['strength'] = 1.
    wire = fmt.wrap(inner,control)
    if (dest/'result.json').exists():
        result = read(dest/'result.json')
        assert (dest/'stream.acsg').read_bytes() == wire
        validate_point(dest,result,row)
        return result
    atomic_bytes(dest/'stream.acsg',wire)
    started = time.monotonic()
    disabled = check == 'G_off'
    execute(run,f'{sid}_{dest.name}','condition_path_decode.py',
        ['--stream',dest/'stream.acsg','--output',dest,'--adapter',
         Path('/nonexistent/generator.pt') if disabled else adapters[(arm,condition)],
         *(['--disable-generation'] if disabled else [])],distributed=True)
    pixels = load_frames(dest/'reconstruction.npz')
    regions, roi = region_metrics(source,pixels,row['metric_regions'],metric)
    result = dict(sample_id=sid,dataset=row['sample']['dataset'],arm=arm,condition=condition,
        prefix=prefix,check=check,bytes=len(wire),quality=quality(source,pixels,metric),
        per_region=regions,roi_quality=roi,fresh_decode=read(dest/'decode.json'),
        process_wall_seconds=time.monotonic()-started,direct=row['points'][directname],
        artifacts={p.name:file_hash(p) for p in (dest/'stream.acsg',dest/'decode.json',dest/'reconstruction.npz')})
    validate_point(dest,result,row)
    atomic_json(dest/'result.json',result)
    print(json.dumps(dict(completed=dest.name,sample=sid,lpips=roi['lpips_alex'])),flush=True)
    return result


def evaluate(root,run):
    adapters, hashes, protocol = prepare(root)
    references = read(OLD/'summary.json')['results']
    assert [r['sample']['sample_id'] for r in references] == list(CLIPS)
    results, checks = [], []
    metric = LPIPSAlex(True)
    for row in references:
        run.check(); sid = row['sample']['sample_id']; source = load_source(row['sample'])
        assert frame_hash(source) == row['source_hash']
        for prefix in (('full','none') if sid == next(iter(CLIPS)) else ('full',)):
            for arm in ARMS:
                pair = []
                for condition in CONDITIONS:
                    item = point(root,row,arm,condition,prefix,adapters,hashes,source,metric,run)
                    results.append(item); pair.append(item)
                    run.update(completed=len(results)+len(checks),total=32)
                assert_pair(*pair)
        if sid == next(iter(CLIPS)):
            for condition in CONDITIONS:
                arrays = [load_frames(destination(root,sid,a,condition,'none')/'reconstruction.npz') for a in ARMS]
                for arr in arrays[1:]: np.testing.assert_array_equal(arr,arrays[0])
            for check in ('repeat','G_off'):
                item = point(root,row,'actual','mean','full',adapters,hashes,source,metric,run,check)
                checks.append(item); run.update(completed=len(results)+len(checks),total=32)
            np.testing.assert_array_equal(load_frames(destination(root,sid,'actual','mean','full')/'reconstruction.npz'),
                load_frames(destination(root,sid,'actual','mean','full','repeat')/'reconstruction.npz'))
    atomic_json(root/'summary.json',dict(complete=True,results=results,checks=checks,protocol=protocol,
        paired_noise_exact=True,sample_legacy_exact=True,no_E_branch_exact=True,repeat_exact=True,
        G_off_without_generator_exact=True,elapsed_seconds=time.monotonic()-run.started,resources=resources()))


def main(args):
    run = Run(args); run.thread.start()
    try:
        if args.command == 'run':
            with exclusive_native_evaluation(run):
                pids = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
                if pids: raise RuntimeError(f'GPU busy: {pids}')
                evaluate(args.output,run)
        from demo.condition_path_report import report
        report(args.output)
        run.update(phase='complete',completed=32,total=32)
        atomic_json(args.output/f'{args.command}.complete.json',dict(complete=True,
            elapsed_seconds=time.monotonic()-run.started,resources=resources()))
    except BaseException as error:
        atomic_json(args.output/'last_failure.json',dict(error=repr(error),phase=run.progress)); raise
    finally:
        run.log_resources(); run.stop.set(); run.thread.join(timeout=3)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command',choices=['run','report'])
    p.add_argument('--output',type=Path,default=DEFAULT); p.add_argument('--max-hours',type=float,default=2.)
    args = p.parse_args()
    if not os.environ.get('TMUX'): p.error('long tasks must run inside tmux')
    main(args)
