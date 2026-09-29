"""Frozen paired evaluation of RGB and received-feature continuations.

Runs in tmux under the native GPU mutex. Original source is available only to
this CPU evaluator; each receiver is a separate source-free torchrun process.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from demo import scalable_cooperation_format as fmt
from demo.chunk_enhancement_experiment import Run, read
from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
from demo.conditioned_generation_pipeline import execute
from demo.conditioned_generation_evaluate import OLD, MODES
from demo.feature_condition_pipeline import DEFAULT
from demo.feature_condition_cache import PREVIOUS
from demo.feature_condition_decode import identities
from demo.patch_prefix_probe import load_frames, verify_artifacts
from demo.scalable_codec import atomic_bytes, atomic_json, file_hash
from demo.scalable_format import frame_hash
from demo.scalable_experiment import load_source, quality, resources
from demo.scalable_cooperation_experiment import region_metrics
from demo.stage_c_three_path_roi_probe import LPIPSAlex
from demo.stage_c_seedvr2_lora_utils import atomic_torch_save


def audit_training(root):
    summaries, logs, configs = {}, {}, {}
    assert read(root/'train.complete.json')['command'] == 'train'
    for name in ('rgb','feature'):
        directory = root/name
        info = read(directory/'complete.json')
        assert info['steps'] == 1000
        assert file_hash(directory/'adapter.pt') == info['adapter_sha256']
        model = torch.load(directory/'adapter.pt', weights_only=True, map_location='cpu')
        resume = torch.load(directory/'resume.pt', weights_only=True, map_location='cpu')
        final = torch.load(directory/'checkpoints/adapter_001000.pt',weights_only=True,map_location='cpu')
        assert resume['step'] == 1000 and model['feature_enabled'] == (name == 'feature')
        config = read(directory/'config.json')
        assert config == resume['config'] == model['metadata']['config']
        for section in ('state_dict','feature_state'):
            for key,value in model[section].items():
                assert torch.isfinite(value).all()
                torch.testing.assert_close(value,resume['adapter'][section][key],rtol=0,atol=0)
                torch.testing.assert_close(value,final[section][key],rtol=0,atol=0)
        rows = [json.loads(line) for line in (directory/'steps.jsonl').read_text().splitlines()]
        assert [r['step'] for r in rows] == list(range(1,1001))
        for row in rows:
            assert all(np.isfinite(row[k]) for k in ('loss','gradient_norm','feature_gradient_norm','feature_side_rms'))
            if row['condition'] == 'none' or name == 'rgb':
                assert row['feature_gradient_norm'] == row['feature_side_rms'] == 0
        assert dict(Counter(r['dataset'] for r in rows)) == info['dataset_steps']
        assert dict(Counter(r['condition'] for r in rows)) == info['condition_steps']
        summaries[name],logs[name],configs[name] = info,rows,config
    for a,b in zip(logs['rgb'],logs['feature'],strict=True):
        assert all(a[k] == b[k] for k in ('step','sample','dataset','condition','crop'))
    assert {k:v for k,v in configs['rgb'].items() if k != 'mode'} == {
            k:v for k,v in configs['feature'].items() if k != 'mode'}
    cache = read(root/'cache.json')
    assert cache['complete'] and len(cache['entries']) == 120
    assert Counter(e['dataset'] for e in cache['entries']) == {'REDS':90,'UVG':30}
    for e in cache['entries']:
        for path, key in [('path','sha256'),('feature_path','feature_hash'),('pair_path','pair_hash')]:
            assert file_hash(Path(e[path])) == e[key]
    info = dict(complete=True,steps=1000,paired_schedule_exact=True,paired_config_exact=True,
        exports_resume_checkpoint_exact=True,cache_hashes_verified=120,training=summaries,
        feature_last100_received_side_rms=float(np.mean([r['feature_side_rms']
            for r in logs['feature'][-100:] if r['feature_coverage'] > 0])),resources=resources())
    atomic_json(root/'training_audit.json', info)
    return info


def ablated_bundle(root):
    """Same trained LoRA and feature weights; explicitly disable the side branch."""
    target = root/'evaluation/models/feature_off.pt'
    source = torch.load(root/'feature/adapter.pt',weights_only=True,map_location='cpu')
    source['feature_enabled'] = False
    if target.exists():
        existing = torch.load(target,weights_only=True,map_location='cpu')
        assert not existing['feature_enabled']
        for section in ('state_dict','feature_state'):
            for k,v in source[section].items():
                torch.testing.assert_close(v,existing[section][k],rtol=0,atol=0)
    else:
        atomic_torch_save(target,source)
    return target


def point(root, row, mode, candidate, adapter, hashes, source, metric, run, previous, *, disabled=False):
    old_name,direct_name = MODES[mode]
    sid = row['sample']['sample_id']
    old_root = OLD/sid
    original = old_root/f'{old_name}.acsg'
    assert file_hash(original) == row['points'][old_name]['stream_sha256']
    control,inner,_,_ = fmt.parse(original.read_bytes())
    control.update(hashes); control['strength'] = 1.
    wire = fmt.wrap(inner,control)
    assert len(wire) == original.stat().st_size and fmt.parse(wire)[1] == inner
    dest = root/sid/f'{candidate}_{mode}'
    dest.mkdir(parents=True,exist_ok=True)
    stream = dest/'stream.acsg'
    result_file = dest/'result.json'
    if result_file.exists():
        result = read(result_file)
        verify_artifacts(dest,result['artifacts'])
        assert stream.read_bytes() == wire
        return result
    atomic_bytes(stream,wire)
    argv = ['--stream',stream,'--output',dest,'--adapter',
            Path('/nonexistent/generator.pt') if disabled else adapter]
    if disabled:
        argv += ['--disable-generation']
    begin = time.monotonic()
    execute(run,f'{sid}_{candidate}_{mode}','feature_condition_decode.py',argv,distributed=True)
    output,report = load_frames(dest/'reconstruction.npz'),read(dest/'decode.json')
    verify_artifacts(old_root/direct_name,row['points'][direct_name]['artifacts'])
    enhanced = load_frames(old_root/direct_name/'reconstruction.npz')
    assert report['generation_input_hash'] == frame_hash(enhanced)
    assert report['output_hash'] == frame_hash(output)
    assert not report['source_frames_read'] and report['feature_bytes_added'] == 0
    assert report['total_bytes'] == len(wire) == stream.stat().st_size
    alpha = fmt.weights(source.shape,control)
    np.testing.assert_array_equal(output[alpha == 0],enhanced[alpha == 0])
    if disabled:
        np.testing.assert_array_equal(output,enhanced)
        assert not report['generation_assets_validated']
    regions,roi = region_metrics(source,output,row['metric_regions'],metric)
    result = dict(sample_id=sid,dataset=row['sample']['dataset'],mode=mode,candidate=candidate,
        bytes=len(wire),quality=quality(source,output,metric),per_region=regions,roi_quality=roi,
        process_wall_seconds=time.monotonic()-begin,fresh_decode=report,
        same_image_payload=True,same_bytes_as_reference=True,
        previous=previous,direct=row['points'][direct_name],
        artifacts={p.name:file_hash(p) for p in (stream,dest/'reconstruction.npz',dest/'decode.json')})
    atomic_json(result_file,result)
    return result


def evaluate(args,run):
    training = audit_training(args.output)
    root = args.output/'evaluation'; root.mkdir(exist_ok=True)
    references = read(OLD/'summary.json')['results']
    previous = {(r['sample_id'],r['mode']):r for r in read(PREVIOUS/'evaluation/summary.json')['results']
                if r['candidate'] == 'image'}
    adapters = {k:args.output/k/'adapter.pt' for k in ('rgb','feature')}
    adapters['feature_off'] = ablated_bundle(args.output)
    hashes = {k:identities(v) for k,v in adapters.items()}
    protocol = dict(version=1,code=file_hash(Path(__file__)),profiles=hashes,
        reference=file_hash(OLD/'summary.json'),previous=file_hash(PREVIOUS/'evaluation/summary.json'),
        training_audit=file_hash(args.output/'training_audit.json'),modes=list(MODES),
        roles='4 existing development clips; UVG same sequences at other training times',
        comparisons='24 paired points + 4 full-prefix branch-off + 3 recovery/repeat checks')
    # Resource snapshots change on an audit rerun; scientific inputs do not.
    if (root/'protocol.json').exists():
        old = read(root/'protocol.json')
        assert {k:v for k,v in old.items() if k != 'training_audit'} == {
                k:v for k,v in protocol.items() if k != 'training_audit'}
        protocol = old
    else:
        atomic_json(root/'protocol.json',protocol)
    metric = LPIPSAlex(True)
    results,ablations,checks = [],[],[]
    for row in references:
        run.check()
        sid = row['sample']['sample_id']
        source = load_source(row['sample'])
        assert frame_hash(source) == row['source_hash']
        for mode in MODES:
            verify_artifacts(PREVIOUS/'evaluation'/sid/f'image_{mode}',previous[(sid,mode)]['artifacts'])
        def execute_point(name,mode,adapter_name=None,disabled=False):
            key = adapter_name or name
            return point(root,row,mode,name,adapters[key],hashes[key],source,metric,run,
                         previous[(sid,mode)],disabled=disabled)
        for name in ('rgb','feature'):
            for mode in MODES:
                results.append(execute_point(name,mode))
                run.update(completed=len(results),total=24)
        ablations.append(execute_point('feature_off','full'))
        if row == references[0]:
            checks.append(execute_point('repeat','full','feature'))
            checks.append(execute_point('off','full','feature',disabled=True))
            checks.append(execute_point('feature_off','none'))
            for a,b in [('feature_full','repeat_full'),('feature_none','feature_off_none')]:
                np.testing.assert_array_equal(load_frames(root/sid/a/'reconstruction.npz'),
                                              load_frames(root/sid/b/'reconstruction.npz'))
    atomic_json(root/'summary.json',dict(complete=True,results=results,ablations=ablations,
        checks=checks,protocol=protocol,training=training,repeat_exact=True,
        no_E_branch_off_exact=True,generation_off_without_weights_exact=True,
        elapsed_seconds=time.monotonic()-run.started,resources=resources()))
    run.update(phase='evaluation_complete',completed=31,total=31)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--max-hours',type=float,default=4.)
    args = p.parse_args(); args.command='feature_evaluation'
    run = Run(args); run.thread.start()
    try:
        with exclusive_native_evaluation(run):
            pids = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
            if pids: raise RuntimeError(f'GPU is busy: {pids}')
            evaluate(args,run)
    except BaseException as error:
        atomic_json(args.output/'evaluation_failure.json',dict(error=repr(error),phase=run.progress))
        raise
    finally:
        run.log_resources(); run.stop.set(); run.thread.join(timeout=3)
