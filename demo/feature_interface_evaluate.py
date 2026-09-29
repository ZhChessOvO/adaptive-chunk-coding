"""Real-stream paired interface evaluation; source is only read by the evaluator."""
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch

from demo.chunk_enhancement_experiment import read
from demo.conditioned_generation_evaluate import OLD, MODES
from demo.conditioned_generation_pipeline import execute
from demo.feature_condition_pipeline import DEFAULT as PREVIOUS
from demo.feature_interface_train import INITIAL, CACHE
from demo.feature_interface_decode import identities
from demo.feature_interface_model import validate_bundle
from demo import scalable_cooperation_format as fmt
from demo.patch_prefix_probe import load_frames, verify_artifacts
from demo.scalable_codec import atomic_bytes, atomic_json, file_hash
from demo.scalable_format import frame_hash
from demo.scalable_experiment import load_source, quality, resources
from demo.scalable_cooperation_experiment import region_metrics
from demo.stage_c_three_path_roi_probe import LPIPSAlex


def audit_training(root):
    expected = read(root/'training.complete.json')['steps']
    initial = torch.load(INITIAL,weights_only=True,map_location='cpu')
    logs,configs,summaries = {},{},{}
    for name in ('actual','zero'):
        directory = root/name
        info,config = read(directory/'complete.json'),read(directory/'config.json')
        model = torch.load(directory/'adapter.pt',weights_only=True,map_location='cpu')
        resume = torch.load(directory/'resume.pt',weights_only=True,map_location='cpu')
        final = torch.load(directory/'checkpoints'/f'adapter_{expected:06d}.pt',weights_only=True,map_location='cpu')
        assert info['steps'] == resume['step'] == expected
        assert info['adapter_sha256'] == file_hash(directory/'adapter.pt')
        assert validate_bundle(model) == name
        assert config == resume['config'] == model['metadata']['config']
        assert config['initial_adapter'] == file_hash(INITIAL)
        for section in ('state_dict','feature_state'):
            for key,value in model[section].items():
                assert torch.isfinite(value).all()
                torch.testing.assert_close(value,resume['adapter'][section][key],rtol=0,atol=0)
                torch.testing.assert_close(value,final[section][key],rtol=0,atol=0)
                if section == 'state_dict':
                    torch.testing.assert_close(value,initial[section][key],rtol=0,atol=0)
        rows = [json.loads(line) for line in (directory/'steps.jsonl').read_text().splitlines()]
        assert [r['step'] for r in rows] == list(range(1,expected+1))
        assert rows[0]['first_step_image_condition_gradient_norm'] > 0
        for row in rows:
            assert all(np.isfinite(row[k]) for k in ('loss','gradient_norm','condition_gradient_norm'))
            assert row['condition_gradient_norm'] > 0 and row['gradient_norm'] > 0
            assert row['condition_statistics']['outside_coverage_exact']
        assert dict(Counter(r['dataset'] for r in rows)) == info['dataset_steps']
        assert dict(Counter(r['condition'] for r in rows)) == info['condition_steps']
        logs[name],configs[name],summaries[name] = rows,config,info
    for a,b in zip(logs['actual'],logs['zero'],strict=True):
        assert all(a[k] == b[k] for k in ('step','sample','dataset','condition','crop','learning_rate'))
    assert {k:v for k,v in configs['actual'].items() if k != 'mode'} == {
        k:v for k,v in configs['zero'].items() if k != 'mode'}
    assert file_hash(CACHE) == configs['actual']['cache_hash']
    cache = read(CACHE)
    assert cache['complete'] and Counter(e['dataset'] for e in cache['entries']) == {'REDS':90,'UVG':30}
    for entry in cache['entries']:
        for path,key in [('path','sha256'),('feature_path','feature_hash'),('pair_path','pair_hash')]:
            assert file_hash(Path(entry[path])) == entry[key]
    info = dict(complete=True,steps=expected,paired_schedule_exact=True,paired_config_exact=True,
        frozen_lora_exact=True,checkpoints_exact=True,cache_hashes_verified=120,training=summaries)
    atomic_json(root/'training_audit.json',info)
    return info


def point(root,row,mode,candidate,adapter,hashes,source,metric,run,previous,disabled=False):
    sid = row['sample']['sample_id']; oldname,directname = MODES[mode]
    original = OLD/sid/f'{oldname}.acsg'
    assert file_hash(original) == row['points'][oldname]['stream_sha256']
    control,inner,_,_ = fmt.parse(original.read_bytes())
    control.update(hashes); control['strength'] = 1.
    wire = fmt.wrap(inner,control)
    assert len(wire) == original.stat().st_size and fmt.parse(wire)[1] == inner
    dest = root/sid/f'{candidate}_{mode}'; dest.mkdir(parents=True,exist_ok=True)
    stream = dest/'stream.acsg'; result_path = dest/'result.json'
    if result_path.exists():
        result = read(result_path); verify_artifacts(dest,result['artifacts'])
        assert stream.read_bytes() == wire
        assert result['candidate'] == candidate and result['mode'] == mode
        return result
    atomic_bytes(stream,wire)
    argv = ['--stream',stream,'--output',dest,'--adapter',
            Path('/nonexistent/generator.pt') if disabled else adapter]
    if disabled: argv += ['--disable-generation']
    started = time.monotonic()
    execute(run,f'{sid}_{candidate}_{mode}','feature_interface_decode.py',argv,distributed=True)
    pixels,d = load_frames(dest/'reconstruction.npz'),read(dest/'decode.json')
    verify_artifacts(OLD/sid/directname,row['points'][directname]['artifacts'])
    enhanced = load_frames(OLD/sid/directname/'reconstruction.npz')
    assert d['generation_input_hash'] == frame_hash(enhanced) and d['output_hash'] == frame_hash(pixels)
    assert not d['source_frames_read'] and d['feature_bytes_added'] == 0
    assert d['total_bytes'] == len(wire) == stream.stat().st_size
    alpha = fmt.weights(source.shape,control)
    np.testing.assert_array_equal(pixels[alpha == 0],enhanced[alpha == 0])
    if disabled:
        np.testing.assert_array_equal(pixels,enhanced)
        assert not d['generation_assets_validated']
    else:
        assert all(s['outside_coverage_exact'] for s in d['generation_runtime']['condition_statistics'])
        if mode == 'none' or candidate == 'branch_off':
            np.testing.assert_array_equal(pixels,load_frames(
                PREVIOUS/'evaluation'/sid/f'rgb_{mode}/reconstruction.npz'))
    regions,roi = region_metrics(source,pixels,row['metric_regions'],metric)
    result = dict(sample_id=sid,dataset=row['sample']['dataset'],mode=mode,candidate=candidate,
        bytes=len(wire),quality=quality(source,pixels,metric),per_region=regions,roi_quality=roi,
        process_wall_seconds=time.monotonic()-started,fresh_decode=d,
        same_image_payload=True,same_bytes_as_reference=True,previous=previous,
        direct=row['points'][directname],
        artifacts={p.name:file_hash(p) for p in (stream,dest/'reconstruction.npz',dest/'decode.json')})
    atomic_json(result_path,result)
    return result


def evaluate(output,run):
    from demo.feature_interface_pipeline import variant
    training = audit_training(output)
    root = output/'evaluation'; root.mkdir(exist_ok=True)
    references = read(OLD/'summary.json')['results']
    previous = {(r['sample_id'],r['mode'],r['candidate']):r
                for r in read(PREVIOUS/'evaluation/summary.json')['results']}
    adapters = {k:output/k/'adapter.pt' for k in ('actual','zero')}
    for name,mode in [('without_content','zero'),('shuffled','shuffle'),('branch_off','off')]:
        adapters[name] = variant(adapters['actual'],root/'models'/f'{name}.pt',mode)
    hashes = {k:identities(v) for k,v in adapters.items()}
    protocol = dict(version=1,code=file_hash(Path(__file__)),profiles=hashes,
        references=file_hash(OLD/'summary.json'),previous=file_hash(PREVIOUS/'evaluation/summary.json'),
        training_audit=file_hash(output/'training_audit.json'),modes=list(MODES),
        roles='four development clips; no independent generalization claim',
        comparisons='24 paired + 12 full-prefix ablations + 2 repeat/G-off = 38 fresh decodes')
    protocol_path = root/'protocol.json'
    if protocol_path.exists(): assert read(protocol_path) == protocol
    else: atomic_json(protocol_path,protocol)
    metric = LPIPSAlex(True); results,ablations,checks = [],[],[]
    for row in references:
        run.check(); sid = row['sample']['sample_id']; source = load_source(row['sample'])
        assert frame_hash(source) == row['source_hash']
        for mode in MODES:
            for name in ('rgb','feature'):
                old = previous[(sid,mode,name)]
                verify_artifacts(PREVIOUS/'evaluation'/sid/f'{name}_{mode}',old['artifacts'])
        def execute_point(name,mode,key=None,disabled=False):
            key = key or name
            result = point(root,row,mode,name,adapters[key],hashes[key],source,metric,run,
                {k:previous[(sid,mode,k)] for k in ('rgb','feature')},disabled)
            run.update(completed=len(results)+len(ablations)+len(checks)+1,total=38)
            return result
        for name in ('actual','zero'):
            for mode in MODES: results.append(execute_point(name,mode))
        for name in ('without_content','shuffled','branch_off'):
            ablations.append(execute_point(name,'full'))
        if row == references[0]:
            checks.append(execute_point('repeat','full','actual'))
            checks.append(execute_point('G_off','full','actual',True))
            np.testing.assert_array_equal(load_frames(root/sid/'actual_full/reconstruction.npz'),
                                          load_frames(root/sid/'repeat_full/reconstruction.npz'))
    atomic_json(root/'summary.json',dict(complete=True,results=results,ablations=ablations,
        checks=checks,protocol=protocol,training=training,repeat_exact=True,no_E_exact=True,
        branch_off_rgb_exact=True,G_off_without_weights_exact=True,
        elapsed_seconds=time.monotonic()-run.started,resources=resources()))
    run.update(phase='evaluation_complete',completed=38,total=38)
