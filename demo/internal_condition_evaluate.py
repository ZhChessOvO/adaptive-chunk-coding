"""Audit paired training and compare three arms on identical charged streams."""
from collections import Counter
import json
import time

import numpy as np
import torch

from demo.chunk_enhancement_experiment import read
from demo.internal_condition_train import INITIAL,CACHE,REPO
from demo.internal_condition_model import validate_bundle
from demo.internal_condition_decode import identities
from demo.internal_condition_pipeline import variant,decode_point,assert_noise,OLD,MODES
from demo.feature_condition_pipeline import DEFAULT as PREVIOUS
from demo.patch_prefix_probe import load_frames,verify_artifacts
from demo.scalable_codec import atomic_json,file_hash
from demo.scalable_format import frame_hash
from demo.scalable_experiment import load_source,quality,resources
from demo.scalable_cooperation_experiment import region_metrics
from demo.stage_c_three_path_roi_probe import LPIPSAlex

ARMS=('off','input','internal','zero')


def audit_training(root):
    expected=read(root/'training.complete.json')['steps']
    initial=torch.load(INITIAL,weights_only=True,map_location='cpu')
    logs,configs,summaries={},{},{}
    for name in ARMS[1:]:
        directory=root/name
        info,cfg=read(directory/'complete.json'),read(directory/'config.json')
        model=torch.load(directory/'adapter.pt',weights_only=True,map_location='cpu')
        resume=torch.load(directory/'resume.pt',weights_only=True,map_location='cpu')
        checkpoint=torch.load(directory/'checkpoints'/f'adapter_{expected:06d}.pt',weights_only=True,map_location='cpu')
        assert info['steps']==resume['step']==expected
        assert info['adapter_sha256']==file_hash(directory/'adapter.pt')
        assert validate_bundle(model)==('input' if name=='input' else 'internal',
                                        'zero' if name=='zero' else 'actual')
        assert cfg==resume['config']==model['metadata']['config']
        assert cfg['initial_adapter']==file_hash(INITIAL) and cfg['cache_hash']==file_hash(CACHE)
        for f,h in cfg['code'].items(): assert file_hash(REPO/'demo'/f)==h
        for section in ('state_dict','branch_state'):
            for k,v in model[section].items():
                assert torch.isfinite(v).all()
                torch.testing.assert_close(v,resume['adapter'][section][k],rtol=0,atol=0)
                torch.testing.assert_close(v,checkpoint[section][k],rtol=0,atol=0)
                if section=='state_dict': torch.testing.assert_close(v,initial[section][k],rtol=0,atol=0)
        rows=[json.loads(line) for line in (directory/'steps.jsonl').read_text().splitlines()]
        assert [r['step'] for r in rows]==list(range(1,expected+1))
        assert rows[0]['first_step_image_condition_gradient_norm']>0
        for r in rows:
            assert all(np.isfinite(r[k]) for k in ('loss','gradient_norm','condition_gradient_norm'))
            assert r['gradient_norm']>0 and r['condition_gradient_norm']>0
            assert r['condition_statistics']['outside_coverage_exact']
        assert dict(Counter(r['dataset'] for r in rows))==info['dataset_steps']
        logs[name],configs[name],summaries[name]=rows,cfg,info
    for name in ('input','zero'):
        assert {k:v for k,v in configs[name].items() if k!='mode'}=={
                k:v for k,v in configs['internal'].items() if k!='mode'}
        for a,b in zip(logs['internal'],logs[name],strict=True):
            for k in ('step','sample','dataset','condition','crop','learning_rate',
                      'raw_condition_identity','diffusion_noise_identity'):
                assert a[k]==b[k],f'nonpaired training: {k}'
    result=dict(complete=True,steps=expected,paired_schedule_condition_noise_exact=True,
        frozen_lora_exact=True,checkpoints_exact=True,training=summaries)
    atomic_json(root/'training_audit.json',result)
    return result


def evaluate(output,run):
    training=audit_training(output)
    root=output/'evaluation';root.mkdir(exist_ok=True)
    refs=read(OLD/'summary.json')['results']
    adapters={k:output/k/'adapter.pt' for k in ARMS[1:]}
    for name,mode in [('off','off'),('without','zero'),('shuffled','shuffle')]:
        adapters[name]=variant(adapters['internal'],root/'models'/f'{name}.pt',mode)
    protocol=dict(code=file_hash(REPO/'demo/internal_condition_evaluate.py'),
        profiles={k:identities(v) for k,v in adapters.items()},
        references=file_hash(OLD/'summary.json'),training_audit=file_hash(output/'training_audit.json'),
        roles='four previously reused development clips, not independent generalization',
        points='48 main + 8 same-weight content/alignment + repeat and G-off = 58')
    if (root/'protocol.json').exists(): assert read(root/'protocol.json')==protocol
    else: atomic_json(root/'protocol.json',protocol)
    metric=LPIPSAlex(True);results=[]
    for row in refs:
        sid=row['sample']['sample_id'];source=load_source(row['sample'])
        assert frame_hash(source)==row['source_hash']
        cases=[(k,m) for m in MODES for k in ARMS]+[('without','full'),('shuffled','full')]
        if row==refs[0]: cases += [('repeat','full'),('G_off','full')]
        reports={};paths={}
        for candidate,mode in cases:
            run.check();key='internal' if candidate in ('repeat','G_off') else candidate
            dest,d=decode_point(root,row,candidate,mode,adapters[key],run,candidate=='G_off')
            reports[candidate,mode]=d;paths[candidate,mode]=dest
            pixels=load_frames(dest/'reconstruction.npz')
            if candidate!='G_off': assert_noise(reports['off',mode],d)
            if mode=='none' or candidate=='repeat':
                reference=paths['off','none'] if mode=='none' else paths['internal','full']
                np.testing.assert_array_equal(pixels,load_frames(reference/'reconstruction.npz'))
            result_path=dest/'result.json'
            if result_path.exists():
                result=read(result_path);verify_artifacts(dest,result['artifacts'])
                assert result['fresh_decode']==d
            else:
                per_region,roi=region_metrics(source,pixels,row['metric_regions'],metric)
                result=dict(sample_id=sid,dataset=row['sample']['dataset'],candidate=candidate,mode=mode,
                    bytes=d['total_bytes'],quality=quality(source,pixels,metric),
                    per_region=per_region,roi_quality=roi,fresh_decode=d,
                    direct=row['points'][MODES[mode][1]],
                    artifacts={p.name:file_hash(p) for p in (dest/'stream.acsg',
                        dest/'decode.json',dest/'reconstruction.npz')})
                atomic_json(result_path,result)
            results.append(result);run.update(phase='evaluation',completed=len(results),total=58)
    assert len(results)==58
    atomic_json(root/'summary.json',dict(complete=True,results=results,training=training,
        protocol=protocol,paired_noise=True,no_E_exact=True,repeat_exact=True,
        G_off_without_weights_exact=True,elapsed_seconds=time.monotonic()-run.started,resources=resources()))
