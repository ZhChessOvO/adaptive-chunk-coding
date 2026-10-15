"""Matched raw-G review of the one-coefficient mixed-view continuation.

Reuse authenticated mixed+400 pixels as the common starting model. New arms
have identical noise, inputs, field of view and additional training budgets.
Keep broad 512px views and slow previews; the 128px scores are not full RD.
"""
import argparse
from collections import defaultdict
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from demo.chunk_enhancement_codec import configure_torch
from demo.routervc_fullview_probe import read, digest, immutable, verify_artifacts
from demo.scalable_codec import atomic_json, atomic_npz
from demo.scalable_experiment import quality, now
from demo.stage_c_a800_teacher import PersistentSeedVR2
from demo.stage_c_seedvr2_lora_utils import load_lora_state_dict
from demo.stage_c_three_path_roi_probe import LPIPSAlex
from routervc.generation.context_geometry import rgb_crop, local_core, master_noise, noise_crop
from routervc.generation.texture_data import REPO
from tools.context_training_report import preview
from tools.g_objective_diagnostic import DEFAULT as DIAGNOSTIC
from tools.texture_clean import CleanRun
from tools.texture_context_probe import DEFAULT as CONTEXT, full_inputs, frozen_restore
from tools.texture_context_report import plotting
from tools.texture_diagnostic import phase_diagnostic
from tools.texture_repair_review import temporal_phase

ARMS = ('control', 'relaxed_rgb_temporal')
NAMES = ('initial', *ARMS)


def indexed(item):
    rows = {(r['name'], r['size']): r for r in item['results']}
    expected = {(n, s) for n in NAMES for s in (256, 512)}
    if len(item['results']) != 6 or set(rows) != expected:
        raise ValueError('missing or duplicate paired endpoint')
    for size in (256, 512):
        ref = rows['initial', size]
        for arm in ARMS:
            row = rows[arm, size]
            if row['crop'] != ref['crop']:
                raise ValueError('different inference views')
            for key in ('condition', 'noise'):
                if row['observation'][key] != ref['observation'][key]:
                    raise ValueError('different inference input: ' + key)
    return rows


def aggregate(cases):
    groups = defaultdict(list)
    for item in cases:
        for (name, size), row in indexed(item).items():
            groups[item['case']['evaluation_group'], item['case']['dataset'], name, size].append(row)
    result = []
    for (group, dataset, name, size), rows in sorted(groups.items()):
        record = dict(group=group, dataset=dataset, name=name, size=size, count=len(rows),
            lpips=float(np.mean([r['quality']['lpips_alex'] for r in rows])),
            psnr=float(np.mean([r['quality']['psnr_db'] for r in rows])),
            temporal_mae=float(np.mean([r['quality']['temporal_delta_mae'] for r in rows])),
            phase=float(np.mean([r['phase_error']['phase_rms_u8'] for r in rows])))
        if size == 512:
            record['wide_384_quality'] = {key: float(np.mean([r['wide_384_quality'][key] for r in rows]))
                for key in ('lpips_alex', 'psnr_db', 'temporal_delta_mae')}
        result.append(record)
    return result


def verify(root):
    done = read(root/'complete.json'); verify_artifacts(root, done['artifacts'])
    cfg = read(root/'protocol.json')
    for path, sha in cfg['inputs'].items():
        if digest(path) != sha: raise ValueError('bound input changed: ' + path)
    for spec in cfg['models'].values():
        if digest(spec['path']) != spec['sha256']: raise ValueError('model changed')
    for protocol_path in cfg['training_protocols']:
        train = read(protocol_path)
        for name, sha in train['code'].items():
            if digest(REPO/name) != sha: raise ValueError('bound source changed: ' + name)
        for spec in (*train['assets'].values(), train['initial']):
            if digest(spec['path']) != spec['sha256']: raise ValueError('bound asset changed')
    summary = read(root/'summary.json'); cases = summary['cases']
    if len(cases) != (2 if cfg['smoke'] else 15): raise ValueError('wrong case count')
    if [i['case']['sample_id'] for i in cases] != cfg['cases']: raise ValueError('case order changed')
    for item in cases:
        indexed(item)
        for row in item['results']:
            if digest(row['output_path']) != row['output_sha256']: raise ValueError('output changed')
            if row['name'] != 'initial':
                if row != read(Path(row['output_path']).parent/'result.json'):
                    raise ValueError('point journal differs')
                if cfg['smoke'] and not row['reload_repeat_exact']:
                    raise ValueError('missing actual adapter reload check')
    if done['new_points'] != 4*len(cases) or done['reused_initial_points'] != 2*len(cases):
        raise ValueError('wrong point count')
    if aggregate(cases) != summary['aggregate']: raise ValueError('aggregate differs')
    return done


def figures(output, item, source, received, historical, steps):
    case, geo = item['case'], item['geometry']; rows = indexed(item)
    panels = [('Source', rgb_crop(source, geo['large'])), ('Received Y', rgb_crop(received, geo['large']))]
    legacy = next(r for r in historical['results'] if r['name']=='legacy' and r['size']==512 and r['seed_index']==0)
    path = CONTEXT/'probe'/legacy['relative_dir']/'output.npz'
    if digest(path) != legacy['artifacts']['output.npz']: raise ValueError('legacy display reference changed')
    with np.load(path) as z: panels.append(('Legacy LoRA (historical)', z['pixels'].copy()))
    for name, title in (('initial', 'Starting mixed-view model'), ('control', f'Control +{steps}: RGB time 0.1'),
                        ('relaxed_rgb_temporal', f'Trial +{steps}: RGB time 0.025')):
        with np.load(rows[name, 512]['output_path']) as z: panels.append((title, z['pixels'].copy()))
    plt = plotting(); fig, axes = plt.subplots(2, 3, figsize=(13, 9), layout='constrained')
    for ax, (title, pixels) in zip(axes.flat, panels):
        ax.imshow(pixels[8], interpolation='nearest'); ax.set_title(title, fontsize=11); ax.axis('off')
    fig.suptitle(case['sample_id']+'\nWhole 512 view / frame 9 / raw G, not full-system RD', fontsize=11)
    fig.savefig(output/'wide_comparison.png', dpi=130); plt.close(fig)
    if case['evaluation_group'] == 'known_artifact': preview(output/'slow_preview.mp4', panels)


def chart(output, groups):
    plt = plotting(); fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout='constrained')
    for i, dataset in enumerate(('REDS', 'UVG')):
        for j, (key, label) in enumerate((('lpips','LPIPS (lower)'), ('psnr','PSNR (higher)'),
                                          ('temporal_mae','Temporal error (auxiliary)'))):
            for size in (256,512):
                rows = {r['name']:r for r in groups if r['group']=='internal_validation'
                        and r['dataset']==dataset and r['size']==size}
                if set(rows) != set(NAMES): continue
                axes[i,j].plot(range(3), [rows[n][key] for n in NAMES], 'o-', label=f'{size}px input')
            axes[i,j].set_title(dataset+' / '+label); axes[i,j].set_xticks(range(3), ['Start','Control','RGB time /4'])
            axes[i,j].grid(alpha=.25)
            if axes[i,j].lines: axes[i,j].legend(fontsize=8)
    fig.suptitle('Same 128px score core / previously used internal locations / not independent test or RD')
    fig.savefig(output/'summary.png', dpi=140); plt.close(fig)


def main(args):
    if not os.environ.get('TMUX'): raise RuntimeError('tmux required')
    output = args.root/('smoke/evaluation' if args.smoke else 'evaluation')
    if (output/'complete.json').exists(): print(json.dumps(verify(output)), flush=True); return
    configure_torch(); torch.use_deterministic_algorithms(True)
    run = CleanRun(SimpleNamespace(output=output, command='objective_trial_review', max_hours=3))
    run.thread.start(); run.log_resources()
    try:
        stage = args.root/('smoke' if args.smoke else 'formal')
        protocols = [stage/a/'protocol.json' for a in ARMS]
        train = read(protocols[0])
        for path in protocols:
            for name, sha in read(path)['code'].items():
                if digest(REPO/name) != sha: raise ValueError('bound source changed: '+name)
        paths = {a:stage/a/('direct' if args.smoke else '')/'adapter.pt' for a in ARMS}
        old_path = CONTEXT/'training/evaluation/summary.json'; probe_path = CONTEXT/'probe/summary.json'
        old_cfg = read(old_path.parent/'protocol.json')
        verify_artifacts(old_path.parent,read(old_path.parent/'complete.json')['artifacts'])
        if old_cfg['models']['mixed'] != train['initial']:
            raise ValueError('reference is not the starting mixed-view adapter')
        old = read(old_path); cases = old['cases']
        if args.smoke: cases = [cases[0], cases[2]]
        probe = {i['case']['sample_id']:i for i in read(probe_path)['cases']}
        inputs = {str(p):digest(p) for p in (*protocols, old_path, probe_path,
                                           old_path.parent/'protocol.json', old_path.parent/'complete.json')}
        for item in cases:
            for row in item['results']:
                if row['name']=='mixed': inputs[row['output_path']] = row['output_sha256']
            legacy = next(r for r in probe[item['case']['sample_id']]['results']
                          if r['name']=='legacy' and r['size']==512 and r['seed_index']==0)
            inputs[str(CONTEXT/'probe'/legacy['relative_dir']/'output.npz')] = legacy['artifacts']['output.npz']
        for path, sha in inputs.items():
            if digest(path) != sha: raise ValueError('reference changed: '+path)
        models = {n:dict(path=str(p),sha256=digest(p)) for n,p in paths.items()}
        models['initial'] = train['initial']
        cfg = dict(inputs=inputs, training_protocols=list(map(str, protocols)), models=models,
                   smoke=args.smoke, cases=[i['case']['sample_id'] for i in cases], production_changed=False)
        immutable(output/'protocol.json',cfg)
        states = {n:torch.load(p, map_location='cpu', weights_only=True)['state_dict'] for n,p in paths.items()}
        assets = {k:Path(v['path']) for k,v in train['assets'].items()}
        model = PersistentSeedVR2(SimpleNamespace(upstream_root=REPO/'third_party/SeedVR2',
            dit_checkpoint=assets['dit'], vae_checkpoint=assets['vae'], positive_embedding=assets['positive'],
            negative_embedding=assets['negative'], lora_checkpoint=paths['control'], lora_strength=1.,
            sample_steps=1, cfg_scale=1., dit_dtype='bfloat16'))
        metric = LPIPSAlex(True); all_items = []; n = 0
        for item in cases:
            run.check(); case, geo = item['case'], item['geometry']; source, received = full_inputs(case)
            folder = output/case['sample_id']; folder.mkdir(exist_ok=True)
            reference = {r['size']:r for r in item['results'] if r['name']=='mixed'}
            seed = reference[256]['binding']['seed']; master = master_noise(seed); rows = []
            wide = [geo['large'][0]+64, geo['large'][1]+64, 384, 384]
            for size, ref in reference.items():
                row = copy.deepcopy(ref); row.update(name='initial', provenance='authenticated_starting_mixed_view_model')
                if size == 512:
                    with np.load(row['output_path']) as z: value = local_core(z['pixels'], row['crop'], wide)
                    row['wide_384_quality'] = quality(rgb_crop(source,wide), value, metric)
                rows.append(row)
            for name, state in states.items():
                load_lora_state_dict(model.runner.dit, state)
                for size, label in ((256,'small'),(512,'large')):
                    run.check(); crop = geo[label]; dest = folder/f'{name}_{size}'; dest.mkdir(exist_ok=True)
                    binding = dict(protocol=digest(output/'protocol.json'), sample=case['sample_id'],
                                   name=name, size=size, seed=seed)
                    if (dest/'result.json').exists():
                        row = read(dest/'result.json')
                        if row['binding']!=binding or digest(row['output_path'])!=row['output_sha256']:
                            raise ValueError('resumed point changed')
                    else:
                        noise = noise_crop(master,crop,geo['large'])
                        pixels, obs = frozen_restore(model, rgb_crop(received,crop), noise, seed)
                        if any(obs[k]!=reference[size]['observation'][k] for k in ('condition','noise')):
                            raise ValueError('unpaired inference input')
                        if args.smoke:
                            other = next(a for a in ARMS if a!=name)
                            load_lora_state_dict(model.runner.dit, states[other]); load_lora_state_dict(model.runner.dit, state)
                            again, repeat_obs = frozen_restore(model,rgb_crop(received,crop),noise,seed)
                            np.testing.assert_array_equal(pixels,again)
                            if any(repeat_obs[k]!=obs[k] for k in ('condition','noise')):
                                raise ValueError('actual reload input mismatch')
                        target = rgb_crop(source,geo['core']); value = local_core(pixels,crop,geo['core'])
                        phase, _ = phase_diagnostic(value,target); path = dest/'output.npz'; atomic_npz(path,pixels=pixels)
                        row = dict(name=name,size=size,crop=crop,binding=binding,quality=quality(target,value,metric),
                            phase_error=phase,temporal_phase=temporal_phase(value,target),observation=obs,
                            output_path=str(path),output_sha256=digest(path),reload_repeat_exact=bool(args.smoke),
                            provenance='new_decode')
                        if size==512: row['wide_384_quality'] = quality(rgb_crop(source,wide),local_core(pixels,crop,wide),metric)
                        atomic_json(dest/'result.json',row)
                    rows.append(row); n+=1
                    run.update(phase='postfit_evaluation',completed=n,total=len(cases)*4,sample=case['sample_id'],arm=name,size=size)
                    print(json.dumps(dict(completed=n,sample=case['sample_id'],arm=name,size=size,quality=row['quality'])),flush=True)
            result = dict(case=case,geometry=geo,results=rows); indexed(result)
            figures(folder,result,source,received,probe[case['sample_id']],6 if args.smoke else 400); all_items.append(result)
            atomic_json(output/'partial_summary.json',dict(cases=all_items))
        groups = aggregate(all_items)
        atomic_json(output/'summary.json',dict(cases=all_items,aggregate=groups,
            scope='same 128 core plus separate 384 interior scores; raw G, not full-system RD or independent test'))
        if not args.smoke: chart(output,groups)
        names = ['protocol.json','summary.json'] + [str(p.relative_to(output)) for p in output.rglob('*')
            if p.is_file() and (p.suffix in ('.png','.mp4') or p.name=='result.json')]
        atomic_json(output/'complete.json',dict(complete=True,new_points=n,reused_initial_points=len(cases)*2,
            smoke=args.smoke,artifacts={p:digest(output/p) for p in names},utc=now()))
        verify(output); run.update(phase='complete_review_required')
    except BaseException as error:
        atomic_json(output/'last_failure.json',dict(error=repr(error),utc=now(),progress=run.progress)); raise
    finally:
        run.log_resources(); run.stop.set(); run.thread.join(); run.lock.close()
        if torch.distributed.is_initialized(): torch.distributed.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=DIAGNOSTIC/'trial')
    p.add_argument('--smoke',action='store_true');main(p.parse_args())
