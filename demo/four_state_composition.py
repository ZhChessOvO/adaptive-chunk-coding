"""Fixed mixed-stream check with noise-matched isolated-neighbor controls.

This is a diagnostic of the table's neighbor-E approximation, not a trained
Router, budget optimization or independent evaluation. No historical edits.
"""
import argparse
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from demo.four_state_core import (DEFAULT, ENHANCEMENT, ADAPTER, ROIS, rows, subset,
    seed_for, control, costs, aggregate_bytes, isolation, crop, verify, immutable_json)
from demo.four_state_receive import PersistentRGB
from demo import scalable_cooperation_format as fmt
from demo.chunk_enhancement_codec import configure_torch
from demo.chunk_enhancement_experiment import Run, read
from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
from demo.conditioned_generation_pipeline import execute
from demo.patch_prefix_probe import load_frames
from demo.scalable_codec import atomic_bytes, atomic_json, atomic_npz, file_hash
from demo.scalable_format import frame_hash
from demo.scalable_experiment import quality
from demo.stage_c_three_path_roi_probe import LPIPSAlex

MAPS = dict(stripes=['B','E','G','EG']*4,
            quadrants=[('B','E','G','EG')[(y//2)*2+x//2] for y in range(4) for x in range(4)])


def prepare(root):
    p = read(root/'protocol.json')
    output = root/'composition'
    output.mkdir(exist_ok=True)
    protocol = dict(code=file_hash(Path(__file__)), teacher=file_hash(root/'protocol.json'),
                    maps=MAPS, role='two reused component-training samples; fixed diagnostic')
    immutable_json(output/'protocol.json',protocol)
    jobs = []
    for row in rows(True):
        sid = row['sample_id']
        folder = root/'encoded'/sid
        verify(folder,read(folder/'complete.json')['artifacts'])
        bank = (folder/'packets.acse').read_bytes()
        for name, states in MAPS.items():
            target = output/sid/name
            target.mkdir(parents=True,exist_ok=True)
            es = [i for i,s in enumerate(states) if s in ('E','EG')]
            gs = [i for i,s in enumerate(states) if s in ('G','EG')]
            c = control(p['profile'],sid,0)
            c['generate'] = [[0,17,*ROIS[i]] for i in gs]
            inner = subset(bank,es)
            wire = fmt.wrap(inner,c)
            costs_all = [costs(bank,p['profile'],sid,i) for i in range(16)]
            expected = aggregate_bytes(costs_all[0]['base_container_bytes'],
                [v['e_packet_bytes'] for v in costs_all],states,costs_all[0]['g_shared_bytes'],fmt.REGION.size)
            assert expected == len(wire)
            path = target/'stream.acsg'
            if path.exists(): assert path.read_bytes() == wire
            else: atomic_bytes(path,wire)
            jobs.append(dict(sample_id=sid,name=name,states=states,es=es,gs=gs,
                             bytes=len(wire),sha256=file_hash(path)))
    immutable_json(output/'jobs.json',dict(jobs=jobs))


def isolated(root):
    """Receiver-side control: same seed/ROI/blend, remove only neighbor E."""
    import torch
    configure_torch()
    generator = None
    for job in read(root/'composition/jobs.json')['jobs']:
        folder = root/'composition'/job['sample_id']/job['name']
        done = folder/'isolated.json'
        if done.exists():
            verify(folder,read(done)['artifacts'])
            continue
        received = root/'received'/job['sample_id']
        verify(received,read(received/'E.json')['artifacts'])
        with np.load(received/'received_E.npz',allow_pickle=False) as cache:
            base, enhanced = cache['base'].copy(),cache['enhanced'].copy()
        c,_,_,_ = fmt.parse((folder/'stream.acsg').read_bytes())
        actual = read(folder/'fresh/decode.json')
        predicted = base.copy()
        for i in job['es']:
            x,y,w,h = ROIS[i]
            predicted[:,y:y+h,x:x+w] = enhanced[:,y:y+h,x:x+w]
        observation = []
        for ordinal,i in enumerate(job['gs']):
            single = dict(c,generate=[[0,17,*ROIS[i]]],seed=c['seed']+65536*ordinal)
            pixels = isolation(base,enhanced,i) if i in job['es'] else base
            if generator is None: generator = PersistentRGB()
            restored, runtime = generator(pixels,single)
            a,b = actual['generation_runtime']['condition_windows'][ordinal],runtime['condition_windows'][0]
            for key in ('before_vae','after_vae','before_diffusion','diffusion_noise'):
                assert a[key] == b[key], 'neighbor diagnostic changed noise'
            x,y,w,h = ROIS[i]
            predicted[:,y:y+h,x:x+w] = restored[:,y:y+h,x:x+w]
            observation.append(dict(region=i,seed=single['seed'],runtime=runtime))
        real = load_frames(folder/'fresh/reconstruction.npz')
        mask = fmt.weights(real.shape,c)
        np.testing.assert_array_equal(real[mask==0],predicted[mask==0])
        atomic_npz(folder/'isolated_prediction.npz',reconstruction=predicted)
        atomic_json(done,dict(source_frames_read=False,noise_matched=True,
            source=file_hash(folder/'fresh/decode.json'),outside_G_exact=True,
            mean_absolute_pixel_difference=float(np.abs(real.astype(float)-predicted).mean()),
            observation=observation,artifacts={'isolated_prediction.npz':file_hash(folder/'isolated_prediction.npz')}))
    if torch.distributed.is_initialized(): torch.distributed.destroy_process_group()


def report(root):
    configure_torch()
    metric = LPIPSAlex(True)
    entries = {r['sample_id']:r for r in rows(True)}
    results = []
    for job in read(root/'composition/jobs.json')['jobs']:
        folder = root/'composition'/job['sample_id']/job['name']
        done = folder/'result.json'
        if done.exists():
            r = read(done)
            verify(folder,r['artifacts'])
        else:
            row = entries[job['sample_id']]
            assert file_hash(Path(row['pair_path'])) == row['pair_hash']
            with np.load(row['pair_path'],allow_pickle=False) as cache: source=cache['source'].copy()
            actual = load_frames(folder/'fresh/reconstruction.npz')
            predicted = load_frames(folder/'isolated_prediction.npz')
            a = quality(source,actual,metric)
            b = quality(source,predicted,metric)
            r = dict(job,actual_quality=a,isolated_prediction_quality=b,
                local=[dict(region=i,actual=quality(crop(source,i),crop(actual,i),metric),
                            isolated=quality(crop(source,i),crop(predicted,i),metric)) for i in range(16)],
                audit=read(folder/'isolated.json'),
                artifacts={n:file_hash(folder/n) for n in ('stream.acsg','fresh/reconstruction.npz',
                           'fresh/decode.json','isolated_prediction.npz','isolated.json')})
            atomic_json(done,r)
        results.append(r)
    immutable_json(root/'composition/summary.json',dict(complete=True,results=results,
        role='noise-matched diagnostic; NOT independent test or trained routing'))


def main(args):
    if args.command == 'isolated':
        isolated(args.output)
        return
    run=Run(args);run.thread.start()
    try:
        with exclusive_native_evaluation(run):
            prepare(args.output)
            for job in read(args.output/'composition/jobs.json')['jobs']:
                folder=args.output/'composition'/job['sample_id']/job['name']
                output=folder/'fresh'
                output.mkdir(exist_ok=True)
                if (output/'decode.json').exists():
                    record=read(output/'decode.json')
                    assert record['stream_sha256']==job['sha256']
                    assert frame_hash(load_frames(output/'reconstruction.npz'))==record['output_hash']
                else:
                    execute(run,f'compose_{job["sample_id"]}_{job["name"]}', 'online_eg_decode.py',
                        ['--stream',folder/'stream.acsg','--output',output,
                         '--enhancement',ENHANCEMENT,'--adapter',ADAPTER],distributed=True)
            execute(run,'composition_isolated','four_state_composition.py',
                    ['isolated','--output',args.output],distributed=True)
            report(args.output)
            run.update(phase='complete')
    finally:
        run.log_resources();run.stop.set();run.thread.join(timeout=3)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['run','isolated'])
    p.add_argument('--output',type=Path,default=DEFAULT.with_name(DEFAULT.name+'_smoke'))
    p.add_argument('--max-hours',type=float,default=3.)
    main(p.parse_args())
