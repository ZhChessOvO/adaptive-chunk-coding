"""Bounded pair-schedule comparison on the same 13 measured q1 E50/G8 streams.

Only the shared policy fingerprint changes; UF/E bytes, selected G support,
Router/E/G weights, and original feather weights remain unchanged. Fresh runs
and two repeated decodes assess the new profile, never overwrite old outputs.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import os
from pathlib import Path
import sys
from types import SimpleNamespace

REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from demo.routervc_fullview_probe import read,digest,save,immutable,verify_artifacts
from demo.routervc_qstep_probe import ORIGINAL,make_protocol,validate_decode

OUTPUT=Path('/root/autodl-fs/DCVC/runs/routervc_efficiency_20261004/G_pair_probe')
REFERENCE='global_local_e0.5_g8'


def peak_G(report):
    return max(w['runtime']['peak_cuda_allocated_bytes'] for w in report['generation_runtime']['windows'])


def protocol_for(root):
    from demo.routervc_scheduled_decode import policy_identity
    old=make_protocol(root)
    return dict(schema='routervc-pair-schedule-probe-v1',original=old['original'],
        source_artifacts=old['source_artifacts'],source_root=old['source_root'],
        code={n:digest(REPO/'demo'/n) for n in ('routervc_schedule_probe.py',
              'routervc_scheduled_decode.py','routervc_generation_schedule.py',
              'routervc_qstep_probe.py','run_routervc_schedule_probe.sh')},
        new_policy=policy_identity(),reference=REFERENCE,original_outputs_unchanged=True,
        new_weights=False,new_mask_bytes=0,source_scope=old['original']['source_scope'],
        sample_count=13,new_fresh_points=13,extra_repeats=2,area_cap=1.5,max_merged_cells=2)


def run_points(run,args,protocol):
    import numpy as np
    import torch
    from PIL import Image
    from demo.routervc_scheduled_decode import rewrap,parse
    from demo.routervc_visual_format import parse as old_parse
    from demo.scalable_codec import atomic_bytes
    from demo.scalable_experiment import quality
    from demo.stage_c_three_path_roi_probe import LPIPSAlex
    from demo.conditioned_generation_pipeline import execute
    from demo.routervc_visual_evaluate import source_shape
    import math
    done=(run.root/'complete.json').exists()
    if done:verify_artifacts(run.root,read(run.root/'complete.json')['artifacts'])
    metric=None;records=[];repeated_datasets=set()
    for entry in protocol['original']['sources']:
        sample=entry['sample'];sid=sample['sample_id'];run.check()
        original=args.root/'samples'/sid/REFERENCE
        folder=run.root/'samples'/sid
        if not done:folder.mkdir(parents=True,exist_ok=True)
        old_wire=(original/'stream.rtvc').read_bytes();wire=rewrap(old_wire)
        if (folder/'stream.rtvc').exists():
            if (folder/'stream.rtvc').read_bytes()!=wire:raise ValueError('saved schedule stream changed')
        else:
            if done:raise ValueError('completed scheduled stream missing')
            atomic_bytes(folder/'stream.rtvc',wire)
            # Expected base, E image and Router decisions come from the original
            # fresh run. This is sender audit metadata, never sent to receiver.
            save(folder/'encode.json',read(original/'encode.json'))
        if parse(wire)[1]!=old_parse(old_wire)[1]:raise ValueError('scheduler changed inner UF/E bytes')
        if (folder/'result.json').exists():
            result=read(folder/'result.json')
            if result['protocol_sha256']!=digest(run.root/'protocol.json'):raise ValueError('schedule result binding changed')
            verify_artifacts(folder,result['artifacts'])
            if validate_decode(folder,protocol,sample)!=result['decode']:raise ValueError('saved scheduled decode changed')
        else:
            if done:raise ValueError('completed replay may not infer/score')
            def fresh(destination,label):
                destination.mkdir(exist_ok=True)
                execute(run,f'{sid}_{label}','routervc_scheduled_decode.py',
                    ['--stream',folder/'stream.rtvc','--output',destination,
                     '--router',protocol['original']['models']['arms']['global_local']['path'],
                     '--enhancement',protocol['original']['enhancement']['path'],
                     '--adapter',protocol['original']['adapter']['path']],distributed=True)
            if not (folder/'fresh/decode.json').exists():fresh(folder/'fresh','fresh')
            decoded=validate_decode(folder,protocol,sample)
            old=read(original/'fresh/decode.json')
            if (decoded['route']!=old['route'] or decoded['total_bytes']!=old['total_bytes']
                    or decoded['generation_input_hash']!=old['generation_input_hash']):
                raise ValueError('schedule changed allocation, E pixels, or charged bytes')
            expected_calls=decoded['generation_runtime']['schedule']['merged_calls']
            if len(decoded['generation_runtime']['windows'])!=expected_calls:
                raise ValueError('17-frame window call count differs from schedule')
            names=['stream.rtvc','encode.json','fresh/decode.json','fresh/reconstruction.npz','fixed_frame.png']
            repeated=False
            if sample['dataset'] not in repeated_datasets:
                if not (folder/'repeat/decode.json').exists():fresh(folder/'repeat','repeat')
                again=read(folder/'repeat/decode.json')
                from demo.scalable_format import frame_hash
                with np.load(folder/'repeat/reconstruction.npz',allow_pickle=False) as repeat:
                    if frame_hash(repeat['reconstruction'])!=decoded['output_hash']:
                        raise ValueError('repeated scheduled pixels differ')
                if again['output_hash']!=decoded['output_hash'] or again['route']!=decoded['route']:
                    raise ValueError('repeated schedule differs')
                names+=['repeat/decode.json','repeat/reconstruction.npz'];repeated=True
            run.update(phase='CPU_schedule_metrics',sample=sid)
            if metric is None:
                torch.set_num_threads(4);metric=LPIPSAlex(True)
            with np.load(args.root/'samples'/sid/'source.npz',allow_pickle=False) as data:source=data['source'].copy()
            with np.load(folder/'fresh/reconstruction.npz',allow_pickle=False) as data:output=data['reconstruction'].copy()
            score=quality(source,output,metric)
            image=folder/'fixed_frame.png';Image.fromarray(output[8]).save(image.with_suffix('.tmp'),format='PNG')
            os.replace(image.with_suffix('.tmp'),image)
            old_result=read(original/'result.json')
            result=dict(complete=True,sample_id=sid,dataset=sample['dataset'],
                protocol_sha256=digest(run.root/'protocol.json'),quality=score,
                reference_quality=old_result['quality'],repeat_exact=repeated,
                bpp=8*len(wire)/math.prod(source_shape(sample)[:3]),bytes=len(wire),
                decode=decoded,original_receiver_seconds=old['seconds'],
                original_G_seconds=old['generation_runtime']['seconds_model_load_excluded'],
                original_G_all_calls_peak_bytes=peak_G(old),
                new_G_all_calls_peak_bytes=peak_G(decoded),
                original_last_call_field_bytes=old['peak_cuda_allocated_bytes'],
                artifacts={n:digest(folder/n) for n in names})
            save(folder/'result.json',result)
        repeated_datasets.add(sample['dataset']);records.append(result)
        run.update(completed=len(records),total=13)
    if not done:
        verify_artifacts(args.root,protocol['source_artifacts'])
        grouped={}
        for dataset in ('REDS','UVG'):
            rows=[r for r in records if r['dataset']==dataset];n=len(rows)
            grouped[dataset]=dict(windows=n,
                quality={k:sum(r['quality'][k] for r in rows)/n for k in rows[0]['quality']},
                quality_delta={k:sum(r['quality'][k]-r['reference_quality'][k] for r in rows)/n for k in rows[0]['quality']},
                reference_G_seconds=sum(r['original_G_seconds'] for r in rows)/n,
                new_G_seconds=sum(r['decode']['generation_runtime']['seconds_model_load_excluded'] for r in rows)/n,
                reference_G_peak_GiB=sum(r['original_G_all_calls_peak_bytes'] for r in rows)/n/2**30,
                new_G_peak_GiB=sum(r['new_G_all_calls_peak_bytes'] for r in rows)/n/2**30,
                new_whole_worker_peak_GiB=sum(r['decode']['peak_cuda_allocated_bytes'] for r in rows)/n/2**30,
                original_receiver_seconds=sum(r['original_receiver_seconds'] for r in rows)/n,
                new_receiver_seconds=sum(r['decode']['seconds'] for r in rows)/n)
        save(run.root/'summary.json',dict(complete=True,windows=13,extra_repeat_decodes=2,
            group_means=grouped,bytes_and_E_inputs_unchanged=True,old_pixels_not_expected_equal=True,
            memory_scope='same-scope comparison uses max of ALL G calls; new whole-worker accumulator reported separately',
            timing_scope='fresh runs at different times; observed timings, not a controlled persistent benchmark'))
        names=['protocol.json','summary.json']+[f"samples/{r['sample_id']}/result.json" for r in records]
        save(run.root/'complete.json',dict(complete=True,artifacts={n:digest(run.root/n) for n in names}))
    print('G_SCHEDULE_VERIFIED_READ_ONLY' if done else 'G_SCHEDULE_COMPLETE',flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ORIGINAL);p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--max-hours',type=float,default=8.)
    args=p.parse_args(argv)
    if not os.environ.get('TMUX'):raise RuntimeError('scheduled probe requires tmux')
    if args.output.resolve().is_relative_to(args.root.resolve()):raise ValueError('output must be separate')
    from demo.chunk_enhancement_experiment import Run
    from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
    run=Run(SimpleNamespace(output=args.output,command='G_pair_probe',max_hours=args.max_hours));run.thread.start()
    previous=os.environ.get('ROUTERVC_SCHEDULE_PARENT')
    try:
        protocol=protocol_for(args.root);immutable(run.root/'protocol.json',protocol)
        os.environ['ROUTERVC_SCHEDULE_PARENT']=str(os.getpid())
        with nullcontext() if (run.root/'complete.json').exists() else exclusive_native_evaluation(run):
            run_points(run,args,protocol)
    except BaseException as error:
        save(run.root/'last_failure.json',dict(error=repr(error),progress=run.progress));raise
    finally:
        if previous is None:os.environ.pop('ROUTERVC_SCHEDULE_PARENT',None)
        else:os.environ['ROUTERVC_SCHEDULE_PARENT']=previous
        run.log_resources();run.stop.set();run.thread.join(timeout=3);run.lock.close()


if __name__=='__main__':main()
