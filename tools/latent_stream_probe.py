"""Stage B: actual rANS B/E bytes, literal prefixes and source-free fresh decode.

No G/Router training. Tests only a native I QP32 and one P8 QP48. This is not yet
a continuous low-rate B reference chain or regional enhancement experiment.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from demo.scalable_codec import atomic_bytes,atomic_json,atomic_npz,file_hash
from demo.scalable_format import frame_hash
from tools.latent_probe import SAMPLES,OLD_PROTOCOL,read,REPO
from routervc.latent import format as wire

DEFAULT=Path('/root/autodl-fs/DCVC/runs/routervc_latent_20261007/stage_b')
WIDTHS=(3,9,17)


def verify(folder):
    result=read(folder/'complete.json')
    for name,sha in result['artifacts'].items():
        if file_hash(folder/name)!=sha: raise ValueError(f'artifact changed: {folder/name}')
    return result


def child_decode(args):
    import torch
    from routervc.latent.codec import LatentCodec
    # Receiver has no original/cached-feature interface. Make accidental source
    # reads fail loudly, including an import accidentally growing such a read.
    def deny_source(event,values):
        if event=='open' and isinstance(values[0],(str,bytes)):
            name=os.fsdecode(values[0])
            own_output=Path(name).resolve().is_relative_to(args.output.resolve())
            if not own_output and Path(name).name in ('source.npz','pair.npz','pixels.npz','symbols.npz','protocol.json'):
                raise RuntimeError('source/cache access forbidden in fresh receiver')
    sys.addaudithook(deny_source)
    began=time.monotonic();data=args.stream.read_bytes()
    torch.cuda.reset_peak_memory_stats();codec=LatentCodec()
    base,out,detail=codec.decode(data,allow_incomplete_tail=args.allow_incomplete_tail)
    torch.cuda.synchronize()
    args.output.mkdir(parents=True,exist_ok=True)
    atomic_npz(args.output/'pixels.npz',base=base,reconstruction=out)
    detail.update(seconds=time.monotonic()-began,pid=os.getpid(),base_hash=frame_hash(base),
        output_hash=frame_hash(out),stream_sha256=file_hash(args.stream),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    atomic_json(args.output/'complete.json',dict(detail,artifacts={'pixels.npz':file_hash(args.output/'pixels.npz')}))


def fresh(run,stream,output,*,allow=False):
    if (output/'complete.json').exists():
        result=verify(output)
        if result['stream_sha256']!=file_hash(stream): raise ValueError('completed receiver input changed')
        return result
    output.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'-m','tools.latent_stream_probe','decode','--stream',str(stream),'--output',str(output)]
    if allow: cmd+=['--allow-incomplete-tail']
    with (output/'worker.log').open('a') as log:
        child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+600
            while child.poll() is None:
                run.check()
                if time.monotonic()>deadline: raise TimeoutError('fresh receiver timeout')
                time.sleep(.3)
            if child.returncode: raise RuntimeError(f'fresh receiver failed: {output}/worker.log')
        finally:
            if child.poll() is None:
                child.terminate()
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired: child.kill();child.wait()
    return verify(output)


def run_main(args):
    import torch
    from demo.chunk_enhancement_experiment import Run
    from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
    from demo.stage_c_three_path_roi_probe import LPIPSAlex,evaluate_variant
    from routervc.latent.codec import LatentCodec,profile_hash,MODEL_I,MODEL_P
    if not os.environ.get('TMUX'): raise RuntimeError('tmux required')
    inputs=read(OLD_PROTOCOL)['inputs'];names=SAMPLES[:args.limit]
    if args.limit not in range(1,5): raise ValueError('limit must be 1..4')
    protocol=dict(schema='RVL1-single-P8-stage-B',q_star=48,I_qp=32,widths=list(WIDTHS),
        sample_role='previously-used diagnostic/development',generation=False,router=False,training=False,
        inputs={s:{k:inputs[s][k] for k in ('source_path','source_sha256')} for s in names},
        profile=profile_hash(),driver_sha256=file_hash(Path(__file__)),
        model_i_sha256=file_hash(MODEL_I),model_p_sha256=file_hash(MODEL_P))
    args.output.mkdir(parents=True,exist_ok=True);pp=args.output/'protocol.json'
    if pp.exists():
        if read(pp)!=protocol: raise ValueError('changed stream experiment; use a distinct output')
    else:
        if args.command=='verify': raise RuntimeError('no completed experiment')
        atomic_json(pp,protocol)
    for item in protocol['inputs'].values():
        if file_hash(Path(item['source_path']))!=item['source_sha256']: raise ValueError('source changed')
    if args.command=='verify' or (args.output/'complete.json').exists():
        done=read(args.output/'complete.json')
        for sid in names:
            folder=args.output/sid
            verify(folder)
            for width in WIDTHS:
                for level in ('B','BE'): verify(folder/f'w{width}_{level}')
        for name,sha in done['artifacts'].items():
            if file_hash(args.output/name)!=sha: raise ValueError('aggregate artifact changed')
        print(json.dumps(dict(verified_samples=len(names),no_inference=True)),flush=True);return
    run=Run(args);run.thread.start();results=[]
    try:
        with exclusive_native_evaluation(run):
            metric=LPIPSAlex(True)
            for sid in names:
                run.check();folder=args.output/sid;folder.mkdir(parents=True,exist_ok=True)
                if (folder/'complete.json').exists():
                    results.append(verify(folder));continue
                run.update(sample=sid,completed=len(results),phase='native_encode_and_rans')
                with np.load(inputs[sid]['source_path'],allow_pickle=False) as f: source=f['source'][:9].copy()
                if (folder/'encoded.json').exists():
                    enc=read(folder/'encoded.json')
                    for n,sha in enc['artifacts'].items():
                        if file_hash(folder/n)!=sha: raise ValueError('encoding checkpoint changed')
                else:
                    began=time.monotonic();codec=LatentCodec();torch.cuda.reset_peak_memory_stats()
                    streams,native,pixels=codec.encode(source,48,WIDTHS)
                    atomic_bytes(folder/'native.bin',native)
                    atomic_npz(folder/'expected.npz',full=pixels)
                    files=['native.bin','expected.npz']
                    for width,(b,be) in streams.items():
                        for level,data in [('B',b),('BE',be)]:
                            filename=f'w{width}_{level}.rvl';atomic_bytes(folder/filename,data);files.append(filename)
                    enc=dict(seconds=time.monotonic()-began,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                             native_full_hash=frame_hash(pixels),artifacts={n:file_hash(folder/n) for n in files})
                    atomic_json(folder/'encoded.json',enc)
                    del codec
                    import gc;gc.collect();torch.cuda.empty_cache()
                with np.load(folder/'expected.npz',allow_pickle=False) as f: expected=f['full'].copy()
                rows=[]
                for width in WIDTHS:
                    b=(folder/f'w{width}_B.rvl').read_bytes();be=(folder/f'w{width}_BE.rvl').read_bytes()
                    if not be.startswith(b): raise RuntimeError('not a literal byte prefix')
                    for level in ('B','BE'):
                        run.update(phase='fresh_source_free_decode',sample=sid,width=width,level=level)
                        point=folder/f'w{width}_{level}'
                        detail=fresh(run,folder/f'w{width}_{level}.rvl',point)
                        with np.load(point/'pixels.npz',allow_pickle=False) as f:
                            out=f['reconstruction'].copy();bottom=f['base'].copy()
                        if level=='BE' and not np.array_equal(out,expected): raise RuntimeError('fresh full endpoint not native-exact')
                        if detail['base_bytes']+detail['enhancement_bytes']!=detail['actual_bytes']: raise RuntimeError('byte accounting mismatch')
                        rows.append(dict(sample=sid,level=level,**detail,
                            quality_P8=evaluate_variant(source[1:],out[1:],metric),
                            quality_all9=evaluate_variant(source,out,metric)))
                    pair=rows[-2:]
                    if pair[0]['base_hash']!=pair[1]['base_hash']: raise RuntimeError('E changed base pixels')
                # A fresh repeat and an incomplete final E explicitly falling back to B.
                checks={}
                if sid==names[0]:
                    full=folder/'w3_BE.rvl';data=full.read_bytes()
                    repeat=fresh(run,full,folder/'repeat_BE')
                    checks['repeat_exact']=repeat['output_hash']==rows[1]['output_hash']
                    atomic_bytes(folder/'incomplete_E.rvl',data[:-1])
                    fallback=fresh(run,folder/'incomplete_E.rvl',folder/'incomplete_fallback',allow=True)
                    checks['incomplete_E_falls_back_to_B']=fallback['output_hash']==rows[0]['output_hash']
                    if not all(checks.values()): raise RuntimeError('repeat/fallback checks failed')
                nbytes=(folder/'native.bin').stat().st_size
                native=dict(bytes=nbytes,bpp=nbytes*8/(source.shape[0]*source.shape[1]*source.shape[2]),
                    quality_P8=evaluate_variant(source[1:],expected[1:],metric),
                    quality_all9=evaluate_variant(source,expected,metric),scope='native I32+P48, same reference')
                files=['encoded.json']+list(enc['artifacts'])+[f'w{w}_{l}/complete.json' for w in WIDTHS for l in ('B','BE')]
                if sid==names[0]:
                    files+=['repeat_BE/complete.json','incomplete_fallback/complete.json','incomplete_E.rvl']
                result=dict(sample=sid,complete=True,rows=rows,native=native,checks=checks,
                    artifacts={n:file_hash(folder/n) for n in files})
                atomic_json(folder/'complete.json',result);results.append(result)
                print(json.dumps(dict(sample=sid,native_bpp=native['bpp'],rates=[(r['width'],r['level'],r['bpp']) for r in rows])),flush=True)
                run.update(completed=len(results),total=len(names))
        atomic_json(args.output/'summary.json',dict(results=results))
        atomic_json(args.output/'complete.json',dict(complete=True,samples=len(results),fresh_points=len(results)*6,
            extra_checks=2,seconds=time.monotonic()-run.started,artifacts={'summary.json':file_hash(args.output/'summary.json')}))
    finally:
        run.stop.set();run.thread.join(timeout=2);run.log_resources()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['run','verify','decode'])
    p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--stream',type=Path)
    p.add_argument('--limit',type=int,default=4)
    p.add_argument('--max-hours',type=float,default=4)
    p.add_argument('--allow-incomplete-tail',action='store_true')
    args=p.parse_args()
    if args.command=='decode': child_decode(args)
    else: run_main(args)


if __name__=='__main__': main()
