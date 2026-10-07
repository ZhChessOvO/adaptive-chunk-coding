"""Stage C: real continuous B-only reference, missing/reordered/repeated E.

17-frame REDS/UVG and a 12-frame REDS partial tail. Arrival tests re-decode the
received file in fresh processes; NOT an optimized persistent online decoder.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from demo.scalable_codec import atomic_bytes,atomic_npz,atomic_json,file_hash
from demo.scalable_format import frame_hash
from tools.latent_probe import read,SAMPLES,OLD_PROTOCOL
from tools.latent_stream_probe import verify

DEFAULT=Path('/root/autodl-fs/DCVC/runs/routervc_latent_20261007/stage_c')
CASES=((SAMPLES[0],17),(SAMPLES[2],17),(SAMPLES[0],12))


def decode_worker(args):
    import torch
    from routervc.latent.chain_codec import ChainCodec
    def audit(event,values):
        if event=='open' and isinstance(values[0],(str,bytes)):
            p=Path(os.fsdecode(values[0])).resolve()
            if not p.is_relative_to(args.output.resolve()) and p.name in ('source.npz','pair.npz','expected.npz','pixels.npz','symbols.npz','protocol.json'):
                raise RuntimeError('source/cache not allowed in chain receiver')
    sys.addaudithook(audit)
    started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    codec=ChainCodec();base,out,detail=codec.decode_chain(args.stream.read_bytes(),allow_incomplete_tail=args.allow_incomplete_tail)
    torch.cuda.synchronize();args.output.mkdir(parents=True,exist_ok=True)
    atomic_npz(args.output/'pixels.npz',base=base,reconstruction=out)
    detail.update(pid=os.getpid(),seconds=time.monotonic()-started,base_hash=frame_hash(base),
        output_hash=frame_hash(out),stream_sha256=file_hash(args.stream),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    atomic_json(args.output/'complete.json',dict(detail,artifacts={'pixels.npz':file_hash(args.output/'pixels.npz')}))


def fresh(run,stream,out,allow=False):
    if (out/'complete.json').exists():
        result=verify(out)
        if result['stream_sha256']!=file_hash(stream): raise ValueError('changed chain worker stream')
        return result
    out.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'-m','tools.latent_chain_probe','decode','--stream',str(stream),'--output',str(out)]
    if allow: cmd+=['--allow-incomplete-tail']
    with (out/'worker.log').open('a') as log:
        child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
        try:
            start=time.monotonic()
            while child.poll() is None:
                run.check()
                if time.monotonic()-start>600: raise TimeoutError('chain fresh timeout')
                time.sleep(.3)
            if child.returncode: raise RuntimeError(f'chain decoder failed: {out}/worker.log')
        finally:
            if child.poll() is None:
                child.terminate()
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired: child.kill();child.wait()
    return verify(out)


def execute(args):
    import torch
    from routervc.latent.chain_codec import ChainCodec,chain_hash
    from demo.chunk_enhancement_experiment import Run
    from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
    from demo.stage_c_three_path_roi_probe import LPIPSAlex,evaluate_variant
    if not os.environ.get('TMUX'): raise RuntimeError('tmux required')
    if args.limit not in (1,2,3): raise ValueError('case limit must be 1..3')
    cases=CASES[:args.limit];sources=read(OLD_PROTOCOL)['inputs']
    protocol=dict(profile='RVLC1-stage-C',cases=list(map(list,cases)),profile_hash=chain_hash(),
        driver_sha256=file_hash(Path(__file__)),generation=False,router=False,training=False,
        inputs={sid:{k:sources[sid][k] for k in ('source_path','source_sha256')} for sid,_ in cases},
        arrival_scope='fresh re-decode for received subsets/order; not persistent incremental execution')
    args.output.mkdir(parents=True,exist_ok=True);path=args.output/'protocol.json'
    if path.exists():
        if read(path)!=protocol: raise ValueError('changed chain protocol; use distinct output')
    else:
        if args.command=='verify': raise RuntimeError('missing chain experiment')
        atomic_json(path,protocol)
    for item in protocol['inputs'].values():
        if file_hash(Path(item['source_path']))!=item['source_sha256']: raise ValueError('changed source')
    if args.command=='verify' or (args.output/'complete.json').exists():
        done=verify(args.output)
        for sid,count in cases:
            folder=args.output/f'{sid}_n{count}';result=verify(folder)
            for name in result['arrivals']: verify(folder/name)
        print(json.dumps(dict(verified_cases=len(cases),no_inference=True)),flush=True);return
    run=Run(args);run.thread.start();results=[]
    try:
        with exclusive_native_evaluation(run):
            metric=LPIPSAlex(True)
            for sid,count in cases:
                run.check();folder=args.output/f'{sid}_n{count}';folder.mkdir(parents=True,exist_ok=True)
                if (folder/'complete.json').exists(): results.append(verify(folder));continue
                run.update(phase='native_continuous_encode',sample=sid,count=count,completed=len(results))
                with np.load(sources[sid]['source_path'],allow_pickle=False) as f: source=f['source'][:count].copy()
                if not (folder/'encoded.json').exists():
                    start=time.monotonic();codec=ChainCodec()
                    b,packets,base,full,native,own,details=codec.encode_chain(source)
                    if len(packets)!=2: raise ValueError('these diagnostics require two P chunks')
                    streams={'B':b,'first_E':b+packets[0],'second_E':b+packets[1],
                        'all_E':b+packets[0]+packets[1],'reverse_E':b+packets[1]+packets[0],
                        'repeat_E':b+packets[0]+packets[1]+packets[0],
                        'incomplete_E':b+packets[0]+packets[1][:-1]}
                    for key,data in streams.items(): atomic_bytes(folder/f'{key}.rvlc',data)
                    atomic_bytes(folder/'native_own_reference.bin',native)
                    atomic_npz(folder/'expected.npz',base=base,full=full,native_own=own)
                    names=[f'{key}.rvlc' for key in streams]+['native_own_reference.bin','expected.npz']
                    atomic_json(folder/'encoded.json',dict(details,seconds=time.monotonic()-start,
                        artifacts={n:file_hash(folder/n) for n in names}))
                    del codec
                    import gc;gc.collect();torch.cuda.empty_cache()
                enc=read(folder/'encoded.json')
                for n,h in enc['artifacts'].items():
                    if file_hash(folder/n)!=h: raise ValueError('changed chain encode checkpoint')
                with np.load(folder/'expected.npz',allow_pickle=False) as f:
                    expected_base=f['base'].copy();expected_full=f['full'].copy();own=f['native_own'].copy()
                arrivals={}
                for name in ('B','first_E','second_E','all_E','reverse_E','repeat_E','incomplete_E'):
                    run.update(phase='fresh_reference_check',arrival=name)
                    result=fresh(run,folder/f'{name}.rvlc',folder/name,name=='incomplete_E')
                    with np.load(folder/name/'pixels.npz',allow_pickle=False) as f: base=f['base'].copy();out=f['reconstruction'].copy()
                    if not np.array_equal(base,expected_base) or result['base_reference_hashes']!=enc['base_reference_hashes']:
                        raise RuntimeError('E arrival changed B pixels/temporal memory')
                    expected=expected_base.copy()
                    for index in result['received_chunks']:
                        start=1+8*index;end=min(start+8,count);expected[start:end]=expected_full[start:end]
                    if not np.array_equal(out,expected): raise RuntimeError('E arrival decoded wrong display pixels')
                    arrivals[name]=dict(result,quality=evaluate_variant(source,out,metric))
                if arrivals['all_E']['output_hash']!=arrivals['reverse_E']['output_hash'] or arrivals['all_E']['output_hash']!=arrivals['repeat_E']['output_hash']:
                    raise RuntimeError('reorder/repeat mismatch')
                if arrivals['first_E']['output_hash']!=arrivals['incomplete_E']['output_hash']: raise RuntimeError('incomplete E fallback mismatch')
                nbytes=(folder/'native_own_reference.bin').stat().st_size
                native=dict(bytes=nbytes,bpp=nbytes*8/(count*source.shape[1]*source.shape[2]),quality=evaluate_variant(source,own,metric),
                            scope='native I32/P48 own full-reference chain; different from same-context endpoint')
                files=list(enc['artifacts'])+['encoded.json']+[f'{n}/complete.json' for n in arrivals]
                result=dict(complete=True,sample=sid,frames=count,arrivals=arrivals,native_own=native,
                            artifacts={n:file_hash(folder/n) for n in files})
                atomic_json(folder/'complete.json',result);results.append(result)
                print(json.dumps(dict(sample=sid,frames=count,all_references_exact=True,
                    B=arrivals['B']['quality'],BE=arrivals['all_E']['quality'],native_own=native)),flush=True)
                run.update(completed=len(results),total=len(cases))
        atomic_json(args.output/'summary.json',dict(results=results))
        atomic_json(args.output/'complete.json',dict(complete=True,cases=len(results),fresh_decodes=7*len(results),
            seconds=time.monotonic()-run.started,artifacts={'summary.json':file_hash(args.output/'summary.json')}))
    finally:
        run.stop.set();run.thread.join(timeout=2);run.log_resources()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['run','verify','decode'])
    p.add_argument('--output',type=Path,default=DEFAULT);p.add_argument('--stream',type=Path)
    p.add_argument('--limit',type=int,default=3);p.add_argument('--max-hours',type=float,default=4)
    p.add_argument('--allow-incomplete-tail',action='store_true');args=p.parse_args()
    if args.command=='decode': decode_worker(args)
    else: execute(args)


if __name__=='__main__': main()
