"""Fresh source/sender-free worker for the explicitly bound G comparison."""
import argparse
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
from demo.chunk_enhancement_codec import configure_torch
from demo.routervc_fullview_probe import read, digest, verify_artifacts
from demo.scalable_codec import atomic_json, atomic_npz
from tools.latent_router_worker import receiver_guard
from tools.cooperative_sender_decode import deny_sender_checkpoint
from tools.texture_repair import RepairRun


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('stream','output','receiver','adapter'): p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--disable-generation',action='store_true')
    p.add_argument('--stop-regions',type=int,default=0)
    args=p.parse_args()
    if not os.environ.get('TMUX'): raise RuntimeError('tmux required')
    configure_torch(); receiver_guard(args.output,args.stream); sys.addaudithook(deny_sender_checkpoint)
    run=RepairRun(SimpleNamespace(output=args.output,command='decode',max_hours=2))
    run.thread.start(); run.log_resources()
    try:
        from routervc.generation.integration import decode
        if (args.output/'complete.json').exists():
            done=read(args.output/'complete.json'); verify_artifacts(args.output,done['artifacts'])
            if done['stream']!=digest(args.stream): raise ValueError('completed input changed')
        else:
            result=decode(args.stream.read_bytes(),args.receiver,args.adapter,args.output,
                args.disable_generation,args.stop_regions,run.check)
            if result is not None:
                base,received,output,detail=result
                atomic_npz(args.output/'pixels.npz',base=base,enhanced=received,reconstruction=output)
                names=['pixels.npz','binding.json']+[str(f.relative_to(args.output)) for f in
                    args.output.glob('g*/*') if f.name in ('raw.npz','complete.json','request.json')]
                detail.update(stream=digest(args.stream),sender_checkpoint_access_guard=True,
                    artifacts={n:digest(args.output/n) for n in names})
                atomic_json(args.output/'complete.json',detail)
    except BaseException as error:
        atomic_json(args.output/'last_failure.json',dict(error=repr(error),progress=run.progress)); raise
    finally:
        run.log_resources(); run.stop.set(); run.thread.join(timeout=3); run.lock.close()
    if torch.distributed.is_initialized(): torch.distributed.destroy_process_group()


if __name__=='__main__': main()
