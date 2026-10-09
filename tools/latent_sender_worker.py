"""Checkpointed new-latent sender labels, independent paired fit, and encoding."""
import argparse
from pathlib import Path
import os

import torch

from demo import routervc_sender_train as fit
from demo.routervc_fullview_probe import read,digest,save,immutable
from demo.chunk_enhancement_codec import configure_torch
from demo.chunk_enhancement_experiment import Run
from routervc.latent import sender_data as data


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('labels','fit','plan','verify'))
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stop-after',type=int,default=0)
    p.add_argument('--stop-renderings',type=int,default=0)
    p.add_argument('--sample-id')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--max-hours',type=float,default=48.)
    args=p.parse_args()
    if not os.environ.get('TMUX'):raise RuntimeError('tmux required')
    protocol=read(args.root/'protocol.json')
    for n,h in protocol['code'].items():
        if digest(data.REPO/n)!=h:raise ValueError('pinned sender source changed: '+n)
    configure_torch()
    run=Run(args);run.thread.start()
    samples=data.Samples(args.root,args.cache,protocol,run.check,run.update)
    samples.stop_after_renderings=args.stop_renderings
    try:
        if args.command=='labels':
            fit.prepare_labels(args.root,protocol,samples,run.check,run.update)
        elif args.command in ('fit','verify'):
            labels=fit.verify_labels(args.root,protocol,samples,run.check)
            if args.command=='verify':
                fit.verify_training(args.output,protocol,labels_binding=labels)
            else:
                scale_path=args.root/'train_scale.json'
                if scale_path.exists():scale=read(scale_path)
                else:
                    scale=fit.fit_scale(protocol,samples.get,run.check);immutable(scale_path,scale)
                fit.run_training(args.output,protocol,samples.get,scale_record=scale,labels_binding=labels,
                    check=run.check,progress=run.update,stop_after=args.stop_after)
        else:
            from routervc.latent.sender import write_prefixes
            from routervc.latent.router_data import source
            row=next(r for r in protocol['rows'] if r['sample_id']==args.sample_id)
            if digest(row['bank_path'])!=row['bank_sha256']:raise ValueError('bank changed')
            write_prefixes(args.output,source(row),Path(row['bank_path']).read_bytes(),args.checkpoint,
                           smoke=protocol['smoke'],check=run.check,progress=run.update)
    except InterruptedError as error:
        if args.stop_renderings and samples.measured>=args.stop_renderings:
            save(args.output/'intentional_stop.json',dict(complete=True,measured=samples.measured,error=str(error)))
        else:raise
    except BaseException as error:
        save(args.output/'last_failure.json',dict(error=repr(error),progress=run.progress));raise
    finally:
        samples.release();run.log_resources();run.stop.set();run.thread.join(timeout=3);run.lock.close()
    if torch.distributed.is_initialized():torch.distributed.destroy_process_group()


if __name__=='__main__':main()
