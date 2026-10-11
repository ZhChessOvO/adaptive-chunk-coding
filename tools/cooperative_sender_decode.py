"""P2b fresh receiver: stream + P2 R_g only; sender/source accesses forbidden."""
import argparse
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
from demo.chunk_enhancement_codec import configure_torch
from demo.chunk_enhancement_experiment import Run
from demo.routervc_fullview_probe import read, save, digest, verify_artifacts
from demo.scalable_codec import atomic_npz
from tools.latent_router_worker import receiver_guard


def deny_sender_checkpoint(event, values):
    if event == 'open' and isinstance(values[0], (str, bytes)):
        path = Path(os.fsdecode(values[0])).resolve()
        if path.suffix in ('.pt', '.pth') and '/runs/' in str(path) and any('sender' in part for part in path.parts):
            raise RuntimeError('fresh receiver attempted sender-checkpoint access: '+str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('decode',))
    parser.add_argument('--stream', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--receiver', type=Path)
    parser.add_argument('--disable-generation', action='store_true')
    args = parser.parse_args()
    if not os.environ.get('TMUX'): raise RuntimeError('tmux required')
    configure_torch(); receiver_guard(args.output, args.stream); sys.addaudithook(deny_sender_checkpoint)
    run = Run(SimpleNamespace(output=args.output, command='decode', max_hours=12)); run.thread.start()
    try:
        if (args.output/'complete.json').exists():
            done = read(args.output/'complete.json'); verify_artifacts(args.output, done['artifacts'])
            if done['stream'] != digest(args.stream): raise ValueError('completed stream changed')
        else:
            from routervc.cooperation.receive import decode
            base, received, output, detail = decode(args.stream.read_bytes(), args.receiver,
                                                    args.disable_generation, run.check)
            atomic_npz(args.output/'pixels.npz', base=base, enhanced=received, reconstruction=output)
            detail.update(stream=digest(args.stream), sender_checkpoint_access_guard=True,
                          artifacts={'pixels.npz':digest(args.output/'pixels.npz')})
            save(args.output/'complete.json', detail)
    except BaseException as error:
        save(args.output/'last_failure.json', dict(error=repr(error), progress=run.progress)); raise
    finally:
        run.log_resources(); run.stop.set(); run.thread.join(timeout=3); run.lock.close()
    if torch.distributed.is_initialized(): torch.distributed.destroy_process_group()


if __name__ == '__main__': main()
