"""Guarded context diagnosis queue. Training requires its separate entry."""
import argparse
import json
import os
from types import SimpleNamespace
from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
from demo.scalable_codec import atomic_json
from demo.scalable_experiment import now
from tools.texture_clean import CleanRun
from tools.texture_repair import execute
from tools.texture_context_probe import DEFAULT, verify


def main(args):
    if not os.environ.get('TMUX'):raise RuntimeError('tmux required')
    if args.command=='verify':print(json.dumps(verify(DEFAULT/'probe')),flush=True);return
    run=CleanRun(SimpleNamespace(output=DEFAULT,command='context_queue',max_hours=5))
    run.thread.start();run.log_resources()
    try:
        with exclusive_native_evaluation(run):
            execute(run,'frozen_context_probe','tools.texture_context_probe',['--output',DEFAULT/'probe'])
            verify(DEFAULT/'probe');run.update(phase='probe_complete_no_training')
    except BaseException as e:
        atomic_json(DEFAULT/'queue_failure.json',dict(error=repr(e),utc=now(),progress=run.progress));raise
    finally:run.log_resources();run.stop.set();run.thread.join();run.lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('probe','verify'))
    main(p.parse_args())
