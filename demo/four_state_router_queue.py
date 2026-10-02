"""CPU-only continuation after the single-A800 teacher completes in tmux."""
import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from demo.four_state_core import DEFAULT, immutable_json
from demo.chunk_enhancement_experiment import Run, read
from demo.conditioned_generation_pipeline import execute
from demo.scalable_codec import atomic_json, file_hash

REPO=Path(__file__).resolve().parents[1]
FILES=('four_state_core.py','four_state_router.py','four_state_router_evaluate.py','four_state_router_queue.py',
       'four_state_report.py','run_four_state_router_queue.sh')


def main(args):
    run=Run(args);run.thread.start()
    try:
        code={p:file_hash(REPO/'demo'/p) for p in FILES}
        immutable_json(args.output/'queue_protocol.json',dict(code=code,teacher=str(args.teacher),
            epochs=240,cpu_only=True,scope='first conditional utility backbone plus equal-capacity context ablation'))
        smoke=DEFAULT.parent/'a800_four_state_router_20261002_smoke'
        assert read(smoke/'smoke.json')['exact_resume']
        assert read(smoke/'data.json')['dependencies']['code']==code['four_state_router.py']
        run.update(phase='waiting_for_teacher',cpu_only=True)
        while not (args.teacher/'complete.json').exists():
            run.check();time.sleep(5)
        done=read(args.teacher/'complete.json')
        assert done['complete'] and done['labels']==file_hash(args.teacher/'labels.json')
        assert read(args.teacher/'replay_audit.json')['inference_forbidden']
        for name,digest in code.items():assert file_hash(REPO/'demo'/name)==digest
        execute(run,'teacher_report','four_state_report.py',['--root',args.teacher])
        execute(run,'router_train','four_state_router.py',
                ['--root',args.teacher,'--output',args.output,'--epochs',240])
        execute(run,'router_table_eval','four_state_router_evaluate.py',['--output',args.output])
        if not (args.output/'complete.json').exists():
            atomic_json(args.output/'complete.json',dict(complete=True,
                table_evaluation=file_hash(args.output/'table_evaluation.json'),
                elapsed_including_wait_seconds=time.monotonic()-run.started,
                cpu_only=True,no_model_promotion=True,
                next='Inspect learned utility/allocation, then fresh mixed-route decoding and G boundary reduction'))
        run.update(phase='complete')
    except BaseException as error:
        atomic_json(args.output/'last_failure.json',dict(error=repr(error),phase=run.progress));raise
    finally:
        run.log_resources();run.stop.set();run.thread.join(timeout=3)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--teacher',type=Path,default=DEFAULT)
    p.add_argument('--output',type=Path,default=DEFAULT.parent/'a800_four_state_router_20261002')
    p.add_argument('--max-hours',type=float,default=24.)
    a=p.parse_args();a.command='router';main(a)
