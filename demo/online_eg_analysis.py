"""CPU-only completed-queue replay: forbid all inference and metric recomputation."""
import argparse
from pathlib import Path
import os
import sys
import time
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from demo import online_eg_evaluate as evaluation
from demo.chunk_enhancement_experiment import Run, read
from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
from demo.scalable_codec import atomic_json, file_hash


def audit(root, run):
    assert read(root/'run.complete.json')['complete'], 'full evaluation/report must finish first'
    targets = [root/'train.complete.json', root/'training_audit.json',
               *sorted((root/'evaluation').rglob('*.json'))]
    before = {str(p.relative_to(root)):file_hash(p) for p in targets}
    def forbidden(*args, **kwargs):
        raise AssertionError('completed replay attempted inference or metric recomputation')
    begin = time.monotonic()
    with patch.object(evaluation, 'execute', forbidden), \
         patch.object(evaluation, 'LPIPSAlex', forbidden), \
         patch.object(evaluation, 'quality', forbidden), \
         patch.object(evaluation, 'region_metrics', forbidden):
        evaluation.evaluate(root, run)
    after = {str(p.relative_to(root)):file_hash(p) for p in targets}
    assert before == after, 'completed results or their original timings changed'
    result = dict(complete=True, points=92, no_inference=True, no_metrics_recomputed=True,
                  original_results_and_timings_unchanged=True,
                  checked_json_files=len(before), elapsed_seconds=time.monotonic()-begin,
                  audit_code_sha256=file_hash(Path(__file__)),
                  summary_sha256=before['evaluation/summary.json'])
    atomic_json(root/'evaluation.resume_audit.json',result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=evaluation.DEFAULT)
    p.add_argument('--max-hours',type=float,default=8.)
    args = p.parse_args()
    if not os.environ.get('TMUX'):
        p.error('Run inside tmux')
    args.command = 'resume-audit'
    run = Run(args)
    run.thread.start()
    try:
        with exclusive_native_evaluation(run):
            audit(args.output,run)
            run.update(phase='complete')
    finally:
        run.log_resources()
        run.stop.set()
        run.thread.join(timeout=3)
