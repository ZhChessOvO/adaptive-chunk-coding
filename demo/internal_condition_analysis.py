"""CPU-only post-run analysis; preserve pinned receivers and original timings."""
import argparse
import json
import os
from pathlib import Path
from statistics import mean
import sys
import time
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from demo.chunk_enhancement_experiment import Run, read
from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
from demo import internal_condition_evaluate as evaluation
from demo import internal_condition_pipeline as pipeline
from demo.internal_condition_train import DEFAULT
from demo.internal_condition_report import grouped, CLIPS
from demo.patch_prefix_probe import load_frames
from demo.scalable_codec import atomic_json, file_hash
from demo.scalable_experiment import resources
from demo import scalable_cooperation_format as fmt


def forbidden(*args, **kwargs):
    raise AssertionError('Completed evaluation must not decode or recompute metrics')


def resume_without_recompute(root, run):
    """Exercise the real resume branch, fail on computation, keep formal summary."""
    target = root / 'evaluation/summary.json'
    original_hash = file_hash(target)
    original = read(target)
    captured = []

    def capture(path, value):
        if path == root / 'training_audit.json':
            assert value == read(path), 'Training audit changed'
            return
        assert path == target, f'Unexpected evaluation write: {path}'
        captured.append(value)

    with patch.object(pipeline, 'execute', forbidden), \
         patch.object(evaluation, 'quality', forbidden), \
         patch.object(evaluation, 'region_metrics', forbidden), \
         patch.object(evaluation, 'LPIPSAlex', return_value=object()), \
         patch.object(evaluation, 'atomic_json', capture):
        evaluation.evaluate(root, run)
    assert len(captured) == 1
    for key in ('results', 'training', 'protocol', 'paired_noise', 'no_E_exact',
                'repeat_exact', 'G_off_without_weights_exact'):
        assert captured[0][key] == original[key], key
    assert file_hash(target) == original_hash
    grouped(captured[0]['results'])
    return dict(points=58, no_decode=True, no_metric_recompute=True,
                formal_summary_unchanged=True, summary_sha256=original_hash)


def summarize(root):
    dest = root / 'evaluation'
    lookup = grouped(read(dest / 'summary.json')['results'])
    details = []
    for sid, name in CLIPS.items():
        point = dest / sid / 'internal_full'
        actual = load_frames(point / 'reconstruction.npz')
        control, *_ = fmt.parse((point / 'stream.acsg').read_bytes())
        mask = fmt.weights(actual.shape, control) > 0
        changes = {}
        for arm in ('off', 'input', 'zero', 'without', 'shuffled'):
            other = load_frames(dest / sid / f'{arm}_full/reconstruction.npz')
            delta = np.abs(actual.astype(np.int16) - other.astype(np.int16))[mask]
            changes[arm] = dict(channel_mae_255=float(delta.mean()),
                changed_fraction=float((delta != 0).mean()),
                local_lpips_gain=lookup[sid, 'full', arm]['roi_quality']['lpips_alex']
                    - lookup[sid, 'full', 'internal']['roi_quality']['lpips_alex'])
        details.append(dict(sample_id=sid, name=name, changes=changes,
            incremental_E={arm:lookup[sid, 'none', arm]['roi_quality']['lpips_alex']
                - lookup[sid, 'full', arm]['roi_quality']['lpips_alex']
                for arm in evaluation.ARMS}))
    ablations = {arm:{metric:mean(lookup[sid, 'full', arm]['roi_quality'][metric]
        for sid in CLIPS) for metric in ('lpips_alex', 'psnr_db', 'temporal_delta_mae')}
        for arm in ('internal', 'zero', 'without', 'shuffled')}
    beats = [json.loads(line) for line in (root / 'heartbeat.jsonl').read_text().splitlines()]
    formal = [row for row in beats if row['mode'] == 'run']
    return dict(clips=details, full_local_ablations=ablations,
        formal_seconds=read(root / 'run.complete.json')['elapsed_seconds'],
        formal_first_heartbeat=formal[0]['utc'], formal_last_heartbeat=formal[-1]['utc'],
        formal_sampled_gpu_peak_mib=max(int(row['gpu'].split(',')[2]) for row in formal),
        ordinary_file_bytes=sum(p.stat().st_size for p in root.rglob('*')
                                if p.is_file() and not p.is_symlink()),
        current_resources=resources())


def main(args):
    run = Run(args)
    run.thread.start()
    try:
        with exclusive_native_evaluation(run):
            for name, digest in read(args.output / 'queue_protocol.json')['code'].items():
                assert file_hash(REPO / 'demo' / name) == digest
            resume = resume_without_recompute(args.output, run)
            result = dict(complete=True, resume=resume, **summarize(args.output),
                analysis_code=file_hash(Path(__file__)), seconds=time.monotonic()-run.started)
            atomic_json(args.output / 'evaluation/analysis.json', result)
            atomic_json(args.output / 'analysis.complete.json',
                        dict(complete=True, seconds=result['seconds'], resume=resume))
            run.update(phase='complete', completed=58)
            print(json.dumps(result, indent=2), flush=True)
    finally:
        run.log_resources()
        run.stop.set()
        run.thread.join(timeout=3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--max-hours', type=float, default=1.)
    args = parser.parse_args()
    args.command = 'analysis'
    if not os.environ.get('TMUX'):
        parser.error('Run inside tmux')
    main(args)
