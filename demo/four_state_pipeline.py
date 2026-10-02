"""Resumable, exclusive GPU queue. No automatic large-model or Router training."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo.four_state_core import (DEFAULT, ENHANCEMENT, ADAPTER, crop, rows,
    protocol, immutable_json)
from demo.chunk_enhancement_experiment import Run, read
from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
from demo.conditioned_generation_pipeline import execute
from demo.scalable_codec import atomic_json, file_hash
from demo.patch_prefix_probe import load_frames
from demo.online_eg_eval_core import noise_pair


def replay(run):
    root = run.root
    paths = sorted((root/'received').rglob('*.json'))
    before = {str(p.relative_to(root)):file_hash(p) for p in paths}
    execute(run, 'readonly_replay', 'four_state_receive.py',
            ['--root', root, '--verify-only'])
    after = {str(p.relative_to(root)):file_hash(p) for p in sorted((root/'received').rglob('*.json'))}
    assert before == after, 'completed receiver replay changed data/timings'
    value = dict(complete=True, saved_json_unchanged=True, files=len(before),
                 inference_forbidden=True, metric_recalculation=False)
    immutable_json(root/'replay_audit.json', value)


def fresh_checks(run):
    root = run.root
    checks = []
    for row in rows(True):
        sid = row['sample_id']
        for i in (0, 5):  # clipped image boundary and full 64px halo
            folder = root/'received'/sid/f'cell_{i:02d}'
            saved = read(folder/'result.json')
            for state in ('G', 'EG'):
                out = root/'fresh_checks'/sid/f'cell_{i:02d}_{state}'
                done = out/'checked.json'
                if done.exists():
                    checked = read(done)
                    for name, digest in checked['artifacts'].items():
                        assert file_hash(out/name) == digest
                else:
                    out.mkdir(parents=True, exist_ok=True)
                    execute(run, f'fresh_{sid}_{i}_{state}', 'online_eg_decode.py',
                        ['--stream',folder/f'{state}.acsg','--output',out,
                         '--enhancement',ENHANCEMENT,'--adapter',ADAPTER], distributed=True)
                    pixels = load_frames(out/'reconstruction.npz')
                    report = read(out/'decode.json')
                    expected = saved['reports'][state]
                    assert not report['source_frames_read'] and report['outside_generate_exact']
                    assert report['output_hash'] == expected['output_hash']
                    assert report['generation_input_hash'] == expected['generation_input_hash']
                    assert report['total_bytes'] == expected['total_bytes']
                    with np.load(folder/'outputs.npz', allow_pickle=False) as cache:
                        np.testing.assert_array_equal(crop(pixels, i), cache[state])
                    noise_pair(report, dict(generation_runtime=expected['runtime']), same_condition=True)
                    checked = dict(complete=True, fresh_receiver_exact=True, cell=i, state=state,
                        artifacts={n:file_hash(out/n) for n in ('reconstruction.npz','decode.json')})
                    atomic_json(done, checked)
                checks.append(str(done))
    immutable_json(root/'fresh_audit.json', dict(complete=True, checks=checks,
                   count=8, fresh_receiver_exact=True, boundary_and_interior=True))


def main(args):
    run = Run(args)
    run.thread.start()
    try:
        with exclusive_native_evaluation(run):
            busy = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                                           '--format=csv,noheader,nounits'],text=True).strip()
            if busy:
                raise RuntimeError(f'GPU busy: {busy}')
            is_smoke = args.command == 'smoke'
            if args.command == 'verify':
                replay(run)
                return
            protocol(run.root, is_smoke)
            if not is_smoke:
                smoke = DEFAULT.with_name(DEFAULT.name+'_smoke')
                assert read(smoke/'fresh_audit.json')['fresh_receiver_exact']
                assert read(smoke/'replay_audit.json')['inference_forbidden']
                # Formal run uses exactly the smoke-tested code and weights.
                old, new = read(smoke/'protocol.json'), read(run.root/'protocol.json')
                for key in ('code','enhancement','adapter','profile'):
                    assert old[key] == new[key]
            execute(run, 'encode', 'four_state_prepare.py', ['--root',run.root])
            if is_smoke and not (run.root/'partial_resume.json').exists():
                execute(run,'receive_partial','four_state_receive.py',
                        ['--root',run.root,'--stop-after',2],distributed=True)
                files = {str(p.relative_to(run.root)):file_hash(p)
                         for p in (run.root/'received').rglob('*.json')}
                atomic_json(run.root/'partial_resume.json',dict(files=files))
            execute(run,'receive','four_state_receive.py',['--root',run.root],distributed=True)
            if is_smoke:
                for path, digest in read(run.root/'partial_resume.json')['files'].items():
                    assert file_hash(run.root/path) == digest, 'resume changed completed point'
                fresh_checks(run)
            replay(run)
            execute(run,'labels','four_state_labels.py',['--root',run.root])
            if not (run.root/'complete.json').exists():
                atomic_json(run.root/'complete.json',dict(complete=True, stage='four_state_teacher',
                    protocol=file_hash(run.root/'protocol.json'),
                    labels=file_hash(run.root/'labels.json'), elapsed_seconds=time.monotonic()-run.started,
                    no_router_training=True, no_model_updates=True))
            run.update(phase='complete')
    except BaseException as error:
        atomic_json(run.root/'last_failure.json',dict(error=repr(error),phase=run.progress))
        raise
    finally:
        run.log_resources(); run.stop.set(); run.thread.join(timeout=3)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command',choices=['smoke','run','verify'])
    p.add_argument('--output',type=Path)
    p.add_argument('--max-hours',type=float,default=24.)
    args = p.parse_args()
    if args.output is None:
        args.output = DEFAULT.with_name(DEFAULT.name+'_smoke') if args.command == 'smoke' else DEFAULT
    main(args)
