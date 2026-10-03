"""CPU-only RouterVC completion queue: audit, saved-result analysis, previews.

Waits for main, supplementary and resilience completion. This coordinator and
its children NEVER take the GPU mutex or run a model/quality metric. Its Run
progress lives under workflow/, separate from every experiment's progress.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from demo.scalable_codec import atomic_json, file_hash

RUNS = Path('/root/autodl-fs/DCVC/runs')
DEFAULT = RUNS/'routervc_20261003'
RESILIENCE = RUNS/'routervc_resilience_20261003'
CODE = ('routervc_finalize.py', 'run_routervc_finalize.sh', 'routervc_audit.py',
        'routervc_analysis.py', 'run_routervc_analysis.sh', 'routervc_preview.py')


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, text):
    if not condition:
        raise RuntimeError(text)


def immutable(path, data):
    if path.exists():
        require(read(path) == data, f'workflow configuration changed: {path}')
    else:
        atomic_json(path, data)


def marker_paths(root, supplement, resilience):
    return {'main': root/'complete.json', 'supplement': supplement/'complete.json',
            'resilience': resilience/'complete.json'}


def wait_for_completions(run, markers, interval=10., sleep=time.sleep):
    if not 0 < interval <= 15:
        raise ValueError('completion polling interval must be within (0, 15] seconds')
    began = time.monotonic()
    while True:
        run.check()
        missing = [name for name, path in markers.items()
                   if not path.is_file() or read(path).get('complete') is not True]
        if not missing:
            return time.monotonic()-began
        run.update(phase='waiting_for_completed_results', missing=missing, cpu_only=True)
        sleep(interval)


def checked(path, expected=None, hashes=None):
    path = Path(path).resolve()
    value = file_hash(path)
    if expected is not None:
        require(value == expected, f'workflow input/output artifact changed: {path}')
    if hashes is not None:
        previous = hashes.get(str(path))
        require(previous is None or previous == value, f'inconsistent artifact identity: {path}')
        hashes[str(path)] = value
    return value


def completion_evidence(root, supplement, resilience):
    """Verify marker bindings and previously executed source-free resilience."""
    hashes = {}
    for path in marker_paths(root, supplement, resilience).values():
        require(read(path).get('complete') is True, f'incomplete dependency: {path}')
        checked(path, hashes=hashes)
    main = read(root/'complete.json')
    checked(root/'summary.json', main['summary'], hashes)
    checked(root/'protocol.json', main['protocol'], hashes)
    supplement_done = read(supplement/'complete.json')
    require(supplement_done.get('readonly_completed_point_replay') is True,
            'baseline supplement has not completed its own no-inference replay')
    checked(supplement/'summary.json', supplement_done['summary'], hashes)
    supplement_summary = read(supplement/'summary.json')
    checked(supplement/'protocol.json', supplement_summary['baseline_protocol'], hashes)
    require(supplement_summary['main_summary'] == main['summary'], 'supplement references a different main summary')
    evidence = read(resilience/'complete.json')
    require(evidence.get('truncated_pixels_and_route_exact') is True
            and evidence.get('missing_E_G_router_fallback_exact') is True
            and evidence.get('strict_truncation_and_CRC_rejected_before_GPU') is True
            and evidence.get('source_frames_read') is False, 'resilience invariants not established')
    expected_cases = {'first_packet', 'truncated_next_packet', 'base_no_models'}
    require(evidence.get('fresh_decodes') == 3 and set(evidence.get('records', {})) == expected_cases,
            'resilience must contain exactly the three executed fresh receiver cases')
    checked(resilience/'protocol.json', evidence['protocol'], hashes)
    rejection_path = resilience/'cpu_rejections.json'
    rejections = read(rejection_path)
    require(rejections.get('complete') is True and rejections.get('gpu_started') is False,
            'CPU rejection evidence is incomplete or ran a GPU')
    require(all(isinstance(rejections.get(key), str) and rejections[key] for key in
                ('strict_truncated', 'crc_strict', 'crc_permissive')),
            'missing strict truncation / strict and permissive CRC rejection evidence')
    expected_rejections = read(resilience/'protocol.json')['details']['rejection']
    require({key: rejections[key] for key in ('strict_truncated', 'crc_strict', 'crc_permissive')}
            == expected_rejections, 'CPU rejections differ from the pinned resilience protocol')
    require(set(rejections.get('artifacts', {})) == {'truncated_next_packet.rtvc', 'crc_rejected_cpu.rtvc'},
            'CPU rejection artifacts are incomplete')
    checked(rejection_path, hashes=hashes)
    for filename, digest in rejections['artifacts'].items():
        checked(resilience/filename, digest, hashes)
    for name, record in evidence['records'].items():
        require(record.get('complete') is True, f'incomplete resilience case: {name}')
        folder = resilience/name
        require(read(folder/'complete.json') == record, f'resilience case summary changed: {name}')
        checked(folder/'complete.json', hashes=hashes)
        for filename, digest in record['artifacts'].items():
            checked(folder/filename, digest, hashes)
        require(read(folder/'decode.json') == record['decoded'], f'resilience receiver evidence changed: {name}')
        checked(resilience/f'{name}.rtvc', record['decoded']['stream_sha256'], hashes)
    return hashes


def child_commands(root, supplement):
    return [
        ('audit', [sys.executable, str(REPO/'demo/routervc_audit.py'), '--output', str(root)], root/'audit.json'),
        ('analysis', ['bash', str(REPO/'demo/run_routervc_analysis.sh'), '--root', str(supplement)],
         supplement/'analysis/summary.json'),
        ('preview', [sys.executable, str(REPO/'demo/routervc_preview.py'), '--root', str(supplement),
          '--profile', 'formal', '--repeats', '1', '--output', str(supplement/'preview_single')],
         supplement/'preview_single/manifest.json'),
    ]


def verify_code(config):
    for name, digest in config['code'].items():
        checked(REPO/'demo'/name, digest)


def run_child(run, name, command):
    """Fresh CPU subprocess; no nested GPU locks and bounded stop handling."""
    run.update(phase=f'CPU_{name}', cpu_only=True)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    with (run.root/f'{name}.log').open('a') as log:
        log.write(json.dumps(dict(command=command, attempt_started=time.time(), cpu_only=True))+'\n')
        log.flush()
        child = subprocess.Popen(command, cwd=REPO, env=environment,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while child.poll() is None:
                run.check()
                time.sleep(2)
            if child.returncode:
                raise RuntimeError(f'CPU {name} exited {child.returncode}; see {run.root/name}.log')
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=10)
            raise


def output_evidence(root, supplement, run=None):
    hashes = {}
    audit_path = root/'audit.json'
    audit = read(audit_path)
    require(audit.get('complete') is True and all(audit.get(k) is True for k in
            ('no_decoder', 'no_router_inference', 'no_generator', 'no_metric_recalculation')),
            'main CPU audit did not establish the required scope')
    checked(audit_path, hashes=hashes)
    for path, digest in audit['file_sha256'].items():
        if run is not None:
            run.check()
        checked(Path(path), digest, hashes)
    for folder, filename, input_key in ((supplement/'analysis', 'summary.json', 'input_hashes'),
                                        (supplement/'preview_single', 'manifest.json', None)):
        path = folder/filename
        result = read(path)
        require(result.get('complete') is True and result.get('no_metric_recalculation') is True,
                f'incomplete or scope-invalid CPU output: {path}')
        require(result.get('no_inference', result.get('no_model_inference')) is True,
                f'CPU output did not declare inference-free processing: {path}')
        checked(path, hashes=hashes)
        inputs = result[input_key] if input_key else result['binding']['input_hashes']
        for name, digest in inputs.items():
            if run is not None:
                run.check()
            checked(Path(name), digest, hashes)
        for name, digest in result['artifacts'].items():
            checked(folder/name, digest, hashes)
    return hashes


def finalize(run, root, supplement, resilience):
    began = time.monotonic()
    config = dict(version=1, root=str(root.resolve()), supplement=str(supplement.resolve()),
        resilience=str(resilience.resolve()), output=str(run.root.resolve()),
        code={name: file_hash(REPO/'demo'/name) for name in CODE}, cpu_only=True,
        holds_GPU_mutex=False, metric_recalculation=False, preview_repeats=1,
        commands=[dict(stage=name, command=command, output=str(marker))
                  for name, command, marker in child_commands(root, supplement)])
    immutable(run.root/'request.json', config)
    waiting = wait_for_completions(run, marker_paths(root, supplement, resilience))
    verify_code(config)
    completed_inputs = completion_evidence(root, supplement, resilience)
    binding = dict(config=config, completed_inputs=completed_inputs)
    immutable(run.root/'inputs.json', binding)
    final_path = run.root/'complete.json'
    if final_path.exists():
        previous = read(final_path)
        require(previous['binding'] == binding, 'completed workflow dependency changed')
        for path, digest in previous['artifacts'].items():
            run.check()
            checked(Path(path), digest)
        run.update(phase='verified_complete_without_recomputation', cpu_only=True, completed=3, total=3)
        return previous
    stages = []
    for index, (name, command, marker) in enumerate(child_commands(root, supplement)):
        run.check()
        verify_code(config)
        step_path = run.root/f'{name}.complete.json'
        if step_path.exists():
            step = read(step_path)
            require(step['command'] == command and step['binding_sha256'] == file_hash(run.root/'inputs.json'),
                    f'completed workflow stage binding changed: {name}')
            checked(marker, step['output_sha256'])
        else:
            phase = time.monotonic()
            run_child(run, name, command)
            require(read(marker).get('complete') is True, f'CPU child completion missing: {name}')
            step = dict(complete=True, stage=name, command=command,
                binding_sha256=file_hash(run.root/'inputs.json'), output=str(marker),
                output_sha256=file_hash(marker), elapsed_seconds=time.monotonic()-phase)
            atomic_json(step_path, step)
        stages.append(step)
        run.update(phase=f'{name}_complete', cpu_only=True, completed=index+1, total=3)
    outputs = output_evidence(root, supplement, run)
    outputs.update(completed_inputs)
    for path in (run.root/'request.json', run.root/'inputs.json',
                 *(run.root/f'{name}.complete.json' for name, _, _ in child_commands(root, supplement))):
        checked(path, hashes=outputs)
    require(completion_evidence(root, supplement, resilience) == completed_inputs,
            'upstream completion evidence changed during CPU finalization')
    verify_code(config)
    elapsed = time.monotonic()-began
    result = dict(complete=True, binding=binding, stages=stages, artifacts=outputs,
        elapsed_seconds_this_attempt=elapsed, waiting_seconds_this_attempt=waiting,
        active_seconds_this_attempt=max(0., elapsed-waiting),
        no_GPU_mutex=True, no_model_inference=True, no_metric_recalculation=True,
        baseline_readonly_replay='already completed by supplement; its GPU-locking verify CLI is not invoked',
        original_timings_preserved=True)
    atomic_json(final_path, result)
    run.update(phase='complete', cpu_only=True, completed=3, total=3)
    return result


def main(args):
    if not os.environ.get('TMUX'):
        raise RuntimeError('run the CPU completion queue in tmux')
    if not math.isfinite(args.max_hours) or args.max_hours <= 0:
        raise ValueError('max-hours must be finite and positive')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    args.root = args.root.resolve()
    supplement = args.root/'supplement'
    if args.output is None:
        args.output = args.root/'workflow'
    args.output = args.output.resolve()
    require(args.output not in (args.root, supplement, args.resilience.resolve()),
            'workflow progress must live in a separate directory')
    args.command = 'finalize'
    from demo.chunk_enhancement_experiment import Run
    run = Run(args)
    run.thread.start()
    try:
        result = finalize(run, args.root, supplement, args.resilience.resolve())
        print(json.dumps(dict(complete=True, output=str(args.output), stages=len(result['stages']))), flush=True)
    except BaseException as error:
        atomic_json(args.output/'last_failure.json', dict(error=repr(error), phase=run.progress))
        raise
    finally:
        run.log_resources()
        run.stop.set()
        run.thread.join(timeout=3)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=DEFAULT)
    p.add_argument('--resilience', type=Path, default=RESILIENCE)
    p.add_argument('--output', type=Path)
    p.add_argument('--max-hours', type=float, default=24.)
    main(p.parse_args())
