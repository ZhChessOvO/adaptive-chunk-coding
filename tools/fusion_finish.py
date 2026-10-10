"""Finish the approved P1 queue without choosing/promoting a new method."""
import argparse
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace


def pending_stage(root):
    """Stop on an upstream failure; a completed stage wins over old failures."""
    for stage, failure_dir in (
        ('p1_data', 'p1_data'), ('p1_fit', 'p1_fit'),
        ('p1_evaluation', 'p1_evaluation/queue'),
    ):
        if (root / stage / 'complete.json').exists():
            continue
        failure = root / failure_dir / 'last_failure.json'
        if failure.exists():
            raise RuntimeError(f'upstream needs attention: {failure}')
        return stage
    return None


def memory_snapshot():
    """This host exposes its container limit through cgroup v1, not free(1)."""
    folder = Path('/sys/fs/cgroup/memory')
    if not (folder / 'memory.stat').exists():
        return {'available': False}
    stats = dict(line.split() for line in (folder / 'memory.stat').read_text().splitlines())
    return dict(available=True, units='bytes',
        limit=int((folder / 'memory.limit_in_bytes').read_text()),
        charged=int((folder / 'memory.usage_in_bytes').read_text()),
        rss=int(stats['rss']), cache=int(stats['cache']),
        shmem=int(stats['shmem']), dirty=int(stats['dirty']),
        failcnt=int((folder / 'memory.failcnt').read_text()))


def main():
    from demo.chunk_enhancement_experiment import Run
    from demo.routervc_fullview_probe import read, save, digest, verify_artifacts
    from tools.fusion_review import execute
    from tools.latent_boundary_report import ROOT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    if not os.environ.get('TMUX'):
        raise RuntimeError('tmux required')

    class FinishRun(Run):
        def log_resources(self):
            super().log_resources()
            with (self.root / 'memory.jsonl').open('a') as log:
                log.write(json.dumps(dict(unix_seconds=time.time(), **memory_snapshot())) + '\n')

    run = FinishRun(SimpleNamespace(output=ROOT/'p1_handoff', command='finish', max_hours=72))
    run.thread.start()
    try:
        if (run.root/'complete.json').exists():
            verify_artifacts(run.root, read(run.root/'complete.json')['artifacts'])
            run.update(phase='complete_already'); return
        while (stage := pending_stage(ROOT)) is not None:
            run.check(); run.update(phase='waiting_for_' + stage); time.sleep(5)
        for stage in ('p1_fit', 'p1_evaluation'):
            verify_artifacts(ROOT/stage, read(ROOT/stage/'complete.json')['artifacts'])
        run.update(phase='learned_boundary_perceptual')
        boundary = ROOT/'p1_boundary_learned'
        if not (boundary/'complete.json').exists():
            execute(run, 'boundary_quality', 'tools.fusion_boundary_quality', ['--learned'])
        verify_artifacts(boundary, read(boundary/'complete.json')['artifacts'])
        paired = read(ROOT/'p1_evaluation/summary.json')
        audit = read(ROOT/'p1_evaluation/fresh_audit.json')
        if not (audit['sourcefree_pixels_exact'] and audit['repeats_exact']
                and audit['Goff_requires_no_Rg_G_F_assets']):
            raise ValueError('fresh receiver checks are incomplete')
        fresh = []
        for row in audit['receipts']:
            path = ROOT/'p1_evaluation/samples'/row['sample_id']/'fresh'/row['variant']/'complete.json'
            if digest(path) != row['receipt']:
                raise ValueError('fresh receipt changed')
            value = read(path)
            fresh.append(dict(sample_id=row['sample_id'], variant=row['variant'],
                resources={k:v for k,v in value.items() if k.startswith('peak_') or 'seconds' in k}))
        upstream = {
            'training': ROOT/'p1_fit/complete.json',
            'paired': ROOT/'p1_evaluation/summary.json',
            'fresh': ROOT/'p1_evaluation/fresh_audit.json',
            'boundary': boundary/'summary.json',
        }
        summary = dict(complete=True, method_promoted=False,
            scope='P1 fixed-width/fixed-router fusion pilot; no P2 or adaptive-width job queued',
            training=read(upstream['training']),
            whole_frame=paired['groups'], boundary=read(upstream['boundary'])['groups'],
            per_view=[dict(sample_id=r['sample_id'], dataset=r['dataset'],
                quality={m:v['quality'] for m,v in r['variants'].items()},
                actual_bytes=r['actual_bytes'], additional_header_bytes=r['additional_header_bytes'],
                additional_mask_bytes=r['additional_mask_bytes'],
                learned_seconds_cached_multiband_no_model_load=r['learned_seconds_cached_multiband_no_model_load'],
                learned_peak_cuda_allocated_bytes=r['learned_peak_cuda_allocated_bytes'],
                comparison_png=str(ROOT/'p1_evaluation/samples'/r['sample_id']/'comparison.png'))
                for r in paired['results']],
            fresh_checks=audit, fresh_resources=fresh,
            note='Diagnostic views, not an unseen-video benchmark. Cached F timing excludes G and model load.',
            memory=memory_snapshot(), upstream={k:dict(path=str(p), sha256=digest(p)) for k,p in upstream.items()})
        save(run.root/'summary.json', summary)
        save(run.root/'complete.json', dict(complete=True, method_promoted=False,
            artifacts={'summary.json': digest(run.root/'summary.json')}))
        run.update(phase='all_approved_P1_work_complete'); run.log_resources()
    except BaseException as error:
        save(run.root/'last_failure.json', dict(error=repr(error), progress=run.progress)); raise
    finally:
        run.stop.set(); run.thread.join(); run.lock.close()


if __name__ == '__main__':
    main()
