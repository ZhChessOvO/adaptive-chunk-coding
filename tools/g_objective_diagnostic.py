"""Single-A800, resumable, read-only audit of the completed mixed-view G losses."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
from demo.routervc_fullview_probe import read, digest, immutable
from demo.scalable_codec import atomic_json
from demo.scalable_experiment import now
from routervc.generation.context_training import choose_view
from routervc.generation.texture_data import REPO, RUNS
from tools.texture_clean import CleanRun
from tools.texture_repair import execute
from tools.texture_context_probe import DEFAULT as CONTEXT

DEFAULT=RUNS/'routervc_g_objectives_20261015'
CODE=('routervc/generation/objective_diagnostic.py','tools/g_objective_diagnostic.py',
      'tools/run_g_objective.sh','demo/feature_condition_train.py')


def protocol(root):
    prior=CONTEXT/'training/formal';old=read(prior/'protocol.json')
    for n,s in old['code'].items():
        if digest(REPO/n)!=s:raise ValueError('pinned dependency changed: '+n)
    initial=prior/'mixed/adapter.pt'; receipt=read(prior/'mixed/complete.json')
    if not receipt['complete'] or digest(initial)!=receipt['adapter_sha256']:raise ValueError('incomplete initial')
    steps=(1,2,3,4,9,10,11,12)
    cases=[];rows={}
    for step in steps:
        row,small,large,crop=choose_view(old['rows'],step,old['seed'],'mixed')
        rows[row['sample_id']]=row
        cases.append(dict(id=f'case_{step:03d}',schedule_step=step,sample_id=row['sample_id'],
            dataset=row['dataset'],E_cap=row['E_cap'],crop=crop,small=small,large=large,repeat=step in (1,2)))
    config=dict(version=1,initial=dict(path=str(initial),sha256=digest(initial)),
        previous_protocol=dict(path=str(prior/'protocol.json'),sha256=digest(prior/'protocol.json')),
        assets=old['assets'],loss=old['loss'],phase_weight=old['phase_weight'],seed=old['seed'],
        rows=list(rows.values()),cases=cases,training=False,
        code={**old['code'],**{n:digest(REPO/n) for n in CODE}},
        scope='eight predetermined training draws; six REDS/two UVG; two local/six large; no optimizer',
        user_feedback='water splash softness is not a major problem or hard gate; prioritize unnatural regular textures and overall naturalness')
    immutable(root/'protocol.json',config);return config


def main(args):
    if not os.environ.get('TMUX'):raise RuntimeError('tmux required')
    root=args.output
    run=CleanRun(SimpleNamespace(output=root,command='objective_diagnostic',max_hours=3))
    run.thread.start();run.log_resources()
    try:
        with exclusive_native_evaluation(run):
            protocol(root)
            execute(run,'gradient_worker','routervc.generation.objective_diagnostic',
                    ['--root',root,'--stop-after',args.stop_after])
            run.update(phase='partial' if args.stop_after else 'complete_pending_interpretation')
    except BaseException as error:
        atomic_json(root/'last_failure.json',dict(error=repr(error),utc=now()));raise
    finally:
        run.log_resources();run.stop.set();run.thread.join();run.lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--stop-after',type=int,default=0);main(p.parse_args())
