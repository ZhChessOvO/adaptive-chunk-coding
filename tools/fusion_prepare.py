"""Bounded P1 data: frozen R_s's real prefixes, source-free R_g/G halo capture."""
import argparse
from collections import Counter
import os
from pathlib import Path
import time
from types import SimpleNamespace

from demo.chunk_enhancement_experiment import Run
from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
from demo.routervc_fullview_probe import read, save, digest, immutable, verify_artifacts
from routervc.latent import sender, sender_data, router_data
from tools.fusion_pilot import worker
from tools.latent_boundary_report import ROOT


def selected_rows(protocol):
    """Take every other existing view within dataset/split; never resplit sequences."""
    selected = set()
    caps = (.25, .5, 1., 0.)
    budget = {}
    for dataset in ('REDS', 'UVG'):
        for split in ('train', 'validation'):
            group = [r for r in protocol['rows'] if r['dataset'] == dataset and r['router_split'] == split][::2]
            for i, row in enumerate(group):
                selected.add(row['sample_id']); budget[row['sample_id']] = caps[i % 4]
    return [dict(r, E_cap=budget[r['sample_id']]) for r in protocol['rows'] if r['sample_id'] in selected]


def prepare(run):
    teacher_root = sender_data.ROOT/'formal'
    checkpoint = teacher_root/'router/source/best.pt'
    teacher = read(teacher_root/'protocol.json')
    rows = selected_rows(teacher)
    if Counter(r['router_split'] for r in rows) != dict(train=48, validation=12):
        raise ValueError('expected the bounded half-sized grouped pilot')
    protocol = dict(version='precision_fusion_on_policy_pilot_v1', rows=rows,
        sender=str(checkpoint), sender_sha256=digest(checkpoint),
        receiver=str(sender_data.RECEIVER), receiver_sha256=digest(sender_data.RECEIVER),
        teacher_protocol=str(teacher_root/'protocol.json'), teacher_sha256=digest(teacher_root/'protocol.json'),
        code={n:digest(Path(__file__).resolve().parents[1]/n) for n in
              ('tools/fusion_prepare.py', 'tools/fusion_capture_worker.py', 'routervc/fusion/capture.py')},
        source_routing=sender.routing.identity(), masks_sent=False, width=3,
        purpose='48 train / 12 validation, real fixed sender/receiver; no Router/G/UF optimization')
    immutable(run.root/'protocol.json', protocol)
    results = []
    for i, row in enumerate(rows):
        run.check(); sid = row['sample_id']; folder = run.root/'samples'/sid
        run.update(phase='actual_sender_prefix', completed=i, total=len(rows), sample=sid)
        if digest(row['bank_path']) != row['bank_sha256']: raise ValueError('bank changed')
        source = router_data.source(row)
        sender.write_prefixes(folder/'planning', source, Path(row['bank_path']).read_bytes(), checkpoint,
                              check=run.check, progress=lambda **kw: run.update(sender_progress=kw))
        del source
        stream = folder/'planning'/f'e{round(100*row["E_cap"]):03d}.rvlrg'
        run.update(phase='sourcefree_halo_capture', sample=sid, E_cap=row['E_cap'])
        worker(run, stream, folder/'capture')
        done = read(folder/'capture/complete.json'); verify_artifacts(folder/'capture', done['artifacts'])
        results.append(dict(sample_id=sid, dataset=row['dataset'], split=row['router_split'],
                            cap=row['E_cap'], stream=str(stream), capture=str(folder/'capture'),
                            receipt_sha256=digest(folder/'capture/complete.json')))
        save(run.root/'index.partial.json', dict(complete=False, samples=results))
    save(run.root/'complete.json', dict(complete=True, protocol_sha256=digest(run.root/'protocol.json'),
                                      samples=results, count=len(rows)))
    run.update(phase='data_complete', completed=len(rows)); run.log_resources()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT/'p1_data')
    args = p.parse_args()
    if not os.environ.get('TMUX'): raise RuntimeError('tmux required')
    run = Run(SimpleNamespace(output=args.output, command='prepare', max_hours=24)); run.thread.start()
    try:
        # Do not hold the GPU mutex while the earlier paired controls are running.
        while not (ROOT/'p1_controls/complete.json').exists():
            if (ROOT/'p1_controls/last_failure.json').exists():
                raise RuntimeError('paired controls need attention before formal data preparation')
            run.check(); run.update(phase='waiting_for_paired_controls'); time.sleep(5)
        with exclusive_native_evaluation(run): prepare(run)
    except BaseException as error:
        save(run.root/'last_failure.json', dict(error=repr(error), progress=run.progress)); raise
    finally: run.stop.set(); run.thread.join(); run.lock.close()


if __name__ == '__main__': main()
