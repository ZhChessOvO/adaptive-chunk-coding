"""Read-only completed receiver audit and compact report, without inference."""
import argparse
from pathlib import Path

import numpy as np

from demo.routervc_fullview_probe import read, digest, save, immutable, verify_artifacts
from demo.scalable_format import frame_hash
from routervc.latent import routing, router_data as data
from tools.verify_latent_routers import verify_stage


def paired_summary(points):
    results = {}
    for dataset in ('REDS', 'UVG'):
        results[dataset] = {}
        for count in (0, 4, 8, 16):
            arms = {a: {p['sample_id']: p for p in points if p['dataset'] == dataset
                       and p['count'] == count and p['arm'] == a}
                    for a in ('old_core', 'adapted_core', 'G_off')}
            if not arms['old_core'] or any(set(v) != set(arms['old_core']) for v in arms.values()):
                raise ValueError('paired samples differ')
            deltas = []
            for sid, old in arms['old_core'].items():
                new = arms['adapted_core'][sid]
                if old['total_bytes'] != new['total_bytes']:
                    raise ValueError('receiver comparison bytes differ')
                deltas.append(dict(sample_id=sid,
                    lpips_gain=old['quality']['lpips_alex']-new['quality']['lpips_alex'],
                    psnr_gain=new['quality']['psnr_db']-old['quality']['psnr_db'],
                    temporal_gain=old['quality']['temporal_delta_mae']-new['quality']['temporal_delta_mae']))
            results[dataset][str(count)] = dict(samples=len(deltas),
                lpips_better=sum(d['lpips_gain'] > 1e-9 for d in deltas),
                lpips_same=sum(abs(d['lpips_gain']) <= 1e-9 for d in deltas),
                **{k:float(np.mean([d[k] for d in deltas]))
                   for k in ('lpips_gain', 'psnr_gain', 'temporal_gain')}, individual=deltas)
    return results


def audit(root, output, run):
    training, review = root/'formal', root/'receiver_review'
    binding = dict(code=digest(Path(__file__)), training=digest(training/'complete.json'),
                   review=digest(review/'complete.json'))
    immutable(output/'protocol.json', binding)
    if (output/'complete.json').exists():
        verify_artifacts(output, read(output/'complete.json')['artifacts'])
        return
    deep = verify_stage(training)
    done, protocol = read(review/'complete.json'), read(review/'protocol.json')
    if done['protocol'] != digest(review/'protocol.json') or protocol['receiver_profile'] != routing.identity():
        raise ValueError('review profile changed')
    verify_artifacts(review, done['artifacts'])
    for model in protocol['models'].values():
        if digest(model['path']) != model['sha256']:
            raise ValueError('receiver checkpoint changed')
    summary = read(review/'summary.json')
    previous, bases, enhanced = {}, {}, {}
    prefixes = 0
    for index, point in enumerate(summary['points']):
        run.check()
        sid, arm, count = point['sample_id'], point['arm'], point['count']
        run.progress.update(phase='verify_fresh_artifacts', point=index+1, total=156)
        folder = review/'samples'/sid/f'{arm}_e{count}'
        if read(folder/'result.json') != point:
            raise ValueError('summary and point differ')
        receipt = read(folder/'receive/complete.json')
        raw = (folder/'stream.rvlrg').read_bytes()
        if (digest(folder/'receive/complete.json') != point['receive_sha256']
                or receipt['actual_bytes'] != len(raw) or point['total_bytes'] != len(raw)
                or receipt['stream_sha256'] != digest(folder/'stream.rvlrg')):
            raise ValueError('fresh bytes/receipt changed')
        verify_artifacts(folder/'receive', receipt['artifacts'])
        inner, config, _ = routing.parse(raw)
        if config['receiver_sha256'] != protocol['models']['adapted_core' if arm == 'G_off' else arm]['sha256']:
            raise ValueError('wrong receiver identity in stream')
        with np.load(folder/'receive/pixels.npz', allow_pickle=False) as f:
            b, y, x = (frame_hash(f[k]) for k in ('base', 'enhanced', 'reconstruction'))
        if b != receipt['base_hash'] or x != receipt['output_hash']:
            raise ValueError('stored fresh pixels differ from receiver receipt')
        if bases.setdefault(sid,b) != b or enhanced.setdefault((sid,count),y) != y:
            raise ValueError('receiver-only comparison changed B/Y')
        if arm == 'G_off' and (x != y or point['generated']):
            raise ValueError('G-off did not preserve actual received Y')
        key = sid, arm
        if key in previous:
            if not raw.startswith(previous[key]):
                raise ValueError('not a literal byte prefix')
            prefixes += 1
        previous[key] = raw
    if len(summary['points']) != 156 or prefixes != 117:
        raise ValueError('incomplete review')
    train = read(training/'router/complete.json')
    report = dict(complete=True, deep_training_audit=deep, fresh_points=156,
        literal_prefix_pairs=prefixes, all_B_and_matched_Y_exact=True,
        same_byte_pairs=True, scores_recomputed=False, inference_executed=False,
        selection=dict(receiver_sha256=protocol['models']['adapted_core']['sha256'],
            epoch=train['best']['epoch'], initial_regret=train['initial']['regret'],
            adapted_regret=train['best']['score']['regret'],
            rationale='better at 7/8 dataset-density means; REDS E16 LPIPS exception retained'),
        paired=paired_summary(summary['points']), groups=summary['groups'])
    save(output/'summary.json',report)
    save(output/'complete.json',dict(complete=True,artifacts={'summary.json':digest(output/'summary.json')}))
    print('LATENT_RECEIVER_AUDIT_OK', deep, 'fresh_points=156 prefix_pairs=117',flush=True)


def main():
    import os
    from demo.chunk_enhancement_experiment import Run
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=data.ROOT)
    p.add_argument('--output',type=Path,default=data.ROOT/'receiver_audit_20261009')
    args=p.parse_args()
    if not os.environ.get('TMUX'):
        raise RuntimeError('run even the long read-only audit in tmux')
    from types import SimpleNamespace
    run=Run(SimpleNamespace(output=args.output,command='receiver_readonly_audit',max_hours=4))
    run.thread.start()
    try:
        audit(args.root,args.output,run)
    finally:
        run.log_resources(); run.stop.set(); run.thread.join(timeout=3); run.lock.close()


if __name__ == '__main__':
    main()
