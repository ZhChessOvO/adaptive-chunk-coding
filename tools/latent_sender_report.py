"""CPU-only report of a completed latent sender review; no training or inference.

Run in tmux after the existing sender and system-review verification commands.
Inputs are immutable completed results. Reports go to a separate directory.
"""
import argparse
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def matched_value(points, rate, metric):
    """Descriptive linear interpolation in log rate, with NO extrapolation."""
    points = sorted(points, key=lambda p: p['bpp'])
    rates = [p['bpp'] for p in points]
    if not rates or min(rates) <= 0 or len(set(rates)) != len(rates):
        raise ValueError('rate curve must be positive and have distinct rates')
    if not rates[0] <= rate <= rates[-1]:
        return None
    return float(np.interp(np.log(rate), np.log(rates), [p[metric] for p in points]))


def report(root, output):
    review = root / 'evaluation'
    output.mkdir(parents=True, exist_ok=True)
    binding = {'code': digest(Path(__file__)),
               'training': digest(root/'formal/complete.json'),
               'review': digest(review/'complete.json')}
    if (output/'complete.json').exists():
        done = read(output/'complete.json')
        if done['binding'] != binding:
            raise ValueError('report binding changed; choose a new output directory')
        for name, value in done['artifacts'].items():
            assert digest(output/name) == value, name
        print('COMPLETED_REPORT_VERIFIED', flush=True)
        return
    s = read(review/'summary.json')
    previous, base_hashes, identities = {}, {}, {}
    prefix_pairs = repeats = 0
    for p in s['points']:
        sid, arm, budget = p['sample_id'], p['arm'], p['budget']
        name = f'UF_q{budget}' if arm == 'UF' else f'{arm}_e{int(budget*100):03d}'
        folder = review/'samples'/sid/name
        receipt = read(folder/'receive/complete.json')
        assert digest(folder/'receive/complete.json') == p['receipt_sha256']
        if arm == 'UF':
            continue  # Native artifacts were checked by --verify-only.
        raw = (folder/'stream.rvlrg').read_bytes()
        assert len(raw) == p['actual_bytes'] == receipt['actual_bytes']
        assert hashlib.sha256(raw).hexdigest() == receipt['stream_sha256']
        assert not receipt['detail']['source_frames_read']
        assert receipt['detail']['additional_mask_bytes'] == 0
        assert base_hashes.setdefault(sid, receipt['base_hash']) == receipt['base_hash']
        key = (sid, arm)
        if key in previous:
            assert raw.startswith(previous[key]), key
            prefix_pairs += 1
        previous[key] = raw
        identities[(sid, arm, budget)] = receipt
        if arm == 'source_Goff':
            paired = identities[(sid, 'source', budget)]
            assert receipt['stream_sha256'] == paired['stream_sha256']
            assert receipt['enhanced_hash'] == paired['enhanced_hash']
            assert receipt['output_hash'] == receipt['enhanced_hash'] and p['G_calls'] == 0
        if (folder/'repeat/complete.json').exists():
            again = read(folder/'repeat/complete.json')
            for field in ('stream_sha256', 'base_hash', 'enhanced_hash', 'output_hash'):
                assert again[field] == receipt[field], field
            for name, value in again['artifacts'].items():
                assert digest(folder/'repeat'/name) == value, name
            repeats += 1
    assert len(s['points']) == 299 and prefix_pairs == 156 and repeats == 2
    for sid in base_hashes:
        outputs = [identities[(sid, arm, 0.)]['output_hash']
                   for arm in ('source', 'zero_source', 'fixed_order')]
        assert len(set(outputs)) == 1
    result = dict(binding=binding, points=299, repeated_decodes=2,
                  literal_prefix_pairs=prefix_pairs, base_invariance=True,
                  source_Goff_same_stream=True, zero_E_outputs_exact=True,
                  inference_executed=False, metrics_recomputed=False)
    result['paired_same_cap_not_equal_rate'] = {}
    result['UF_interpolated_at_group_mean_bpp'] = {}
    for ds, arms in s['groups'].items():
        paired, interpolated = {}, {}
        for budget in (.25, .5, 1.):
            key = str(budget)
            paired[key] = {}
            for other in ('zero_source', 'fixed_order', 'source_Goff'):
                a, b = arms['source'][key], arms[other][key]
                values = []
                for p in s['points']:
                    if p['dataset'] != ds or p['arm'] != 'source' or p['budget'] != budget:
                        continue
                    q = next(q for q in s['points'] if q['sample_id'] == p['sample_id']
                             and q['arm'] == other and q['budget'] == budget)
                    values.append(q['quality']['lpips_alex']-p['quality']['lpips_alex'])
                paired[key][other] = dict(lpips_gain=b['lpips_alex']-a['lpips_alex'],
                    bpp_change=a['bpp']-b['bpp'], bpp_change_percent=100*(a['bpp']/b['bpp']-1),
                    psnr_gain=a['psnr_db']-b['psnr_db'], better=sum(v>1e-9 for v in values),
                    tied=sum(abs(v)<=1e-9 for v in values), samples=len(values))
        for budget, a in arms['source'].items():
            interpolated[budget] = {m: matched_value(list(arms['UF'].values()), a['bpp'], m)
                                    for m in ('lpips_alex', 'psnr_db')}
        result['paired_same_cap_not_equal_rate'][ds] = paired
        result['UF_interpolated_at_group_mean_bpp'][ds] = interpolated
    result['interpolation_scope'] = 'dataset mean curve; log-bpp linear; no extrapolation; NOT new measurement or BD-rate'
    planning = []
    for row in s['scope']['rows']:
        for arm in ('source', 'zero_source'):
            folder = review/'samples'/row['sample_id']/f'{arm}_planning'
            done, order = read(folder/'complete.json'), read(folder/'order.json')
            for name, value in done['artifacts'].items():
                assert digest(folder/name) == value, name
            planning.append(dict(sample_id=row['sample_id'], dataset=row['dataset'], arm=arm,
                selected_regions=len(done['order']), seconds=order.get('seconds'),
                points=done['points']))
    result['planning'] = planning
    result['stages'] = {}
    for stage, file in (('labels', root/'formal/label_worker/heartbeat.jsonl'),
                        ('fit', root/'formal/router/heartbeat.jsonl'),
                        ('review', review/'queue/heartbeat.jsonl')):
        events = [json.loads(line) for line in file.read_text().splitlines()]
        if stage == 'review':
            end = next(i for i,e in enumerate(events) if e.get('phase') == 'system_review_complete')
            events = events[:end+1]
            active = [i for i,e in enumerate(events) if e.get('phase') not in (None, 'wait_sender_no_GPU_lock')]
            events = events[active[0]:]
            start = datetime.fromisoformat(events[0]['utc'])
        else:
            start = datetime.fromisoformat(events[0]['utc'])-timedelta(seconds=events[0]['elapsed_seconds'])
        end = datetime.fromisoformat(events[-1]['utc'])
        memory = [int(e['gpu'].split(',')[2]) for e in events if isinstance(e.get('gpu'), str)]
        result['stages'][stage] = dict(start_utc=start.isoformat(), end_utc=end.isoformat(),
            wall_seconds=(end-start).total_seconds(), sampled_device_MiB=max(memory),
            scope='30-second samples; review excludes waiting, start approximate')
    result['output_file_bytes'] = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    history = read(root/'formal/router/history.json')
    selected = read(root/'formal/router/complete.json')['selected']
    initial = read(root/'formal/router/initial_validation.json')
    result['training'] = {arm: dict(epoch=selected[arm]['epoch'], step=selected[arm]['step'],
        initial_regret=initial[arm]['regret'], selected_regret=selected[arm]['validation']['regret'],
        first_train_loss=history[0]['train_loss'][arm], last_train_loss=history[-1]['train_loss'][arm])
        for arm in ('source', 'zero_source')}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for arm, color in (('source', 'tab:blue'), ('zero_source', 'tab:orange')):
        x = [h['epoch'] for h in history]
        axes[0].plot(x, [h['train_loss'][arm] for h in history], label=arm, color=color)
        axes[1].plot([0]+x, [initial[arm]['regret']]+[h['validation'][arm]['regret'] for h in history],
                     label=arm, color=color)
        v = selected[arm]
        axes[1].scatter([v['epoch']], [v['validation']['regret']], color=color, marker='*', s=160)
    for ax, label in zip(axes, ('Mean training loss', 'Validation normalized density regret')):
        ax.set_xlabel('Epoch'); ax.set_ylabel(label); ax.grid(alpha=.2); ax.legend()
    fig.suptitle('120 epochs per sender; stars = selected checkpoints\n'
                 'Local selection among 3 measured additions per state, NOT system RD')
    fig.savefig(output/'sender_training.png', dpi=160); plt.close(fig)
    write(output/'summary.json', result)
    write(output/'complete.json', dict(complete=True, binding=binding,
        artifacts={n:digest(output/n) for n in ('summary.json', 'sender_training.png')}))
    print('SENDER_REPORT_COMPLETE', output, flush=True)


if __name__ == '__main__':
    if not os.environ.get('TMUX'):
        raise RuntimeError('run this read-only report in tmux with tools.storage_guard')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/root/autodl-fs/DCVC/runs/routervc_latent_sender_20261009'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report(args.root, args.output or args.root/'report_20261010')
