"""CPU-only diagnostics of saved q1/q2 labels and fixed training histories.

No inference, checkpoint selection, mixed-region oracle or new RD measurement.
Regional scores describe the component-training pool, not independent tests.
"""
import argparse
import math
import os
from pathlib import Path
from types import SimpleNamespace

from demo.routervc_fullview_probe import read, digest, save, immutable, verify_artifacts
from demo.routervc_light_teacher import OUTPUT, HISTORY
from demo.routervc_light_evaluate import REVISION, ARMS

STATES = ('B', 'E', 'G', 'EG')


def paired_regions(old, new):
    for field in ('sample_id', 'dataset', 'sequence', 'router_split'):
        if old[field] != new[field]:
            raise ValueError('q1/q2 teacher identity differs')
    if len(old['regions']) != 16 or len(new['regions']) != 16:
        raise ValueError('expected sixteen measured regions')
    rows = []
    for i, (a, b) in enumerate(zip(old['regions'], new['regions'])):
        if a['region'] != i or b['region'] != i or a['roi'] != b['roi']:
            raise ValueError('q1/q2 region alignment differs')
        if any(a['quality'][s] != b['quality'][s] for s in ('B', 'G')):
            raise ValueError('unchanged B/G teacher differs')
        row = {k: old[k] for k in ('sample_id', 'dataset', 'sequence', 'router_split')}
        row['region'] = i
        for q, r in (('q1', a), ('q2', b)):
            values = {s: r['quality'][s]['lpips_alex'] for s in STATES}
            if not all(math.isfinite(v) and v >= 0 for v in values.values()):
                raise ValueError('invalid regional LPIPS')
            cost = r['costs']['e_packet_bytes']
            if type(cost) is not int or cost <= 0:
                raise ValueError('invalid measured E packet cost')
            row[q] = dict(lpips=values, e_packet_bytes=cost,
                E_over_B=values['B']-values['E'], G_over_B=values['B']-values['G'],
                EG_over_E=values['E']-values['EG'], EG_over_G=values['G']-values['EG'],
                EG_better_than_both=values['EG'] < min(values['E'], values['G']))
        rows.append(row)
    return rows


def aggregate(rows):
    if not rows:
        raise ValueError('empty label diagnostic group')
    result = dict(windows=len({r['sample_id'] for r in rows}), regions=len(rows),
                  sequences=len({(r['dataset'], r['sequence']) for r in rows}))
    for q in ('q1', 'q2'):
        values = [r[q] for r in rows]
        result[q] = dict(
            mean_regional_lpips={s: sum(v['lpips'][s] for v in values)/len(values) for s in STATES},
            mean_gains={s: sum(v[s] for v in values)/len(values)
                        for s in ('E_over_B', 'G_over_B', 'EG_over_E', 'EG_over_G')},
            e_packet_bytes=sum(v['e_packet_bytes'] for v in values),
            EG_better_than_G=sum(v['EG_over_G'] > 0 for v in values),
            EG_better_than_both=sum(v['EG_better_than_both'] for v in values))
    return result


def charts(groups, histories, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from demo.routervc_visual_report import save_plot

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    colors = ('#6684ab', '#864ab8')
    for ax, ds in zip(axes, ('REDS', 'UVG')):
        group = groups[ds]['all']
        for j, q in enumerate(('q1', 'q2')):
            bars = ax.bar([i+(j-.5)*.34 for i in range(4)],
                [group[q]['mean_regional_lpips'][s] for s in STATES], .34, label=q, color=colors[j])
            ax.bar_label(bars, labels=['' if s in ('B', 'G') else
                f'{group[q]["mean_regional_lpips"][s]:.3f}' for s in STATES], fontsize=8)
        for i in (0, 2):
            value = group['q1']['mean_regional_lpips'][STATES[i]]
            ax.annotate(f'{value:.3f}', (i, value), xytext=(0, 3),
                        textcoords='offset points', ha='center', fontsize=8)
        ax.set_xticks(range(4), STATES)
        ax.set_title(f'{ds}: {group["windows"]} training-pool windows')
        ax.set_ylabel('Mean isolated-region LPIPS (lower)')
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
        ax.margins(y=.18)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=2,
               bbox_to_anchor=(.5, .07), frameon=False)
    fig.suptitle('Do cheaper packets still help generation? Saved teacher measurements')
    fig.text(.5, .02, 'B/G are identical reuse. Equal region means; not whole-video RD or independent test evidence.',
             ha='center', fontsize=8)
    fig.subplots_adjust(top=.85, bottom=.22, wspace=.25)
    save_plot(fig, output/'teacher_states.png')

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for i, version in enumerate(('old', 'new')):
        for j, arm in enumerate(ARMS):
            ax = axes[i, j]; history = histories[f'{version}_{arm}']['history']
            ax.plot([r['epoch'] for r in history], [r['train_loss'] for r in history], label='Train')
            ax.plot([r['epoch'] for r in history], [r['validation']['loss'] for r in history], label='Validation')
            ax.set_title(f'{version}: {arm}'); ax.set_xlabel('Epoch')
            ax.set_ylabel('Normalized gain prediction loss'); ax.grid(alpha=.2); ax.legend()
    fig.suptitle('Fixed 120-epoch endpoints: recorded histories, no checkpoint reselection')
    fig.text(.5, .02, 'Each teacher has its own train-set normalization; losses across q1/q2 are not raw-quality gains.',
             ha='center', fontsize=8)
    fig.subplots_adjust(top=.9, bottom=.12, hspace=.43, wspace=.26)
    save_plot(fig, output/'training_histories.png')


def build(root, output):
    import torch
    torch.set_num_threads(2)
    paths = [HISTORY/'labels.json', root/'teacher/labels.json']
    manifests = [read(p) for p in paths]
    if not all(m['complete'] for m in manifests):
        raise ValueError('teacher incomplete')
    indices = [{r['sample_id']: r for r in m['samples']} for m in manifests]
    if (indices[0].keys() != indices[1].keys()
            or any(len(index) != len(m['samples']) for index, m in zip(indices, manifests))):
        raise ValueError('teacher sample set differs or duplicates')
    rows = []
    for sid in indices[0]:
        labels = []
        for index in indices:
            entry = index[sid]; path = Path(entry['path'])
            if digest(path) != entry['sha256']:
                raise ValueError('teacher label changed')
            paths.append(path); labels.append(read(path))
        rows += paired_regions(*labels)
    groups = {ds: {split: aggregate([r for r in rows if r['dataset'] == ds
                    and (split == 'all' or r['router_split'] == split)])
                  for split in ('all', 'train', 'validation')} for ds in ('REDS', 'UVG')}
    histories = {}
    for version, parent in (('old', REVISION/'visual_router'), ('new', root/'router')):
        for arm in ARMS:
            folder = parent/arm; complete = read(folder/'complete.json')
            verify_artifacts(folder, complete['artifacts'])
            paths += [folder/'complete.json', folder/'resume.pt']
            state = torch.load(folder/'resume.pt', weights_only=True, map_location='cpu')
            history = state['history']
            if state['epoch'] != 120 or len(history) != 120 or history[-1]['epoch'] != 120:
                raise ValueError('not the paired fixed 120-epoch history')
            best = min(history, key=lambda h: h['validation']['loss'])
            histories[f'{version}_{arm}'] = dict(history=history, final=history[-1],
                minimum_recorded_validation=dict(epoch=best['epoch'], loss=best['validation']['loss']),
                checkpoint_reselected=False)
    binding = {str(p): digest(p) for p in paths}
    immutable(output/'inputs.json', binding)
    charts(groups, histories, output)
    save(output/'summary.json', dict(complete=True, groups=groups, rows=rows, training=histories,
        scope='isolated teacher regions in component-training pool; no whole-video RD inference',
        no_inference=True, no_checkpoint_reselection=True, semantic_supervision=False))
    return binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=OUTPUT)
    parser.add_argument('--output', type=Path, default=OUTPUT/'label_diagnostics')
    args = parser.parse_args()
    if not os.environ.get('TMUX'):
        raise RuntimeError('formal report requires tmux')
    if args.output.resolve() == args.root.resolve() or any(args.output.resolve().is_relative_to(
            (args.root/n).resolve()) for n in ('teacher', 'router', 'evaluation', 'report')):
        raise ValueError('diagnostics must not overwrite source artifacts')
    from demo.chunk_enhancement_experiment import Run
    run = Run(SimpleNamespace(output=args.output, command='light_label_report', max_hours=1))
    run.thread.start()
    try:
        done = args.output/'complete.json'
        if done.exists():
            marker = read(done); verify_artifacts(args.output, marker['artifacts'])
            if marker['code_sha256'] != digest(__file__):
                raise ValueError('diagnostic source changed')
            for path, sha in marker['inputs'].items():
                if digest(path) != sha:
                    raise ValueError('diagnostic input changed')
            print('LIGHT_LABEL_REPORT_VERIFIED_READ_ONLY', flush=True)
            return
        run.update(phase='saved_labels_and_histories')
        binding = build(args.root, args.output)
        names = ('inputs.json', 'summary.json', 'teacher_states.png', 'training_histories.png')
        save(done, dict(complete=True, code_sha256=digest(__file__), inputs=binding,
                       artifacts={n: digest(args.output/n) for n in names}))
        run.update(phase='complete', completed=120, total=120)
        print('LIGHT_LABEL_REPORT_COMPLETE', flush=True)
    finally:
        run.log_resources(); run.stop.set(); run.thread.join(timeout=3); run.lock.close()


if __name__ == '__main__':
    main()
