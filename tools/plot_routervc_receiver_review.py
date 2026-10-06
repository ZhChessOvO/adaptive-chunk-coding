"""Readable supplemental plots from completed receiver results; no inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def inputs(root):
    """Check the three consumed artifacts against their completion receipts."""
    paths = (root / 'report/summary.json', root / 'formal/router/history.json',
             root / 'formal/router/initial_validation.json')
    for path in paths:
        receipt = read(path.parent / 'complete.json')
        if not receipt['complete'] or receipt['artifacts'][path.name] != digest(path):
            raise ValueError(f'completed input changed: {path}')
    return [read(path) for path in paths], {str(path): digest(path) for path in paths}


def render(root, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    (summary, history, initial), hashes = inputs(root)
    if output.resolve() == (root / 'report').resolve():
        raise ValueError('do not overwrite the immutable original report')
    output.mkdir(parents=True, exist_ok=True)
    names = ('rd_readable.png', 'paired_lpips.png', 'training_selection.png')
    if any((output / name).exists() for name in (*names, 'complete.json')):
        raise FileExistsError('use a fresh supplemental output directory')
    plt.rcParams.update({'font.size': 11})
    colors = {'shared': '#777777', 'core': '#276fbf', 'halo': '#ae4b82'}
    labels = {'shared': 'Old shared Router', 'core': 'Independent Rg / core',
              'halo': 'Independent Rg / halo'}
    ratios = (0, .25, .5)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for i, dataset in enumerate(('REDS', 'UVG')):
        group = summary['group_means'][dataset]
        curves = [('Native DCVC-UF', '#303030', 'o', '-',
                   [group[f'uf_qp{q}'] for q in (8,16,24,32,40,48,56)])]
        for arm, marker, style in (('shared', 's', '--'), ('core', 'o', '-'), ('halo', '^', ':')):
            curves.append((labels[arm], colors[arm], marker, style,
                           [group[f'{arm}_e{r:g}_g8'] for r in ratios]))
        curves.append(('Base + whole-frame G', '#23905a', '*', 'None',
                       [group['wholeframe_g_one_roi']]))
        for j, metric in enumerate(('lpips_alex', 'psnr_db')):
            ax = axes[i, j]
            for label, color, marker, style, values in curves:
                ax.plot([v['bpp'] for v in values], [v[metric] for v in values],
                        label=label, color=color, marker=marker, linestyle=style,
                        linewidth=1.6, markersize=6 if marker != '*' else 12)
            ax.set_xscale('log')
            ticks = [.007, .015, .03, .07, .15] if dataset == 'REDS' else [.004, .008, .016, .032, .064]
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.grid(alpha=.2)
            ax.set_xlabel('Actual complete-stream bpp (log scale)')
            ax.set_ylabel('LPIPS (lower is better)' if j == 0 else 'PSNR / dB (higher is better)')
            ax.set_title('REDS: 6 resized full views' if dataset == 'REDS' else 'UVG: 7 existing crops')
    fig.suptitle('Receiver-only comparison: same E packets, frozen UF / E / G')
    fig.legend(*axes[0,0].get_legend_handles_labels(), loc='lower center', ncol=3,
               bbox_to_anchor=(.5,.035), frameon=False)
    fig.text(.5,.012,'Measured diagnostic means, not a full benchmark. Routes can change diffusion noise assignment.',
             ha='center', fontsize=9)
    fig.subplots_adjust(top=.92,bottom=.19,hspace=.36,wspace=.23)
    fig.savefig(output / names[0], dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1,2,figsize=(12,4.4),sharey=True)
    for ax, dataset in zip(axes, ('REDS','UVG')):
        group = summary['group_means'][dataset]
        for arm, marker in (('core','o'),('halo','^')):
            deltas=[group[f'{arm}_e{r:g}_g8']['lpips_alex']-
                    group[f'shared_e{r:g}_g8']['lpips_alex'] for r in ratios]
            ax.plot(range(3),deltas,color=colors[arm],marker=marker,label=labels[arm])
        ax.axhline(0,color='#555555',linestyle='--',linewidth=1)
        ax.set_xticks(range(3),('E0','E25','E50'))
        ax.set_xlabel('Fixed E byte-budget setting')
        ax.set_title(dataset + (' / 6 full views' if dataset=='REDS' else ' / 7 crops'))
        ax.grid(alpha=.2)
    axes[0].set_ylabel('LPIPS difference vs old shared Router\nNegative = better; positive = worse')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False)
    fig.suptitle('A small aggregate change, with dataset-dependent trade-offs')
    fig.subplots_adjust(top=.84,bottom=.24,wspace=.14)
    fig.savefig(output/names[1],dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1,2,figsize=(12,4.6))
    for arm in ('core','halo'):
        epochs=[h['epoch'] for h in history]
        axes[0].plot(epochs,[h['train_loss'][arm] for h in history],color=colors[arm],label=arm)
        values=[initial[arm]['regret']]+[h['validation'][arm]['regret'] for h in history]
        axes[1].plot([0]+epochs,values,color=colors[arm],label=arm)
        best=min(history,key=lambda h:(h['validation'][arm]['regret'],h['validation'][arm]['loss']))
        axes[1].scatter([best['epoch']],[best['validation'][arm]['regret']],s=80,
                        marker='*',color=colors[arm],zorder=5)
    for ax in axes:
        ax.set_xlabel('Completed epoch');ax.grid(alpha=.2);ax.legend(frameon=False)
    axes[0].set_ylabel('Training loss (lower)')
    axes[1].set_ylabel('Validation local-G ranking regret (lower)')
    axes[0].set_title('Training fit continues to improve')
    axes[1].set_title('Best validation checkpoints: core 2, halo 1')
    fig.suptitle('120 epochs completed; evaluation uses best, not last')
    fig.text(.5,.01,'Local-G ranking regret is a selection metric, not whole-video LPIPS or RD.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.94))
    fig.savefig(output/names[2],dpi=160)
    plt.close(fig)
    receipt=dict(complete=True,inputs=hashes,code_sha256=digest(Path(__file__)),
                 artifacts={name:digest(output/name) for name in names},inference=False)
    (output/'complete.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    render(args.root,args.output)
