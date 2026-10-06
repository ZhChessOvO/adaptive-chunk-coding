"""Read-only, CPU-only overview of completed RouterVC and native-UF measurements.

This supplements (never replaces) the immutable training/evaluation reports.
It does not train, infer, choose checkpoints, extrapolate RD, or recompute scores.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

from demo.routervc_fullview_probe import digest, read, save, verify_artifacts

ROOT = Path('/root/autodl-fs/DCVC/runs/routervc_sender_20261005')
METRICS = ('bpp', 'lpips_alex', 'psnr_db', 'temporal_delta_mae')
RATIOS = (0., .25, .5)
DATASETS = ('REDS', 'UVG')


def nearest_rate(rows, bpp):
    """Pick a measured native-UF point by rate alone, never by picture quality."""
    if bpp <= 0 or not rows or any(r['bpp'] <= 0 for r in rows):
        raise ValueError('positive rates and nonempty candidates required')
    return min(rows, key=lambda r: (abs(math.log(r['bpp'] / bpp)), r['point']))


def memory_peak(decode):
    """Do not mistake the last ROI's reset counter for all G calls' peak."""
    values = [decode['peak_cuda_allocated_bytes']]
    values += [w['runtime']['peak_cuda_allocated_bytes']
               for w in (decode.get('generation_runtime') or {}).get('windows', [])]
    return max(values) / 2**30


def means(rows, keys=METRICS):
    if not rows:
        raise ValueError('empty mean')
    return {k: sum(r[k] for r in rows) / len(rows) for k in keys}


def center_box(width, height):
    w, h = width // 3, height // 3
    x, y = (width - w) // 2, (height - h) // 2
    return x, y, x + w, y + h


def collect(root):
    report = root / 'report'
    receipt = read(report / 'complete.json')
    verify_artifacts(report, receipt['artifacts'])
    verify_artifacts(Path('/'), receipt['inputs'])
    data = read(report / 'summary.json')
    protocol = read(root / 'evaluation/protocol.json')
    inputs = {}

    def bind(path, expected=None):
        path = Path(path)
        actual = digest(path)
        if expected is not None and actual != expected:
            raise ValueError(f'input changed: {path}')
        inputs[str(path)] = actual

    for path in (report / 'complete.json', report / 'summary.json',
                 root / 'evaluation/protocol.json'):
        bind(path)
    for item in protocol['inputs'].values():
        bind(item['source_path'], item['source_sha256'])
    resources = []
    for row in data['rows'] + data['baselines']:
        is_current = row.get('arm') == 'source'
        if 'arm' in row and not is_current:
            continue
        folder = Path(row['output_folder'] if is_current else row['reference'])
        rec = read(folder / 'result.json')
        # Authenticate every consumed historical reconstruction and measurement.
        verify_artifacts(folder, rec['artifacts'])
        for name in ('result.json', 'fresh/decode.json', 'fresh/reconstruction.npz'):
            bind(folder / name)
        stream = folder / ('stream.rvrc' if is_current else
                           'stream.acsg' if row['point'] == 'wholeframe_g_one_roi' else 'stream.bin')
        if not stream.exists() and is_current:
            # Exact name is recorded among the point's immutable artifacts.
            stream = next(folder / n for n in rec['artifacts'] if n.endswith('.rvrc'))
        if stream.stat().st_size != row['bytes'] or row['bytes'] != rec['bytes']:
            raise ValueError('measured rate is not the on-disk stream size')
        bind(stream)
        if any(abs(row[k] - (rec[k] if k == 'bpp' else rec['quality'][k])) > 1e-12
               for k in METRICS):
            raise ValueError('summary metric mismatch')
        d = read(folder / 'fresh/decode.json')
        family = 'RouterVC' if is_current else 'Full G' if row['point'] == 'wholeframe_g_one_roi' else 'UF'
        resources.append(dict(sample_id=row['sample_id'], dataset=row['dataset'],
            family=family, point=row['point'], historical=not is_current,
            peak_allocated_GiB=memory_peak(d),
            peak_reserved_GiB=d.get('peak_cuda_reserved_bytes', 0) / 2**30 or None,
            seconds=d['seconds'], policy_seconds=d.get('policy_seconds'),
            allocation_seconds=row.get('order_allocation_seconds'),
            G_seconds=(d.get('generation_runtime') or {}).get('seconds_model_load_excluded'),
            base_bytes=d.get('base_bytes', d.get('native_bytes')),
            packet_bytes=d.get('packet_bytes', 0),
            framing_bytes=d.get('container_header_bytes', 0) + d.get('generation_control_bytes', 0)))
    return data, protocol, resources, inputs


def rd_figure(data, output):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullLocator
    grouped = {}
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9))
    for i, dataset in enumerate(DATASETS):
        g = data['group_means'][dataset]
        current = [g[f'source_e{r:g}_g8'] for r in RATIOS]
        off = [dict(bpp=current[j]['bpp'], **means([
            row['Goff'] for row in data['rows'] if row['arm'] == 'source'
            and row['dataset'] == dataset and row['ratio'] == ratio], METRICS[1:]))
            for j, ratio in enumerate(RATIOS)]
        grouped[dataset] = dict(current=current, same_wire_G_off=off)
        curves = [('Native DCVC-UF', '#444444', 'o', '-', [g[f'uf_qp{q}'] for q in (8,16,24,32,40,48,56)]),
                  ('RouterVC: source Rs + core Rg, G8', '#226bc4', 'o', '-', current),
                  ('Same RouterVC stream, G off', '#b57b35', 's', ':', off),
                  ('Base + whole-frame G (historical)', '#16835f', '*', 'None', [g['wholeframe_g_one_roi']])]
        for j, key in enumerate(('lpips_alex', 'psnr_db')):
            ax = axes[i, j]
            for title, color, marker, style, points in curves:
                ax.plot([p['bpp'] for p in points], [p[key] for p in points],
                        label=title, color=color, marker=marker, linestyle=style,
                        linewidth=2 if title.startswith('RouterVC:') else 1.5, markersize=8 if marker=='*' else 5)
            for label, point in zip(('E0', 'E25', 'E50'), current):
                ax.annotate(label, (point['bpp'], point[key]), xytext=(4, 7),
                            textcoords='offset points', fontsize=8, color='#226bc4')
            ax.set_xscale('log'); ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xticks([.007, .015, .03, .07, .15] if dataset == 'REDS' else [.004,.008,.016,.032,.064])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
            ax.grid(alpha=.2); ax.set_xlabel('Measured transmitted bpp (log scale)')
            ax.set_ylabel('LPIPS-Alex / lower is better' if j == 0 else 'PSNR / dB / higher is better')
            ax.set_title('REDS: 6 full views, 1024 x 576' if dataset == 'REDS' else 'UVG: 7 crops, 512 x 512')
    fig.suptitle('Current RouterVC vs DCVC-UF | measured 17-frame windows', fontsize=15)
    fig.legend(*axes[0,0].get_legend_handles_labels(), loc='lower center', ncol=2,
               bbox_to_anchor=(.5, .035), frameon=False, fontsize=10)
    fig.text(.5,.012,'Not a full benchmark. E25/E50 are enhancement-byte caps, not area fractions. Whole-frame G uses a different seed.',ha='center',fontsize=9)
    fig.subplots_adjust(top=.92,bottom=.20,hspace=.32,wspace=.25)
    fig.savefig(output/'rd_current.png',dpi=170); plt.close(fig)
    return grouped


def resource_figures(rows, output):
    import matplotlib.pyplot as plt
    import numpy as np
    grouped = {}
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8))
    for i, dataset in enumerate(DATASETS):
        group = {}
        for family in ('UF', 'RouterVC', 'Full G'):
            records = [r for r in rows if r['dataset'] == dataset and r['family'] == family]
            group[family] = dict(n=len(records),
                max_allocated_GiB=max(r['peak_allocated_GiB'] for r in records),
                max_reserved_GiB=max((r['peak_reserved_GiB'] or 0 for r in records), default=0) or None,
                mean_seconds=means(records, ('seconds',))['seconds'],
                min_seconds=min(r['seconds'] for r in records), max_seconds=max(r['seconds'] for r in records),
                mean_policy_seconds=sum(r['policy_seconds'] or 0 for r in records)/len(records) if family=='RouterVC' else None,
                historical=family!='RouterVC')
        grouped[dataset] = group
        for j, metric in enumerate(('max_allocated_GiB', 'mean_seconds')):
            ax = axes[i,j]; values = [group[f][metric] for f in ('UF','RouterVC','Full G')]
            bars=ax.bar(('UF*','RouterVC','Full G*'), values, color=('#666666','#226bc4','#16835f'))
            ax.bar_label(bars, fmt='%.2f', padding=4); ax.set_ylim(0,max(values)*1.25)
            ax.set_ylabel('Max recorded CUDA allocated / GiB' if j==0 else 'Mean fresh decode / seconds per 17 frames')
            ax.set_title(dataset + (' / 1024 x 576' if dataset=='REDS' else ' / 512 x 512'))
            ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Memory and decode time | single A800 80GB',fontsize=15)
    fig.text(.5,.035,'* Historical fresh-process controls; timings are indicative, not a new synchronized speed benchmark.',ha='center',fontsize=9)
    fig.text(.5,.012,'CUDA allocated is not device VRAM capacity. Decode includes model loading; no real-time or 8GB-device claim.',ha='center',fontsize=9)
    fig.subplots_adjust(top=.9,bottom=.14,hspace=.34,wspace=.25)
    fig.savefig(output/'resources.png',dpi=170);plt.close(fig)
    fig, axes=plt.subplots(1,2,figsize=(11,4.8))
    for ax,dataset in zip(axes,DATASETS):
        groups=[means([r for r in rows if r['dataset']==dataset and r['point']==f'source_e{v:g}_g8'],
                      ('base_bytes','packet_bytes','framing_bytes')) for v in RATIOS]
        bottom=np.zeros(3)
        for key,label,color in [('base_bytes','UF base','#666666'),('packet_bytes','E packets + packet headers','#226bc4'),('framing_bytes','Container + receiver profile','#e8b453')]:
            values=np.array([g[key]/1024 for g in groups]);ax.bar(('E0','E25','E50'),values,bottom=bottom,label=label,color=color);bottom+=values
        ax.set_title(dataset);ax.set_ylabel('Mean actual stream / KiB per 17-frame window')
        ax.set_ylim(0,max(bottom)*1.15)
        for x,y in enumerate(bottom):ax.text(x,y+.2,f'{y:.2f}',ha='center',fontsize=9)
    fig.suptitle('Where the bytes go | zero explicit E / G / protection masks')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False)
    fig.subplots_adjust(top=.86,bottom=.2,wspace=.25)
    fig.savefig(output/'bytes.png',dpi=170);plt.close(fig)
    return grouped


def qualitative(data, protocol, output):
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    folder=output/'qualitative';folder.mkdir(exist_ok=True)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    records=[]
    for row in data['rows']:
        if row['arm']!='source' or row['ratio']!=.5:continue
        sid=row['sample_id'];refs=[r for r in data['baselines'] if r['sample_id']==sid]
        uf=nearest_rate([r for r in refs if r['point'].startswith('uf_qp')],row['bpp'])
        full=next(r for r in refs if r['point']=='wholeframe_g_one_roi')
        with np.load(protocol['inputs'][sid]['source_path'],allow_pickle=False) as n:
            source=n['source'][8].copy()
        with np.load(Path(row['output_folder'])/'fresh/reconstruction.npz',allow_pickle=False) as n:
            final=n['reconstruction'][8].copy(); enhanced=n['enhanced'][8].copy()
        with np.load(Path(uf['reference'])/'fresh/reconstruction.npz',allow_pickle=False) as n:
            uf_pixels=n['reconstruction'][8].copy()
        with np.load(Path(full['reference'])/'fresh/reconstruction.npz',allow_pickle=False) as n:
            full_pixels=n['reconstruction'][8].copy()
        panels=[('Original',source,None),('DCVC-UF '+uf['point'].replace('uf_',''),uf_pixels,uf),
                ('RouterVC E50 / G8',final,row),('Same stream / G off',enhanced,dict(bpp=row['bpp'],**row['Goff'])),
                ('Base + full G (historical)',full_pixels,full)]
        if any(pixels.shape!=source.shape or pixels.dtype!=np.uint8 for _,pixels,_ in panels):
            raise ValueError('non-matching image geometry or dtype')
        width=360; height=round(source.shape[0]*width/source.shape[1]); gap=12
        box=center_box(source.shape[1],source.shape[0]);zoom_h=round((box[3]-box[1])*width/(box[2]-box[0]))
        canvas=Image.new('RGB',(5*width+6*gap,184+height+zoom_h),'white');draw=ImageDraw.Draw(canvas)
        draw.text((gap,9),sid+' | fixed frame 8 | numbers: full 17-frame window',font=font,fill='black')
        mismatch=100*(uf['bpp']/row['bpp']-1)
        draw.text((gap,36),f'UF chosen by nearest log-rate only (UF rate {mismatch:+.1f}% vs RouterVC); not exact-rate matching.',font=small,fill='#444444')
        for i,(name,pixels,metrics) in enumerate(panels):
            x=gap+i*(width+gap);draw.text((x,67),name,font=font,fill='black')
            if metrics:
                draw.text((x,93),f"bpp {metrics['bpp']:.6f} | LPIPS {metrics['lpips_alex']:.4f}",font=small,fill='black')
                draw.text((x,116),f"PSNR {metrics['psnr_db']:.2f} dB",font=small,fill='black')
            view=Image.fromarray(pixels).resize((width,height),Image.Resampling.LANCZOS)
            canvas.paste(view,(x,145))
            draw.rectangle((x+box[0]*width/source.shape[1],145+box[1]*height/source.shape[0],
                            x+box[2]*width/source.shape[1],145+box[3]*height/source.shape[0]),outline='#edab3a',width=2)
            draw.text((x,151+height),'Same fixed center detail (not selected for gain)',font=small,fill='#666666')
            canvas.paste(Image.fromarray(pixels).crop(box).resize((width,zoom_h),Image.Resampling.NEAREST),(x,180+height))
        path=folder/f'{sid}.png';canvas.save(path)
        records.append(dict(sample_id=sid,dataset=row['dataset'],image=str(path),frame=8,
            center_box=list(box),router_point=row['point'],uf_point=uf['point'],uf_rate_difference_percent=mismatch,
            router_bpp=row['bpp'],uf_bpp=uf['bpp'],selection='all 13 windows; E50; fixed middle frame and center crop; nearest log-rate UF, no quality selection'))
        print('QUALITATIVE',sid,flush=True)
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();output=args.output or args.root/'overview_20261007'
    if not os.environ.get('TMUX'):
        raise RuntimeError('overview requires tmux')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    done=output/'complete.json'
    if done.exists():
        receipt=read(done)
        if receipt['code_sha256']!=digest(__file__):raise ValueError('overview code changed')
        verify_artifacts(Path('/'),receipt['inputs']);verify_artifacts(output,receipt['artifacts'])
        print('OVERVIEW_VERIFIED');return
    output.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    data,protocol,resources,inputs=collect(args.root)
    groups=rd_figure(data,output);resource_groups=resource_figures(resources,output)
    pictures=qualitative(data,protocol,output)
    fields=('dataset','sample_id','point','bpp','bytes','lpips_alex','psnr_db','temporal_delta_mae')
    with (output/'all_points.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        writer.writerows(data['rows']+data['baselines'])
    save(output/'summary.json',dict(rd=groups,resources=resource_groups,resource_rows=resources,
        pictures=pictures,full_benchmark=False,semantic_protection_validated=False,
        new_training=False,new_inference=False,
        rate_scope='native UF stream only (SPS/NAL included, audit JSON excluded); RouterVC and full-G all container bytes included',
        timing_scope='current fresh worker including model loads; UF/full-G historical fresh workers; no complete source encoding timing',
        peak_scope='CUDA allocated, max over available worker and per-G-call counters; not total device memory'))
    artifacts={str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()}
    save(done,dict(complete=True,code_sha256=digest(__file__),inputs=inputs,artifacts=artifacts))
    print('OVERVIEW_COMPLETE',flush=True)


if __name__=='__main__':
    main()
