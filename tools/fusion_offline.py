"""Frozen-policy learned-F evaluation on cached raw G; no new generation here."""
import argparse
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from demo.chunk_enhancement_experiment import Run
from demo.chunk_enhancement_codec import configure_torch
from demo.routervc_fullview_probe import read, save, digest, immutable, verify_artifacts
from demo.scalable_codec import atomic_npz, atomic_bytes
from demo.scalable_format import frame_hash
from demo.scalable_experiment import quality
from demo.stage_c_three_path_roi_probe import LPIPSAlex
from routervc.fusion import stream
from routervc.fusion.receive import load_fusion, learned_pixels
from routervc.fusion.boundaries import edges, CATEGORIES, measure, aggregate
from tools.latent_boundary_report import ROOT, EVALUATION
from tools.fusion_pilot import load_capture


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT/'p1_evaluation')
    p.add_argument('--checkpoint', type=Path, default=ROOT/'p1_fit/best.pt')
    a = p.parse_args()
    if not os.environ.get('TMUX'): raise RuntimeError('tmux required')
    configure_torch(); torch.use_deterministic_algorithms(True)
    run = Run(SimpleNamespace(output=a.output, command='offline', max_hours=12)); run.thread.start()
    try:
        rows = read(EVALUATION/'summary.json')['scope']['rows']; sha = digest(a.checkpoint)
        immutable(run.root/'protocol.json', dict(checkpoint=str(a.checkpoint), checkpoint_sha256=sha,
            rows=rows, fusion_profile=stream.identity(), modes=list(stream.MODES), E_cap=.5,
            extra_header_bytes=stream.HEADER_BYTES, offline_code=digest(Path(__file__))))
        model = load_fusion(a.checkpoint, sha); metric = LPIPSAlex(True)
        records = []
        for index, row in enumerate(rows):
            run.check(); sid = row['sample_id']; out = run.root/'samples'/sid; out.mkdir(parents=True, exist_ok=True)
            run.update(phase='offline_fusion', completed=index, total=len(rows), sample=sid)
            if (out/'complete.json').exists():
                result = read(out/'complete.json'); verify_artifacts(out, result['artifacts'])
                records.append(result); continue
            cache = ROOT/'p1_controls/samples'/sid
            base, received, current, patches, captured = load_capture(cache)
            paired = read(cache/'controls/complete.json'); verify_artifacts(cache/'controls', paired['artifacts'])
            with np.load(cache/'controls/controls.npz') as z: multiband = z['multiband']
            detail = read(cache/'received.json')['detail']
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start = time.monotonic()
            pixels = learned_pixels(model, base, received, current, multiband,
                                    detail['received_regions'], captured['generated'])
            torch.cuda.synchronize(); seconds = time.monotonic()-start
            peak = torch.cuda.max_memory_allocated()
            with np.load(row['source_path']) as z: source = z['source']
            if digest(Path(row['source_path'])) != row['source_sha256']: raise ValueError('source changed')
            boundary = edges(base.shape, detail['received_regions'], captured['generated'])
            variants = dict(paired['variants'])
            variants['learned'] = dict(quality=quality(source, pixels, metric),
                boundaries={c:aggregate([measure(source, pixels, e) for e in boundary if e['category'] == c])
                            for c in CATEGORIES})
            atomic_npz(out/'learned.npz', pixels=pixels)
            old_wire = (EVALUATION/'samples'/sid/'source_e050/stream.rvlrg').read_bytes()
            for mode in stream.MODES:
                atomic_bytes(out/f'{mode}.rvlf', stream.wrap(old_wire, mode, sha if mode == 'learned' else '0'*64))
            # Verify literal prefixes using actual completed packets, not a mock.
            prefixes = [stream.wrap((EVALUATION/'samples'/sid/f'source_e{cap:03d}/stream.rvlrg').read_bytes(),
                                    'learned', sha) for cap in (0, 25, 50, 100)]
            if any(not b.startswith(a) for a,b in zip(prefixes, prefixes[1:])): raise ValueError('F header broke prefixes')
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
            eligible = [e for e in boundary if e['category'] == 'G_G' and 8 in e['frames']]
            edge = eligible[0] if eligible else boundary[0]; pos=(edge['lo']+edge['hi'])//2
            x,y = (edge['pos']-48,pos-48) if edge['axis']=='x' else (pos-48,edge['pos']-48)
            for j,(name,v) in enumerate([('source',source),('current',current),('multiband',multiband),('learned',pixels)]):
                axs[0,j].imshow(v[8]); axs[0,j].axis('off'); axs[0,j].set_title(name)
                axs[1,j].imshow(v[8,y:y+96,x:x+96]); axs[1,j].axis('off')
            fig.suptitle(sid+' | fixed frame 9, first G/G edge | same packets and G')
            fig.savefig(out/'comparison.png',dpi=140); plt.close(fig)
            names=['learned.npz','comparison.png']+[f'{m}.rvlf' for m in stream.MODES]
            result=dict(complete=True,sample_id=sid,dataset=row['dataset'],variants=variants,
                actual_bytes=(out/'learned.rvlf').stat().st_size, old_bytes=len(old_wire),
                bpp=(out/'learned.rvlf').stat().st_size*8/np.prod(base.shape[:3]),
                additional_header_bytes=stream.HEADER_BYTES,additional_mask_bytes=0,
                output_hash=frame_hash(pixels),base_hash=frame_hash(base),enhanced_hash=frame_hash(received),
                generated=captured['generated'],literal_prefix_pairs=3,
                learned_seconds_cached_multiband_no_model_load=seconds,
                learned_peak_cuda_allocated_bytes=peak,
                artifacts={n:digest(out/n) for n in names})
            save(out/'complete.json',result); records.append(result)
        groups={}
        for dataset in ('REDS','UVG'):
            subset=[r for r in records if r['dataset']==dataset]
            groups[dataset]={m:{k:float(np.mean([r['variants'][m]['quality'][k] for r in subset]))
                                for k in ('lpips_alex','psnr_db','temporal_delta_mae')} for m in stream.MODES}
        save(run.root/'summary.json',dict(complete=True,groups=groups,results=records))
        save(run.root/'offline.complete.json',dict(complete=True,points=len(records),
            fresh_decode_pending=True,artifacts={n:digest(run.root/n) for n in ('protocol.json','summary.json')}))
        run.update(phase='offline_complete',completed=len(rows));run.log_resources()
    finally:run.stop.set();run.thread.join();run.lock.close()


if __name__=='__main__':main()
