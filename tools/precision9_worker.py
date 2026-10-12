"""One resumable source-side width encoding/planning job; no G trial calls."""
import argparse
import gc
import os
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import torch
from demo.chunk_enhancement_codec import configure_torch
from demo.chunk_enhancement_experiment import Run
from demo.routervc_fullview_probe import read, save, digest, immutable, verify_artifacts
from demo.scalable_codec import atomic_bytes
from demo.scalable_format import frame_hash
from demo.four_state_receive import codec_precision
from demo.routervc_sender_router import prefix_under_budget
from routervc.precision9 import codec, format as packets, stream, sender, budget
from tools.cooperative_review import source_rgb


def work(args, run):
    from tools.precision9_pilot import check_protocol, RATIOS
    protocol = check_protocol(args.root)
    row = next(r for r in protocol['rows'] if r['sample_id']==args.sample_id)
    folder = args.root/'banks'/row['sample_id']/f'w{args.width}'
    folder.mkdir(parents=True,exist_ok=True)
    binding = dict(row=row, width=args.width, profile=codec.identity(), sender=protocol['sender'],
        teacher=protocol['teacher'], ratios=list(RATIOS), protocol=digest(args.root/'protocol.json'))
    immutable(folder/'binding.json', binding)
    if (folder/'complete.json').exists():
        verify_artifacts(folder,read(folder/'complete.json')['artifacts']); return
    source = source_rgb(row); started = time.monotonic()
    if (folder/'encoded.json').exists():
        encoded = read(folder/'encoded.json'); verify_artifacts(folder,encoded['artifacts'])
        bank = (folder/'bank.rvlp').read_bytes()
    else:
        run.update(phase='encode_own_reference_chain', sample=row['sample_id'], width=args.width)
        with codec_precision():
            engine = codec.PrecisionCodec()
            bank, base, full, detail = engine.encode(source,args.width,run.check)
            refs = detail['base_reference_hashes']
            for indices in ([],[0,1,4,5],list(range(16))):
                decoded_b,decoded_y,report = engine.decode(packets.subset(bank,indices))
                np.testing.assert_array_equal(decoded_b,base)
                if report['base_reference_hashes'] != refs: raise ValueError('E changed B reference')
                if not indices: np.testing.assert_array_equal(decoded_y,base)
                if len(indices)==16: np.testing.assert_array_equal(decoded_y,full)
            del engine
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.set_stream(torch.cuda.default_stream())
        parity = None
        if args.width == 3:
            from routervc.latent import packet_format as old_packets, chain_format as old_chain
            from routervc.latent.packet_codec import packet_hash
            if digest(row['bank_path']) != row['bank_sha256']: raise ValueError('legacy bank changed')
            old_bank = Path(row['bank_path']).read_bytes()
            before,after = old_packets.parse(old_bank), packets.parse(bank)
            old_inner,new_inner = old_chain.parse(before['base']),packets.parse_inner(after['base'])
            if (before['profile'] != packet_hash() or before['packets'] != after['packets']
                    or old_inner['intra'] != new_inner['intra'] or old_inner['chunks'] != new_inner['chunks']
                    or len(bank) != len(old_bank) or before['base_end'] != after['base_end']):
                raise ValueError('width3 payload/framing parity failed')
            if digest(row['encoded_path']) != row['encoded_sha256']: raise ValueError('legacy receipt changed')
            old = read(row['encoded_path'])
            if old['base_hash'] != frame_hash(base) or old['base_reference_hashes'] != refs:
                raise ValueError('width3 base/reference parity failed')
            parity = dict(payloads_exact=True, byte_counts_exact=True, base_pixels_exact=True, references_exact=True)
        atomic_bytes(folder/'bank.rvlp',bank)
        costs = packets.bundle_bytes(bank); base_bytes = len(packets.subset(bank,[]))+stream.HEADER_BYTES
        encoded = dict(complete=True,**detail,base_hash=frame_hash(base), full_hash=frame_hash(full),
            source_hash=frame_hash(source), base_bytes=base_bytes, full_e_bytes=sum(costs), width3_legacy_parity=parity,
            sender_features=budget.features(source,base,base_bytes,sum(costs)),
            seconds=time.monotonic()-started, artifacts={'bank.rvlp':digest(folder/'bank.rvlp')})
        save(folder/'encoded.json',encoded)
    if (folder/'order.json').exists():
        order = read(folder/'order.json')
        if order['binding'] != binding: raise ValueError('order binding differs')
    else:
        model,_ = sender.load(Path(protocol['sender']['path']))
        order = dict(binding=binding, **sender.ordering(source,bank,model,protocol['teacher'],
            check=run.check,progress=run.update))
        if order['base_hash'] != encoded['base_hash'] or order['full_hash'] != encoded['full_hash']:
            raise ValueError('planner decoder endpoint differs')
        save(folder/'order.json',order); del model
    costs = np.asarray(order['bundle_bytes'],np.int64); teacher = protocol['teacher']; points = []; previous = None
    for cap in RATIOS:
        prefix = prefix_under_budget(order['order'],costs,int(sum(costs)*cap))
        raw = stream.wrap(packets.subset(bank,prefix['indices']),teacher['receiver']['sha256'],
                          teacher['G_assets_hash'],teacher['max_g'],teacher['seed'])
        if previous is not None and not raw.startswith(previous): raise ValueError('literal prefix lost')
        previous = raw; name = f'e{round(cap*100):03d}.rvlw'; atomic_bytes(folder/name,raw)
        if len(raw) != encoded['base_bytes']+prefix['packet_bytes']: raise ValueError('byte count differs')
        points.append(dict(cap=cap,file=name,actual_bytes=len(raw),**prefix))
    names = ['binding.json','encoded.json','bank.rvlp','order.json',*(p['file'] for p in points)]
    save(folder/'complete.json',dict(complete=True,points=points,width=args.width,
        no_optimization=True,zero_shot_from_width3=True,artifacts={n:digest(folder/n) for n in names}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--sample-id',required=True)
    parser.add_argument('--width',type=int,choices=(9,),required=True)
    args = parser.parse_args()
    if not os.environ.get('TMUX'): raise RuntimeError('tmux required')
    configure_torch()
    run = Run(SimpleNamespace(output=args.root/'workers'/args.sample_id/f'w{args.width}',command='prepare',max_hours=12))
    run.thread.start()
    try: work(args,run)
    except BaseException as error:
        save(run.root/'last_failure.json',dict(error=repr(error),progress=run.progress)); raise
    finally:
        run.log_resources();run.stop.set();run.thread.join(timeout=3);run.lock.close()


if __name__ == '__main__': main()
