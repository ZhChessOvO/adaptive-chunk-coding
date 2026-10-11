"""P2b conditional source-aware encoding; receiver requires no sender weights."""
import hashlib
from pathlib import Path
import numpy as np
from demo import routervc_sender_router as network
from demo.routervc_fullview_probe import read, save, digest, immutable, verify_artifacts
from demo.scalable_codec import atomic_bytes
from demo.scalable_format import frame_hash
from routervc.latent import routing, sender as previous
from routervc.cooperation import sender_data as data, stream


def load(path, *, smoke=False):
    model, payload = network.load_model(path)
    protocol = payload['binding']['protocol']
    if (protocol.get('format') != data.FORMAT or protocol['receiver_profile'] != stream.identity()
            or protocol['teacher']['receiver']['sha256'] != data.RECEIVER_SHA
            or protocol['smoke'] and not smoke):
        raise ValueError('sender must bind reviewed P2 and fixed multiband')
    for name, expected in protocol['code'].items():
        if digest(data.REPO/name) != expected: raise ValueError('sender dependency changed: '+name)
    return model, payload


def write_prefixes(output, source, bank, checkpoint, *, smoke=False, check=lambda:None, progress=lambda **kw:None):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    model, payload = load(checkpoint, smoke=smoke); teacher = payload['binding']['protocol']['teacher']
    binding = dict(source_hash=frame_hash(source), bank_sha256=hashlib.sha256(bank).hexdigest(),
        sender_sha256=digest(checkpoint), receiver=teacher['receiver'], code=digest(Path(__file__)),
        profile=stream.identity(), ratios=[0., .25, .5, 1.])
    immutable(output/'protocol.json', binding)
    if (output/'complete.json').exists():
        done = read(output/'complete.json'); verify_artifacts(output, done['artifacts']); return done
    if (output/'order.json').exists():
        record = read(output/'order.json')
        if record['binding'] != binding: raise ValueError('cached sender order changed')
    else:
        record = dict(binding=binding, **previous.ordering(source, bank, model, teacher, check=check, progress=progress))
        record['source_visible_to_sender'] = not model.config.zero_source
        save(output/'order.json', record)
    total = sum(record['bundle_bytes']); points = []; prior = None
    for ratio in binding['ratios']:
        prefix = network.prefix_under_budget(record['order'], np.asarray(record['bundle_bytes'], np.int64), int(total*ratio))
        inner = routing.subset(bank, prefix['indices'])
        wire = stream.wrap(inner, teacher['receiver']['sha256'], teacher['G_assets_hash'], teacher['max_g'], teacher['seed'])
        if prior is not None and not wire.startswith(prior): raise ValueError('literal byte prefix lost')
        prior = wire; name = f'e{round(100*ratio):03d}.rvlcoop'; atomic_bytes(output/name, wire)
        if len(wire) != stream.HEADER_BYTES+len(routing.subset(bank, []))+prefix['packet_bytes']:
            raise ValueError('actual transmitted bytes differ')
        points.append(dict(ratio=ratio, file=name, actual_bytes=len(wire), **prefix))
    done = dict(complete=True, points=points, order=record['order'], receiver_needs_sender=False,
        mask_bytes=0, actual_bytes=True, G_executed_during_planning=False,
        artifacts={n:digest(output/n) for n in ['protocol.json', 'order.json', *(p['file'] for p in points)]})
    save(output/'complete.json', done); return done
