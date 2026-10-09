"""Budget-independent conditional E ordering on actual decoded latent states."""
import hashlib
from pathlib import Path
import time

import numpy as np

from demo import routervc_sender_router as network
from demo.routervc_fullview_probe import read, digest, save, immutable, verify_artifacts
from demo.scalable_codec import atomic_bytes
from demo.scalable_format import frame_hash
from routervc.latent import routing, router_data, sender_data


def load(path, *, smoke=False):
    model,payload=network.load_model(path)
    protocol=payload['binding']['protocol']
    if (protocol.get('revision')!='new_latent_final_marginal_v1'
            or protocol['teacher']['receiver']['sha256']!=sender_data.RECEIVER_SHA
            or protocol['receiver_profile']!=routing.identity()
            or (protocol['smoke'] and not smoke)):
        raise ValueError('sender must be adapted to the explicitly selected new-latent receiver')
    for name,expected in protocol['code'].items():
        if digest(router_data.REPO/name)!=expected:
            raise ValueError('sender training dependency changed: '+name)
    return model,payload


def ordering(source,bank,model,teacher,*,decode=router_data.decode,predict=network.predict,
             check=lambda:None,progress=lambda **kw:None):
    """Recompute Y and R_s after each bundle; never execute G or use GT scores."""
    began=time.monotonic()
    base,full,detail=decode(bank)
    base_hash=frame_hash(base)
    costs=np.asarray(routing.bundle_bytes(bank),dtype=np.int64)
    chosen,steps=[],[]
    for _ in range(16):
        check()
        inner=routing.subset(bank,chosen)
        b,y,current=decode(inner)
        if frame_hash(b)!=base_hash or current['base_reference_hashes']!=detail['base_reference_hashes']:
            raise ValueError('sender mixed decode changed B reference')
        cov=routing.coverage(routing.packets.parse(inner))
        gains=predict(model,source,b,y,full,cov,costs,teacher['max_g'])[0].numpy()
        ranked=network.rank_candidates(gains,costs,cov)
        steps.append(dict(selected=chosen.copy(),received_hash=frame_hash(y),
            gains=gains.tolist(),ranked=ranked,actual_input_bytes=len(inner)))
        if not ranked:
            break
        chosen.append(ranked[0])
        progress(phase='source_sender_order',selected_regions=len(chosen))
    return dict(order=chosen,steps=steps,bundle_bytes=costs.tolist(),
        base_hash=base_hash,full_hash=frame_hash(full),seconds=time.monotonic()-began,
        G_executed_during_planning=False,actual_mixed_Y_decoded=True,
        source_visible_to_sender=True,mask_bytes=0,
        scope='conditional positive-gain prefix; bank preparation excluded from this timing')


def write_prefixes(output,source,bank,checkpoint,*,smoke=False,check=lambda:None,progress=lambda **kw:None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    model,payload=load(checkpoint,smoke=smoke)
    teacher=payload['binding']['protocol']['teacher']
    binding=dict(source_hash=frame_hash(source),bank_sha256=hashlib.sha256(bank).hexdigest(),
        sender_sha256=digest(checkpoint),receiver=teacher['receiver'],code=digest(Path(__file__)),
        ratios=[0.,.25,.5,1.])
    immutable(output/'protocol.json',binding)
    if (output/'complete.json').exists():
        done=read(output/'complete.json');verify_artifacts(output,done['artifacts']);return done
    if (output/'order.json').exists():
        record=read(output/'order.json')
        if record['binding']!=binding:raise ValueError('cached sender ordering changed')
    else:
        record=dict(binding=binding,**ordering(source,bank,model,teacher,check=check,progress=progress))
        save(output/'order.json',record)
    total=sum(record['bundle_bytes']);points=[];previous=None
    for ratio in binding['ratios']:
        budget=int(total*ratio)
        prefix=network.prefix_under_budget(record['order'],np.asarray(record['bundle_bytes'],np.int64),budget)
        inner=routing.subset(bank,prefix['indices'])
        wire=routing.wrap(inner,teacher['receiver']['sha256'],teacher['G_assets_hash'],
                          max_g=teacher['max_g'],seed=teacher['seed'])
        if previous is not None and not wire.startswith(previous):raise ValueError('sender lost byte prefix')
        previous=wire
        name=f'e{int(100*ratio):03d}.rvlrg';atomic_bytes(output/name,wire)
        if len(wire)!=routing.HEADER_BYTES+len(routing.subset(bank,[]))+prefix['packet_bytes']:
            raise ValueError('actual additional packet bytes differ')
        points.append(dict(ratio=ratio,file=name,actual_bytes=len(wire),**prefix))
    done=dict(complete=True,points=points,order=record['order'],
        actual_bytes=True,mask_bytes=0,receiver_needs_sender=False,
        artifacts={n:digest(output/n) for n in ['protocol.json','order.json',*(p['file'] for p in points)]})
    save(output/'complete.json',done);return done
