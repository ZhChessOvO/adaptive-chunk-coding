"""Zero-shot source-aware ordering with the fixed P2b checkpoint."""
import time
import numpy as np
from demo import routervc_sender_router as network
from demo.scalable_format import frame_hash
from routervc.latent import routing
from routervc.precision9 import codec, format as packets
from routervc.cooperation.sender import load


def ordering(source,bank,model,teacher,*,decode=codec.decode,predict=network.predict,
             check=lambda:None,progress=lambda **kw:None):
    """Recompute Y and R_s after each bundle; never execute G or use GT scores."""
    began=time.monotonic()
    base,full,detail=decode(bank)
    base_hash=frame_hash(base)
    costs=np.asarray(packets.bundle_bytes(bank),dtype=np.int64)
    chosen,steps=[],[]
    for _ in range(16):
        check()
        inner=packets.subset(bank,chosen)
        b,y,current=decode(inner)
        if frame_hash(b)!=base_hash or current['base_reference_hashes']!=detail['base_reference_hashes']:
            raise ValueError('sender mixed decode changed B reference')
        cov=routing.coverage(packets.parse(inner))
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
