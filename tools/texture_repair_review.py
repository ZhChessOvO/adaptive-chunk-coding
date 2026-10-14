"""CPU-only temporal phase audit: distinguish removal from phase cancellation."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import gaussian_filter

from demo.routervc_fullview_probe import read, immutable, digest, verify_artifacts
from demo.scalable_codec import atomic_json
from tools.texture_diagnostic import core


def temporal_phase(pixels, reference, period=16):
    error=pixels.astype(np.float32)-reference.astype(np.float32)
    error=error[:,16:-16,16:-16]
    error-=gaussian_filter(error,sigma=(0,3,3,0))
    h,w=(error.shape[1]//period)*period,(error.shape[2]//period)*period
    if min(h,w)<period: raise ValueError('insufficient phase cells')
    folded=error[:,:h,:w].reshape(len(error),h//period,period,w//period,period,3).mean((1,3))
    folded-=folded.mean((1,2),keepdims=True)
    mean=folded.mean(0)
    return dict(period=period,
        stationary_rms=float(np.sqrt(np.mean(mean**2))),
        per_frame_rms_mean=float(np.sqrt(np.mean(folded**2,axis=(1,2,3))).mean()),
        phase_time_std=float(np.sqrt(np.mean((folded-mean)**2))),
        phase_frame_difference=float(np.sqrt(np.mean(np.diff(folded,axis=0)**2))))


def contact_sheet(root,item,final):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    case=item['case']; folder=root/case['sample_id']
    with np.load(folder/'inputs.npz') as z: source=core(z['source'],case)
    panels=[('Source',source)]
    for name in ('initial',f'control_{final:06d}',f'repair_{final:06d}'):
        with np.load(folder/name/'output.npz') as z: panels.append((name,core(z['pixels'],case)))
    fig,axes=plt.subplots(4,5,figsize=(13,10),layout='constrained')
    for i,(name,value) in enumerate(panels):
        for j,t in enumerate((0,4,8,12,16)):
            ax=axes[i,j];ax.imshow(value[t],interpolation='nearest');ax.axis('off')
            ax.set_title(f'{name} / frame {t+1}',fontsize=9)
    fig.suptitle(case['sample_id']+' | SAME ROI through time',fontsize=12)
    path=folder/'temporal_contact.png';fig.savefig(path,dpi=150);plt.close(fig)
    return str(path.relative_to(root))


def main(args):
    began=time.monotonic()
    while not (args.root/'evaluation/complete.json').exists():
        if not args.wait or time.monotonic()-began > 3600:
            raise RuntimeError('paired evaluation not complete')
        time.sleep(5)
    root=args.root/'evaluation'; done=read(root/'complete.json'); verify_artifacts(root,done['artifacts'])
    summary=read(root/'summary.json');final=read(args.root/'formal/protocol.json')['steps']
    immutable(root/'temporal_audit_protocol.json',dict(code=digest(Path(__file__)),
        input_complete=digest(root/'complete.json'),period=16,
        purpose='descriptive per-frame error-phase audit, not a validated quality score'))
    rows=[];artifacts={}
    for item in summary['cases']:
        case=item['case'];folder=root/case['sample_id']
        with np.load(folder/'inputs.npz') as z: source=core(z['source'],case)
        for result in item['results']:
            dest=folder/result['name'];verify_artifacts(dest,result['artifacts'])
            with np.load(dest/'output.npz') as z: pixels=core(z['pixels'],case)
            phase=temporal_phase(pixels,source)
            if not np.isclose(phase['stationary_rms'],result['error_phase16']['phase_rms_u8'],rtol=1e-5):
                raise ValueError('stationary phase implementation differs')
            rows.append(dict(sample=case['sample_id'],group=case['evaluation_group'],
                             dataset=case['dataset'],name=result['name'],**phase))
        if case['evaluation_group']=='known_artifact':
            name=contact_sheet(root,item,final);artifacts[name]=digest(root/name)
    atomic_json(root/'temporal_audit.json',dict(complete=True,rows=rows,artifacts=artifacts))
    print(json.dumps(dict(complete=True,points=len(rows),seconds=time.monotonic()-began)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--wait',action='store_true');main(p.parse_args())
