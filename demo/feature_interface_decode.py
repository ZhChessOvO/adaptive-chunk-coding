"""New pinned receiver profile; all feature perturbations are in hashed bundles."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from demo import feature_condition_decode as parent
from demo.feature_interface_model import FeatureInterface, condition_statistics, validate_bundle
from demo.scalable_codec import atomic_json, file_hash

PARENT_IDENTITIES=parent.identities
PARENT_RESTORE=parent.restore
PARENT_CONDITION=parent.conditioned_latent


def identities(adapter):
    result=PARENT_IDENTITIES(adapter)
    spec=dict(parent=result['profile'],receiver=file_hash(Path(__file__)),
              interface=file_hash(REPO/'demo/feature_interface_model.py'))
    result['profile']=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
    return result


def restore(enhanced,control,adapter,packets):
    bundle=torch.load(adapter,weights_only=True,map_location='cpu')
    mode=validate_bundle(bundle); stats=[]
    def condition(net,raw,received,**kwargs):
        effective,side,coverage=PARENT_CONDITION(net,raw,received,**kwargs)
        stats.append(condition_statistics(raw,effective,side,coverage))
        return effective,side,coverage
    with patch.object(parent,'FeatureCondition',lambda:FeatureInterface(mode)), \
            patch.object(parent,'conditioned_latent',condition):
        pixels,report=PARENT_RESTORE(enhanced,control,adapter,packets)
    report.update(interface_mode=mode,condition_statistics=stats)
    return pixels,report


def decode(args):
    with patch.object(parent,'identities',identities),patch.object(parent,'restore',restore):
        parent.decode(args)


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--stream',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--adapter',type=Path,required=True)
    p.add_argument('--disable-generation',action='store_true')
    decode(p.parse_args())
