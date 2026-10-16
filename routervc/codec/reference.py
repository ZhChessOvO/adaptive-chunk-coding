"""Reference. Extracted from the validated RouterVC research implementation."""

from __future__ import annotations
import hashlib
from routervc.codec.backbone import LatentCodec
from routervc.codec.transition import load_transition


def chain_hash():
    from routervc.assets import code_identity

    return code_identity("codec")


def reference_hash(context):
    result = {}
    for name in ("memory", "ctx"):
        result[name] = hashlib.sha256(
            context[name].contiguous().cpu().numpy().tobytes()
        ).hexdigest()
    return result


class ChainCodec(LatentCodec):

    def __init__(self):
        super().__init__()
        self.transition = load_transition()
        self.chain_identities = (*self.identities, chain_hash())

    def advance(self, context, feature, reset):
        proxy = self.codec.p_net.proxy
        self.bridge.restore(proxy, context)
        self.transition.supply_base(proxy, feature, reset)
        decoder_state = self.bridge.context(proxy)
        self.transition.prepare_encoder(proxy)
        ready = self.bridge.context(proxy)
        return (ready, decoder_state)
