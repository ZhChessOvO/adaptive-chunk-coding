"""Spatial refinement of native residual symbols, no G or trained Router yet.

Entropy probabilities depend only on B. Spatial conditional means are replayed
on the ACTUALLY received mixed coarse/fine representation, not a hidden full y.
A tile's complete symbols do not imply native-exact pixels inside that tile:
UF's spatial prediction and synthesis couple neighboring tiles. Full coverage
does preserve the native endpoint. Display-only E never advances the B reference.
"""
import hashlib
from pathlib import Path

import numpy as np

from demo.scalable_codec import file_hash
from routervc.latent.codec import LatentCodec, profile_hash
from routervc.latent import entropy, format as base_format, regional_format as wire
from routervc.latent.split import restore_symbols, coarse_representatives


def regional_hash():
    files = ('regional_format.py', 'regional_codec.py')
    return hashlib.sha256((profile_hash() + ''.join(
        file_hash(Path(__file__).with_name(n)) for n in files)).encode()).hexdigest()


class RegionalCodec(LatentCodec):
    def _base(self, data):
        """All probability/context inputs recovered from received B, no source."""
        import torch
        from demo.stage_c_three_path_roi_probe import rgb_from_tensor
        parsed = base_format.parse(data)
        if parsed.identities != self.identities or parsed.qp != 48 or parsed.width != 3:
            raise ValueError('wrong regional base identity/configuration')
        height, width = parsed.height, parsed.image_width
        torch.cuda.set_stream(self.codec.stream)
        intra = self.codec.i_net.decompress(parsed.intra,
            {'height': height, 'width': width}, 32, parsed.i_parallel)['x_hat'].clone()
        self.codec.p_net.add_ref_feature_from_frame(intra)
        proxy = self.codec.p_net.proxy
        context = self.bridge.context(proxy)
        z = entropy.decode_z(parsed.z, (1, 128, (height+63)//64, (width+63)//64),
                             self.codec.p_net.bit_estimator_z.get_cdf_info(), 48)
        zg = self.tensor(z)
        scales = self.bridge.prior(proxy, zg, 48, context)['scales'].cpu().numpy()
        coarse = entropy.decode_coarse(parsed.coarse, scales, 3)
        return parsed, context, zg, scales, coarse, rgb_from_tensor(intra, height, width)

    def repacketize(self, full):
        """Reuse the authenticated one-encoding stream, not a new high-QP encode."""
        import torch
        parsed = base_format.parse(full)
        if parsed.fine is None:
            raise ValueError('full endpoint stream required to partition E')
        inner = full[:parsed.base_end]
        with torch.inference_mode():
            _, _, _, scales, coarse, _ = self._base(inner)
            fine = entropy.decode_fine(parsed.fine, coarse, scales, 3)
            base = wire.base_stream(inner, regional_hash())
            identity = wire.digest(base)
            packets = []
            for index in range(16):
                region = wire.region_slice(coarse.shape, index)
                payload = entropy.encode_fine(coarse[region], fine[region], scales[region], 3)
                packets.append(wire.packet(index, payload, identity))
        return base, packets

    def decode_regional(self, data, *, allow_incomplete_tail=False):
        import torch
        from tools.latent_probe import context_exact
        parsed = wire.parse(data, allow_incomplete_tail=allow_incomplete_tail)
        if parsed['profile'] != regional_hash():
            raise ValueError('regional profile mismatch')
        with torch.inference_mode():
            inner, context, z, scales, coarse, intra = self._base(parsed['base'])
            h, w = inner.height, inner.image_width
            proxy = self.codec.p_net.proxy
            mixed = coarse_representatives(coarse, 3).copy()
            base = self.bridge.synthesize(proxy, z, self.tensor(mixed), 48, context)
            base_rgb = self.rgb(base, h, w)
            for index, payload in parsed['regions'].items():
                region = wire.region_slice(coarse.shape, index)
                fine = entropy.decode_fine(payload, coarse[region], scales[region], 3)
                mixed[region] = restore_symbols(coarse[region], fine, 3)
            output = base_rgb
            if parsed['regions']:
                displayed = self.bridge.synthesize(proxy, z, self.tensor(mixed), 48, context)
                output = self.rgb(displayed, h, w)
            if not context_exact(self.bridge, proxy, context):
                raise RuntimeError('regional display changed B reference')
            torch.cuda.synchronize()
        return np.concatenate([intra[None], base_rgb]), np.concatenate([intra[None], output]), dict(
            frame_count=9, source_frames_read=False, base_reference_unchanged=True,
            received_regions=sorted(parsed['regions']), additional_mask_bytes=0,
            generation=False, router=False, base_bytes=parsed['base_end'],
            actual_bytes=len(data), bpp=len(data)*8/(9*h*w),
            inner_base_bytes=len(parsed['base']), region_wrapper_bytes=wire.HEADER.size,
            E_wire_bytes=len(data)-parsed['base_end']-parsed['ignored_tail_bytes'],
            E_unique_payload_bytes=sum(map(len, parsed['regions'].values())),
            E_header_bytes=(len(parsed['regions'])+parsed['duplicates'])*wire.EHEADER.size,
            duplicates=parsed['duplicates'], ignored_tail_bytes=parsed['ignored_tail_bytes'],
            scope='single I32+P8 QP48; width3; fixed 4x4 compact-latent tiles, spatially coupled')
