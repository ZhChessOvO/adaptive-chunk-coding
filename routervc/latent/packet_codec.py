"""Compact regional refinement, shared by single-P8 and continuous B chains."""
import hashlib
from pathlib import Path

import numpy as np
from demo.scalable_codec import file_hash
from routervc.latent.chain_codec import ChainCodec, chain_hash, reference_hash
from routervc.latent.regional_codec import RegionalCodec, regional_hash
from routervc.latent import entropy, compact_entropy as compact, packet_format as wire
from routervc.latent import format as single, chain_format as chain
from routervc.latent.split import restore_symbols, coarse_representatives


def packet_hash():
    names = ('compact_rans.cpp', 'compact_entropy.py', 'packet_format.py', 'packet_codec.py')
    return hashlib.sha256((chain_hash()+regional_hash()+''.join(
        file_hash(Path(__file__).with_name(n)) for n in names)).encode()).hexdigest()


class PacketCodec(ChainCodec, RegionalCodec):
    def _states(self, inner, kind, refs):
        """Yield B-derived conditions; commit ONLY B after each display callback."""
        import torch
        from demo.stage_c_three_path_roi_probe import rgb_from_tensor
        from tools.latent_probe import context_exact
        if kind == 1:
            p, ctx, zg, scales, c, intra = self._base(inner)
            yield dict(index=-1, rgb=intra)
            groups = [(p.z, p.coarse)]
            h, w, count, ready = p.height, p.image_width, 9, ctx
        else:
            p = chain.parse(inner)
            m = p['meta']
            if tuple(m['identities']) != self.chain_identities:
                raise ValueError('continuous B identity mismatch')
            h, w, count = m['height'], m['width'], m['frames']
            torch.cuda.set_stream(self.codec.stream)
            intra = self.codec.i_net.decompress(p['intra'], {'height': h, 'width': w},
                                               32, m['I_parallel'])['x_hat'].clone()
            self.codec.p_net.add_ref_feature_from_frame(intra)
            ready = self.bridge.context(self.codec.p_net.proxy)
            yield dict(index=-1, rgb=rgb_from_tensor(intra, h, w))
            groups = p['chunks']
        proxy = self.codec.p_net.proxy
        for index, (zd, yd) in enumerate(groups):
            start = 1+8*index
            valid = min(8, count-start)
            z = entropy.decode_z(zd, (1, 128, (h+63)//64, (w+63)//64),
                                self.codec.p_net.bit_estimator_z.get_cdf_info(), 48)
            zg = self.tensor(z)
            scales = self.bridge.prior(proxy, zg, 48, ready)['scales'].cpu().numpy()
            c = entropy.decode_coarse(yd, scales, 3)
            bottom = self.bridge.synthesize(proxy, zg, self.tensor(coarse_representatives(c, 3)), 48, ready)
            yield dict(index=index, c=c, scales=scales, z=zg, context=ready,
                       rgb=self.rgb(bottom, h, w)[:valid], h=h, w=w, valid=valid)
            if not context_exact(self.bridge, proxy, ready):
                raise RuntimeError('regional display modified B reference')
            if kind == 2:
                ready, _ = self.advance(ready, bottom['feature'], (start+8)%32 == 1)
                refs.append(reference_hash(ready))

    def repacketize(self, data, kind):
        import torch
        if kind == 1:
            p = single.parse(data)
            if p.fine is None:
                raise ValueError('full E required')
            inner, fine = data[:p.base_end], {0: p.fine}
        else:
            p = chain.parse(data)
            if set(p['enhancements']) != set(range(len(p['chunks']))):
                raise ValueError('full chunk E required')
            inner, fine = data[:p['base_end']], p['enhancements']
        base = wire.base_stream(inner, kind, packet_hash())
        packets, audit, refs = {}, [], []
        with torch.inference_mode():
            for s in self._states(inner, kind, refs):
                j = s['index']
                if j < 0:
                    continue
                c, scales = s['c'], s['scales']
                r = entropy.decode_fine(fine[j], c, scales, 3)
                for k in range(16):
                    roi = wire.region_slice(c.shape, k)
                    cs, rs, ss = c[roi], r[roi], scales[roi]
                    payload = compact.encode(cs, rs, ss)
                    np.testing.assert_array_equal(compact.decode(payload, cs, ss), rs)
                    old = entropy.encode_fine(cs, rs, ss, 3)
                    packets[j, k] = wire.packet(j, k, payload, wire.digest(base))
                    audit.append(dict(chunk=j, region=k, old_payload_bytes=len(old),
                        new_payload_bytes=len(payload), symbols=int(cs.size),
                        **compact.old_group_overhead(cs)))
        return base, packets, dict(regions=audit, base_reference_hashes=refs)

    def decode_packets(self, data, *, allow_incomplete_tail=False):
        import torch
        p = wire.parse(data, allow_incomplete_tail=allow_incomplete_tail)
        if p['profile'] != packet_hash():
            raise ValueError('compact packet profile mismatch')
        base, out, refs = [], [], []
        with torch.inference_mode():
            for s in self._states(p['base'], p['kind'], refs):
                j = s['index']
                if j < 0:
                    base.append(s['rgb']); out.append(s['rgb'])
                    continue
                c, scales = s['c'], s['scales']
                mixed = coarse_representatives(c, 3).copy()
                received = [(k, body) for (ch, k), body in p['packets'].items() if ch == j]
                for k, body in received:
                    roi = wire.region_slice(c.shape, k)
                    r = compact.decode(body, c[roi], scales[roi])
                    mixed[roi] = restore_symbols(c[roi], r, 3)
                shown = s['rgb']
                if received:
                    full = self.bridge.synthesize(self.codec.p_net.proxy, s['z'], self.tensor(mixed), 48, s['context'])
                    shown = self.rgb(full, s['h'], s['w'])[:s['valid']]
                base.extend(s['rgb']); out.extend(shown)
            torch.cuda.synchronize()
        info = p['info']
        return np.stack(base), np.stack(out), dict(source_frames_read=False, generation=False, router=False,
            additional_mask_bytes=0, base_reference_unchanged=True, base_reference_hashes=refs,
            received_regions=[list(k) for k in sorted(p['packets'])], frame_count=info['frames'],
            actual_bytes=len(data), bpp=len(data)*8/(info['frames']*info['height']*info['width']),
            base_bytes=p['base_end'], inner_base_bytes=len(p['base']), wrapper_bytes=wire.HEADER.size,
            E_wire_bytes=len(data)-p['base_end']-p['ignored_tail_bytes'],
            E_unique_payload_bytes=sum(map(len, p['packets'].values())),
            E_header_bytes=(len(p['packets'])+p['duplicates'])*wire.EHEADER.size,
            duplicates=p['duplicates'], ignored_tail_bytes=p['ignored_tail_bytes'])
