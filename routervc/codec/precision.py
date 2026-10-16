"""Precision. Extracted from the validated RouterVC research implementation."""

from __future__ import annotations
import gc
import numpy as np
from routervc.codec.reference import ChainCodec
from routervc.codec.reference import reference_hash
import routervc.codec.entropy as entropy
from routervc.codec.split import split_symbols
from routervc.codec.split import restore_symbols
from routervc.codec.split import coarse_representatives
import routervc.codec.format as wire
import routervc.codec.refinement_entropy as compact


def identity():
    from routervc.assets import code_identity

    return code_identity("codec")


class PrecisionCodec(ChainCodec):

    def __init__(self):
        super().__init__()
        self.precision_identities = (*self.identities, identity())

    def encode(self, source, width, check=lambda: None):
        import torch
        from routervc.codec.uf import tensor_from_rgb
        from routervc.codec.uf import rgb_from_tensor

        compact.validate(width)
        if (
            source.dtype != np.uint8
            or source.ndim != 4
            or source.shape[-1] != 3
            or (not 2 <= len(source) <= 65)
        ):
            raise ValueError("invalid RGB source")
        h, w = source.shape[1:3]
        count = len(source)
        if any((v < 64 or v > 2048 or v % 16 for v in (h, w))):
            raise ValueError("unsupported geometry")
        pad_r, pad_b = self.codec.i_net.get_padding_size(h, w, 16)
        torch.cuda.set_stream(self.codec.stream)
        with torch.inference_mode():
            i = self.codec.i_net.compress(
                tensor_from_rgb(source[0], self.codec.device), 32, pad_b, pad_r
            )
            iframe = i["x_hat"].clone()
            self.codec.p_net.add_ref_feature_from_frame(
                iframe, apply_feature_adaptor=False
            )
            proxy = self.codec.p_net.proxy
            decoder_state = self.bridge.context(proxy)
            self.codec.p_net.add_ref_feature_from_frame(iframe)
            ready = self.bridge.context(proxy)
            first = rgb_from_tensor(iframe, h, w)
            base = [first]
            full_frames = [first]
            chunks, payloads, refs, native_checks = ([], {}, [], [])
            for index, start in enumerate(range(1, count, 8)):
                check()
                valid = min(8, count - start)
                reset = (start + 8) % 32 == 1
                frames = [
                    tensor_from_rgb(s, self.codec.device)
                    for s in source[start : start + valid]
                ]
                frames += [frames[-1]] * (8 - valid)
                self.bridge.restore(proxy, ready)
                p = self.codec.p_net.compress(
                    torch.cat(frames, 1), 48, reset, pad_b, pad_r
                )
                enc = self.bridge.encoded(proxy)
                y = enc["symbols"].cpu().numpy().astype(np.int16)
                z = enc["z"].cpu().numpy()
                scales = (
                    self.bridge.prior(proxy, enc["z"], 48, ready)["scales"]
                    .cpu()
                    .numpy()
                )
                full = self.bridge.synthesize(
                    proxy, enc["z"], enc["symbols"], 48, ready
                )
                full_rgb = self.rgb(full, h, w)[:valid]
                self.bridge.restore(proxy, decoder_state)
                native = self.codec.p_net.decompress(
                    p["bit_stream"],
                    {"height": h, "width": w},
                    48,
                    p["ec_parallel"],
                    reset,
                )["x_hat"]
                endpoint = np.stack([rgb_from_tensor(x, h, w) for x in native[:valid]])
                if not np.array_equal(endpoint, full_rgb):
                    raise RuntimeError("native endpoint differs")
                self.bridge.restore(proxy, ready)
                c, r = split_symbols(y, width)
                bottom = self.bridge.synthesize(
                    proxy,
                    enc["z"],
                    self.tensor(coarse_representatives(c, width)),
                    48,
                    ready,
                )
                base.extend(self.rgb(bottom, h, w)[:valid])
                full_frames.extend(full_rgb)
                chunks.append(
                    (
                        entropy.encode_z(
                            z, self.codec.p_net.bit_estimator_z.get_cdf_info(), 48
                        ),
                        entropy.encode_coarse(c, scales, width),
                    )
                )
                for k in range(16):
                    roi = wire.region_slice(c.shape, k)
                    payload = compact.encode(c[roi], r[roi], scales[roi], width)
                    np.testing.assert_array_equal(
                        compact.decode(payload, c[roi], scales[roi], width), r[roi]
                    )
                    payloads[index, k] = payload
                ready, decoder_state = self.advance(ready, bottom["feature"], reset)
                refs.append(reference_hash(ready))
                native_checks.append(
                    dict(
                        endpoint_exact=True,
                        valid_frames=valid,
                        reset=reset,
                        native_P_bytes=len(p["bit_stream"]),
                    )
                )
            meta = dict(
                profile="RVL91",
                identities=list(self.precision_identities),
                frames=count,
                height=h,
                width=w,
                q_star=48,
                bin_width=width,
                I_qp=32,
                I_parallel=int(i["ec_parallel"]),
            )
            base_wire = wire.base_stream(
                wire.pack_inner(meta, bytes(i["bit_stream"]), chunks), identity()
            )
            bank = base_wire + b"".join(
                (
                    wire.packet(j, k, payloads[j, k], wire.digest(base_wire))
                    for k in range(16)
                    for j in range(len(chunks))
                )
            )
            torch.cuda.synchronize()
        return (
            bank,
            np.stack(base),
            np.stack(full_frames),
            dict(
                base_reference_hashes=refs,
                same_context_chunks=native_checks,
                bin_width=width,
                separate_reference_chain=True,
            ),
        )

    def _states(self, inner, refs):
        import torch
        from routervc.codec.uf import rgb_from_tensor
        from routervc.codec.checks import context_exact

        p = wire.parse_inner(inner)
        m = p["meta"]
        width = m["bin_width"]
        if tuple(m["identities"]) != self.precision_identities:
            raise ValueError("B identities differ")
        h, w, count = (m["height"], m["width"], m["frames"])
        torch.cuda.set_stream(self.codec.stream)
        intra = self.codec.i_net.decompress(
            p["intra"], {"height": h, "width": w}, 32, m["I_parallel"]
        )["x_hat"].clone()
        self.codec.p_net.add_ref_feature_from_frame(intra)
        proxy = self.codec.p_net.proxy
        ready = self.bridge.context(proxy)
        yield dict(index=-1, rgb=rgb_from_tensor(intra, h, w))
        for index, (zd, yd) in enumerate(p["chunks"]):
            start = 1 + 8 * index
            valid = min(8, count - start)
            z = entropy.decode_z(
                zd,
                (1, 128, (h + 63) // 64, (w + 63) // 64),
                self.codec.p_net.bit_estimator_z.get_cdf_info(),
                48,
            )
            zg = self.tensor(z)
            scales = self.bridge.prior(proxy, zg, 48, ready)["scales"].cpu().numpy()
            c = entropy.decode_coarse(yd, scales, width)
            bottom = self.bridge.synthesize(
                proxy, zg, self.tensor(coarse_representatives(c, width)), 48, ready
            )
            yield dict(
                index=index,
                c=c,
                scales=scales,
                z=zg,
                context=ready,
                rgb=self.rgb(bottom, h, w)[:valid],
                h=h,
                w=w,
                valid=valid,
                bin_width=width,
            )
            if not context_exact(self.bridge, proxy, ready):
                raise RuntimeError("E mutated B reference")
            ready, _ = self.advance(ready, bottom["feature"], (start + 8) % 32 == 1)
            refs.append(reference_hash(ready))

    def decode(self, data, *, allow_incomplete_tail=False):
        import torch

        p = wire.parse(data, allow_incomplete_tail=allow_incomplete_tail)
        if p["profile"] != identity():
            raise ValueError("precision codec profile differs")
        base, out, refs = ([], [], [])
        with torch.inference_mode():
            for s in self._states(p["base"], refs):
                j = s["index"]
                if j < 0:
                    base.append(s["rgb"])
                    out.append(s["rgb"])
                    continue
                c, scales, width = (s["c"], s["scales"], s["bin_width"])
                mixed = coarse_representatives(c, width).copy()
                received = [
                    (k, body) for (ch, k), body in p["packets"].items() if ch == j
                ]
                for k, body in received:
                    roi = wire.region_slice(c.shape, k)
                    r = compact.decode(body, c[roi], scales[roi], width)
                    mixed[roi] = restore_symbols(c[roi], r, width)
                shown = s["rgb"]
                if received:
                    full = self.bridge.synthesize(
                        self.codec.p_net.proxy,
                        s["z"],
                        self.tensor(mixed),
                        48,
                        s["context"],
                    )
                    shown = self.rgb(full, s["h"], s["w"])[: s["valid"]]
                base.extend(s["rgb"])
                out.extend(shown)
            torch.cuda.synchronize()
        info = p["info"]
        return (
            np.stack(base),
            np.stack(out),
            dict(
                source_frames_read=False,
                generation=False,
                router=False,
                additional_mask_bytes=0,
                base_reference_unchanged=True,
                base_reference_hashes=refs,
                received_regions=[list(k) for k in sorted(p["packets"])],
                frame_count=info["frames"],
                bin_width=info["bin_width"],
                actual_bytes=len(data),
                bpp=len(data) * 8 / (info["frames"] * info["height"] * info["width"]),
                base_bytes=p["base_end"],
                inner_base_bytes=len(p["base"]),
                wrapper_bytes=wire.HEADER.size,
                E_wire_bytes=len(data) - p["base_end"] - p["ignored_tail_bytes"],
                E_unique_payload_bytes=sum(map(len, p["packets"].values())),
                E_header_bytes=(len(p["packets"]) + p["duplicates"])
                * wire.EHEADER.size,
                duplicates=p["duplicates"],
                ignored_tail_bytes=p["ignored_tail_bytes"],
            ),
        )


def decode(data):
    import torch
    from routervc.utils.precision import codec_precision

    with codec_precision():
        codec = PrecisionCodec()
        result = codec.decode(data)
    del codec
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.set_stream(torch.cuda.default_stream())
    return result
