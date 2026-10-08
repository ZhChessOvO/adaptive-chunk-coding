"""Source-free R_g on real width3 B/E; independent of the future sender R_s.

RVLRG001 identifies a shared receiver and frozen G, not a transmitted action
map. Region-addressed seeds make a cell's G result independent of which other
cells are selected. Every G call conditions on the same ungenerated Y.
"""
import hashlib
import json
from pathlib import Path
import struct
import time
import zlib

import numpy as np

from demo.scalable_codec import file_hash
from routervc.latent import packet_format as packets
from routervc.latent import generation as fixed_g

MAGIC = b'RVLRG001'
FIELDS = struct.Struct('<8s32s32s32sQBf')
HEADER_BYTES = FIELDS.size + 4
SEED = 20261008


def identity():
    from demo.routervc_receiver_router import policy_identity
    from demo import routervc_policy
    return hashlib.sha256((file_hash(Path(__file__)) + fixed_g.profile() +
        policy_identity() + file_hash(Path(routervc_policy.__file__)) +
        file_hash(Path(__file__).resolve().parents[2]/'demo/four_state_receive.py')).encode()).hexdigest()


def wrap(inner, model_sha, asset_sha, *, max_g=8, seed=SEED, boundary=0.):
    p = packets.parse(inner)
    if p['info']['frames'] != 17 or p['kind'] != 2:
        raise ValueError('latent Router profile requires I + two P8 chunks')
    if (type(max_g) is not int or not 0 <= max_g <= 16 or type(seed) is not int
            or not 0 <= seed < 2**48 or not np.isfinite(boundary) or not 0 <= boundary <= 1
            or any(type(h) is not str or len(h) != 64 for h in (model_sha, asset_sha))):
        raise ValueError('invalid receiver configuration')
    header = FIELDS.pack(MAGIC, bytes.fromhex(identity()), bytes.fromhex(model_sha),
                        bytes.fromhex(asset_sha), seed, max_g, boundary)
    return header + struct.pack('<I', zlib.crc32(header)) + inner


def parse(data):
    if len(data) < HEADER_BYTES:
        raise ValueError('incomplete receiver header')
    fields = FIELDS.unpack_from(data)
    magic, code, model, assets, seed, max_g, boundary = fields
    if (magic != MAGIC or code.hex() != identity()
            or struct.unpack_from('<I', data, FIELDS.size)[0] != zlib.crc32(data[:FIELDS.size])):
        raise ValueError('invalid/corrupt latent receiver profile')
    inner = data[HEADER_BYTES:]
    # Reuse the exact encoder validation, including finite numeric values.
    if wrap(inner, model.hex(), assets.hex(), max_g=max_g, seed=seed, boundary=boundary) != data:
        raise ValueError('noncanonical receiver stream')
    return inner, dict(receiver_sha256=model.hex(), assets_sha256=assets.hex(),
                      seed=seed, max_g=max_g, boundary=boundary), packets.parse(inner)


def coverage(parsed):
    """Packet-derived temporal coverage; I is not an E-capable frame."""
    info = parsed['info']
    denominator = info['frames'] - 1
    result = np.zeros(16, np.float32)
    for chunk, region in parsed['packets']:
        result[region] += min(8, denominator - 8*chunk) / denominator
    return result


def subset(bank, selected):
    """Region-major, complete two-P8 bundles; B and headers are charged once."""
    parsed = packets.parse(bank)
    if (len(set(selected)) != len(selected)
            or any(type(i) is not int or not 0 <= i < 16 for i in selected)):
        raise ValueError('invalid region selection')
    base = bank[:parsed['base_end']]
    result = [base]
    for region in selected:
        for chunk in range(parsed['info']['chunks']):
            key = chunk, region
            if key not in parsed['packets']:
                raise ValueError('incomplete candidate bundle')
            result.append(packets.packet(chunk, region, parsed['packets'][key], packets.digest(base)))
    return b''.join(result)


def bundle_bytes(bank):
    parsed = packets.parse(bank)
    return [sum(packets.EHEADER.size + len(parsed['packets'][chunk, i])
                for chunk in range(parsed['info']['chunks'])) for i in range(16)]


def route(base, received, inner, config, model):
    """No source or sender model is accepted by this interface."""
    from demo.routervc_receiver_router import predict
    from demo.routervc_policy import select_generate
    parsed = packets.parse(inner)
    info = parsed['info']
    if (base.shape != received.shape or base.shape != (info['frames'], info['height'], info['width'], 3)
            or getattr(model, '_receiver_model_sha256', None) != config['receiver_sha256']):
        raise ValueError('received geometry/model identity differs')
    cov = coverage(parsed)
    gains = predict(model, base, received, cov)[0].numpy()
    result = select_generate(gains[:, 0], config['max_g'], config['boundary'])
    result.update(coverage=cov.tolist(), predictions=gains.tolist(),
        source_frames_used=False, sender_router_loaded=False, mask_bytes=0)
    return result


def control(shape, hashes, region, seed=SEED):
    from demo.routervc_visual_router import grid_rois
    if type(region) is not int or not 0 <= region < 16 or shape[0] != 17:
        raise ValueError('invalid single G region')
    roi = grid_rois(*shape[1:3])[region]
    x, y, w, h = roi
    x0, y0 = max(0, x-64), max(0, y-64)
    cw, ch = min(shape[2], x+w+64)-x0, min(shape[1], y+h+64)-y0
    if min(w, h) <= 32 or x0 % 8 or y0 % 8 or cw % 16 or ch % 16:
        raise ValueError('G processing geometry is unaligned')
    return dict(hashes, seed=seed+65536*region, strength=1., blend=1., window=17, stride=8,
                context=64, feather=16, processing_scale=1, protect=[],
                generate=[[0, 17, *roi]])


def render(received, indices, hashes, generator, *, seed=SEED, check=lambda: None):
    """Selected disjoint cores; order cannot change noise or conditioning RGB."""
    from demo.scalable_cooperation_format import weights
    if len(set(indices)) != len(indices):
        raise ValueError('duplicate generated region')
    output, reports = received.copy(), []
    for region in indices:
        check()
        settings = control(received.shape, hashes, region, seed)
        began = time.monotonic()
        generated, runtime = generator(received, settings)
        alpha = weights(received.shape, settings)
        np.testing.assert_array_equal(generated[alpha == 0], received[alpha == 0])
        output[alpha > 0] = generated[alpha > 0]
        reports.append(dict(region=region, seconds=time.monotonic()-began, runtime=runtime))
    return output, reports
