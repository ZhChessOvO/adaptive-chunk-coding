"""Frozen G transfer diagnostic on new received B/E, not trained new Routers.

G support is a fixed shared four-center-ROI policy derived from dimensions, NOT
an encoder mask. The envelope identifies that policy and shared G assets. E8
contains B, E-only and EG areas; E0 includes G-only. No feature-delta E is read.
"""
import hashlib
import json
from pathlib import Path
import struct
import zlib

from demo.scalable_codec import file_hash
from routervc.latent.packet_codec import packet_hash
from routervc.latent import packet_format

REPO = Path(__file__).resolve().parents[2]
ADAPTER = Path('/root/autodl-fs/DCVC/runs/a800_online_eg_20261002/joint/adapter.pt')
MAGIC = b'RVLGEN01'
HEADER = struct.Struct('<8s32s32sI')
REGIONS = (5, 6, 9, 10)


def profile():
    names = ('internal_condition_decode.py', 'scalable_cooperation_format.py',
             'scalable_generation_format.py', 'stage_c_a800_teacher.py',
             'stage_c_seedvr2_bridge.py', 'stage_c_seedvr2_lora_utils.py',
             'internal_condition_model.py', 'routervc_policy.py')
    return hashlib.sha256((packet_hash()+file_hash(Path(__file__))+''.join(
        file_hash(REPO/'demo'/n) for n in names)).encode()).hexdigest()


def assets(adapter=ADAPTER):
    from demo.online_eg_decode import identities
    return identities(adapter)


def asset_hash(values):
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def wrap(inner, asset_identity):
    p = packet_format.parse(inner)
    if p['info']['frames'] != 17 or len(asset_identity) != 64:
        raise ValueError('G transfer requires exactly 17 frames and shared model hash')
    identities = bytes.fromhex(profile())+bytes.fromhex(asset_identity)
    return HEADER.pack(MAGIC, identities[:32], identities[32:], zlib.crc32(identities))+inner


def parse(data):
    if len(data) < HEADER.size:
        raise ValueError('incomplete G envelope')
    tag, code, identity, crc = HEADER.unpack_from(data)
    if tag != MAGIC or code.hex() != profile() or zlib.crc32(code+identity) != crc:
        raise ValueError('unknown/corrupt G transfer profile')
    inner = data[HEADER.size:]
    p = packet_format.parse(inner)
    if p['info']['frames'] != 17:
        raise ValueError('G probe only supports 17-frame windows')
    return inner, identity.hex(), p


def control(shape, hashes):
    from demo.routervc_policy import grid_rois
    t, h, w, _ = shape
    if t != 17:
        raise ValueError('wrong G window length')
    rois = grid_rois(h, w)
    selected = [rois[i] for i in REGIONS]
    for x, y, rw, rh in selected:
        x0, y0 = max(0, x-64), max(0, y-64)
        cw, ch = min(w, x+rw+64)-x0, min(h, y+rh+64)-y0
        if min(rw, rh) <= 32 or x0%8 or y0%8 or cw%16 or ch%16:
            raise ValueError('unaligned G geometry; no silent resizing')
    return dict(hashes, seed=20261008, strength=1., blend=1., window=17, stride=8,
                context=64, feather=16, processing_scale=1, protect=[],
                generate=[[0, t, *roi] for roi in selected])
