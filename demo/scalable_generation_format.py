"""ACSG v1: generation control around an unaltered appendable ACSE v2 stream.

Model weights are pre-shared, identified by full SHA256, not sent per video.
The version fixes one-step BF16 SeedVR2, cfg=1, base-only input, 17/8 temporal
overlap, crop-at-native-resolution and inner feathering. No implicit ROI CLI.
This research format does not yet carry presentation timestamps/audio.
"""
import math
import struct
import zlib

import numpy as np

from demo.scalable_format import parse as parse_enhancement

HEADER = struct.Struct("<4sBII")
CONTROL = struct.Struct("<Qd4H2H")
REGION = struct.Struct("<II4H")
HASHES = ("dit", "vae", "positive", "negative", "lora", "profile")
KEYS = set(HASHES) | {"seed", "strength", "window", "stride", "context", "feather",
                      "generate", "protect"}


def region_slice(region):
    start, count, x, y, w, h = region
    return np.s_[start:start+count, y:y+h, x:x+w]


def validate(control, inner):
    if set(control) != KEYS:
        raise ValueError("unknown/missing generation control")
    if (type(control["seed"]) is not int or not 0 <= control["seed"] < 2**63
            or type(control["strength"]) not in (int, float)
            or not math.isfinite(control["strength"]) or not 0 <= control["strength"] <= 1
            or control["window"] != 17 or control["stride"] != 8
            or any(type(control[k]) is not int or not 0 <= control[k] <= 256
                   for k in ("context", "feather"))):
        raise ValueError("unsupported generation profile/control")
    for key in HASHES:
        value = control[key]
        if (not isinstance(value, str) or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)):
            raise ValueError("invalid canonical model/profile hash")
    m = inner.meta
    if not control["generate"] or sum(len(control[k]) for k in ("generate", "protect")) > 256:
        raise ValueError("invalid region count")
    # Pairwise rectangle intersection avoids allocating video-sized masks in parser.
    for key in ("generate", "protect"):
        regions = control[key]
        if not isinstance(regions, list):
            raise ValueError("regions must be lists")
        for i, r in enumerate(regions):
            if not isinstance(r, list) or len(r) != 6 or any(type(v) is not int for v in r):
                raise ValueError("invalid region record")
            t,n,x,y,w,h = r
            if (min(t,x,y) < 0 or min(n,w,h) < 1 or t+n > m["frame_count"]
                    or x+w > m["width"] or y+h > m["height"]
                    or (key == "generate" and (n < 17 or min(w,h) <= 2*control["feather"]))):
                raise ValueError("region outside video or too small")
            for s in regions[:i]:
                if intersects(r, s):
                    raise ValueError("overlapping same-kind regions")


def intersects(a, b):
    return all(max(a[i],b[i]) < min(a[i]+a[j], b[i]+b[j])
               for i,j in ((0,1),(2,4),(3,5)))


def wrap(inner_bytes, control):
    inner = parse_enhancement(inner_bytes)
    if inner_bytes[4] != 2:
        raise ValueError("generation envelope requires ACSE v2")
    validate(control, inner)
    body = CONTROL.pack(*(control[k] for k in
        ("seed", "strength", "window", "stride", "context", "feather")),
        len(control["generate"]), len(control["protect"]))
    body += b"".join(bytes.fromhex(control[k]) for k in HASHES)
    body += b"".join(REGION.pack(*r) for k in ("generate", "protect") for r in control[k])
    return HEADER.pack(b"ACSG", 1, len(body), zlib.crc32(body)) + body + inner_bytes


def parse(data, *, allow_incomplete_tail=False):
    if len(data) < HEADER.size:
        raise ValueError("truncated generation header")
    magic, version, size, crc = HEADER.unpack_from(data)
    if magic != b"ACSG" or version != 1 or not CONTROL.size+32*len(HASHES) <= size <= 8192:
        raise ValueError("invalid generation header")
    end = HEADER.size+size
    body = data[HEADER.size:end]
    if len(body) != size or zlib.crc32(body) != crc:
        raise ValueError("truncated/corrupt generation control")
    values = CONTROL.unpack_from(body)
    c = dict(zip(("seed", "strength", "window", "stride", "context", "feather"), values[:6]))
    ng, np_ = values[6:]
    offset = CONTROL.size
    for k in HASHES:
        c[k] = body[offset:offset+32].hex()
        offset += 32
    if size != offset + REGION.size*(ng+np_):
        raise ValueError("invalid generation control size")
    for key, n in (("generate",ng),("protect",np_)):
        c[key] = [list(REGION.unpack_from(body, offset+i*REGION.size)) for i in range(n)]
        offset += n*REGION.size
    inner_bytes = data[end:]
    if len(inner_bytes) < 5 or inner_bytes[4] != 2:
        raise ValueError("requires ACSE v2")
    inner = parse_enhancement(inner_bytes, allow_incomplete_tail=allow_incomplete_tail)
    validate(c, inner)
    return c, inner_bytes, inner, end


def windows(count, window=17, stride=8):
    if count < window:
        raise ValueError("short generation window unsupported by v1")
    result = list(range(0, count-window+1, stride))
    if result[-1] != count-window:
        result.append(count-window)
    return result


def weights(shape, control, inner):
    """Nonnegative inner feather; generation cannot write protection/E pixels."""
    out = np.zeros(shape[:3], np.float32)
    feather = control["feather"]
    for r in control["generate"]:
        _,_,_,_,w,h = r
        xx, yy = np.arange(w), np.arange(h)
        distance = np.minimum(np.minimum(xx,w-1-xx)[None,:], np.minimum(yy,h-1-yy)[:,None])
        alpha = np.minimum(distance/max(feather,1),1) if feather else np.ones((h,w))
        out[region_slice(r)] = alpha
    for r in control["protect"]:
        out[region_slice(r)] = 0
    for p in inner.packets:
        out[region_slice([p.meta["start"],p.meta["count"],*p.meta["roi"]])] = 0
    return out
