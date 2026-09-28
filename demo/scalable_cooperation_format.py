"""ACSG v2: optional generation conditioned on the received enhancement.

ACSE bytes remain immutable. Unlike v1, E and G supports may overlap. Controls
are charged, not an implicit encoder-only mask. No trained receiver policy yet.
"""
import math
import struct
import zlib

import numpy as np

from demo import scalable_generation_format as legacy
from demo.scalable_format import parse as parse_enhancement

HEADER = legacy.HEADER
CONTROL = struct.Struct("<Qd4H2HHd")
REGION = legacy.REGION
HASHES = legacy.HASHES
KEYS = legacy.KEYS | {"processing_scale", "blend"}
region_slice = legacy.region_slice
windows = legacy.windows


def validate(control, inner):
    if set(control) != KEYS:
        raise ValueError("unknown/missing cooperation control")
    if (type(control["processing_scale"]) is not int or control["processing_scale"] not in (1, 2)
            or type(control["blend"]) not in (int, float)
            or not math.isfinite(control["blend"]) or not 0 <= control["blend"] <= 1):
        raise ValueError("unsupported scale/blend")
    # Share v1's exact geometry/hash validation without changing its semantics.
    basic = {k: v for k, v in control.items() if k in legacy.KEYS}
    if not isinstance(basic["generate"], list):
        raise ValueError("regions must be lists")
    if basic["generate"]:
        legacy.validate(basic, inner)
    else:
        # Validate the common controls even when generation is entirely skipped.
        m = inner.meta
        from types import SimpleNamespace
        virtual = SimpleNamespace(meta=dict(m, frame_count=max(17, m["frame_count"]),
                                            width=max(1024, m["width"]), height=max(1024, m["height"])))
        legacy.validate(dict(basic, generate=[[0, 17, 0, 0, 1024, 1024]]), virtual)
        # Protection still refers to the actual video, not the virtual rectangle.
        for t, n, x, y, w, h in basic["protect"]:
            if t+n > m["frame_count"] or x+w > m["width"] or y+h > m["height"]:
                raise ValueError("protection outside video")


def wrap(inner_bytes, control):
    inner = parse_enhancement(inner_bytes)
    if inner_bytes[4] != 2:
        raise ValueError("cooperation requires ACSE v2")
    validate(control, inner)
    body = CONTROL.pack(*(control[k] for k in
        ("seed", "strength", "window", "stride", "context", "feather")),
        len(control["generate"]), len(control["protect"]), control["processing_scale"], control["blend"])
    body += b"".join(bytes.fromhex(control[k]) for k in HASHES)
    body += b"".join(REGION.pack(*r) for k in ("generate", "protect") for r in control[k])
    return HEADER.pack(b"ACSG", 2, len(body), zlib.crc32(body)) + body + inner_bytes


def parse(data, *, allow_incomplete_tail=False):
    if len(data) < HEADER.size:
        raise ValueError("truncated cooperation header")
    magic, version, size, crc = HEADER.unpack_from(data)
    if magic != b"ACSG" or version != 2 or not CONTROL.size+32*len(HASHES) <= size <= 8192:
        raise ValueError("invalid cooperation header")
    end = HEADER.size+size
    body = data[HEADER.size:end]
    if len(body) != size or zlib.crc32(body) != crc:
        raise ValueError("truncated/corrupt cooperation control")
    values = CONTROL.unpack_from(body)
    control = dict(zip(("seed", "strength", "window", "stride", "context", "feather"), values[:6]))
    ng, np_, control["processing_scale"], control["blend"] = values[6:]
    offset = CONTROL.size
    for key in HASHES:
        control[key] = body[offset:offset+32].hex()
        offset += 32
    if size != offset+REGION.size*(ng+np_):
        raise ValueError("invalid cooperation control size")
    for key, count in (("generate", ng), ("protect", np_)):
        control[key] = [list(REGION.unpack_from(body, offset+i*REGION.size)) for i in range(count)]
        offset += count*REGION.size
    inner_bytes = data[end:]
    if len(inner_bytes) < 5 or inner_bytes[4] != 2:
        raise ValueError("requires ACSE v2")
    inner = parse_enhancement(inner_bytes, allow_incomplete_tail=allow_incomplete_tail)
    validate(control, inner)
    return control, inner_bytes, inner, end


def weights(shape, control):
    """E does not veto G. Explicit protection and zero blend still veto writes."""
    from types import SimpleNamespace
    mask = legacy.weights(shape, control, SimpleNamespace(packets=[]))
    return mask*control["blend"]


def combine(enhanced, generated, alpha):
    return np.rint(enhanced.astype(np.float32)+alpha[..., None]*(
        generated.astype(np.float32)-enhanced.astype(np.float32))).clip(0, 255).astype(np.uint8)
