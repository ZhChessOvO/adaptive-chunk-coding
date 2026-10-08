"""RVLPACK2: fixed-width addressed E on an unchanged RVL1 or RVLC1 base.

One entropy stream per (chunk,region); no extra mask or repeated packet magic.
All lengths, identities, addresses and checksums are charged. This diagnostic
format retains the legacy B framing to isolate E packing, not optimize B yet.
"""
import hashlib
import struct
from routervc.latent import format as single, chain_format as chain
from routervc.latent.regional_format import region_slice

MAGIC = b'RVLPACK2'
HEADER = struct.Struct('<8sBI32s')
EHEADER = struct.Struct('<BBI16s')


def digest(data):
    return hashlib.sha256(data).digest()


def inner_info(data, kind):
    if kind == 1:
        p = single.parse(data)
        if p.fine is not None or p.qp != 48 or p.width != 3:
            raise ValueError('unsupported single base')
        return dict(frames=9, height=p.height, width=p.image_width, chunks=1)
    if kind == 2:
        p = chain.parse(data)
        if p['enhancements']:
            raise ValueError('inner B contains E')
        m = p['meta']
        return dict(frames=m['frames'], height=m['height'], width=m['width'], chunks=len(p['chunks']))
    raise ValueError('invalid B format')


def base_stream(data, kind, profile):
    inner_info(data, kind)
    if len(profile) != 64:
        raise ValueError('invalid profile')
    return HEADER.pack(MAGIC, kind, len(data), bytes.fromhex(profile)) + data


def packet(chunk, region, payload, identity):
    if type(chunk) is not int or not 0 <= chunk < 8 or type(region) is not int or not 0 <= region < 16:
        raise ValueError('bad E address')
    if not 4 <= len(payload) <= 64 << 20 or len(identity) != 32:
        raise ValueError('bad E size/identity')
    check = digest(identity + struct.pack('<BBI', chunk, region, len(payload)) + payload)[:16]
    return EHEADER.pack(chunk, region, len(payload), check) + payload


def parse(data, *, allow_incomplete_tail=False):
    if len(data) < HEADER.size:
        raise ValueError('short B header')
    magic, kind, size, profile = HEADER.unpack_from(data)
    if magic != MAGIC or kind not in (1, 2) or not 190 <= size <= 128 << 20:
        raise ValueError('invalid B header')
    end = HEADER.size + size
    if end > len(data):
        raise ValueError('incomplete B')
    inner = data[HEADER.size:end]
    info = inner_info(inner, kind)
    identity = digest(data[:end])
    cursor, duplicates, ignored = end, 0, 0
    packets = {}
    while cursor < len(data):
        remain = len(data)-cursor
        if remain < EHEADER.size:
            if not allow_incomplete_tail:
                raise ValueError('incomplete E header')
            ignored = remain
            break
        chunk, region, length, check = EHEADER.unpack_from(data, cursor)
        if chunk >= info['chunks'] or region >= 16 or not 4 <= length <= 64 << 20:
            raise ValueError('invalid E header')
        if remain < EHEADER.size+length:
            if not allow_incomplete_tail:
                raise ValueError('incomplete E payload')
            ignored = remain
            break
        body = data[cursor+EHEADER.size:cursor+EHEADER.size+length]
        if packet(chunk, region, body, identity) != data[cursor:cursor+EHEADER.size+length]:
            raise ValueError('corrupt E or wrong B identity')
        key = (chunk, region)
        if key in packets:
            if packets[key] != body:
                raise ValueError('conflicting duplicate E')
            duplicates += 1
        packets[key] = body
        cursor += EHEADER.size+length
    return dict(base=inner, kind=kind, info=info, profile=profile.hex(), packets=packets,
                base_end=end, ignored_tail_bytes=ignored, duplicates=duplicates)
