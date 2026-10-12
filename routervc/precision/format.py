"""RVLWPAC1 / RVLW1: same framing sizes as width3, explicit small width.

Different widths are different B streams, never cross-width prefixes. The
only progressive units are complete addressed E packets on one fixed B.
"""
import hashlib
import json
import math
import struct
from routervc.latent.packet_format import EHEADER, packet, digest
from routervc.latent.regional_format import region_slice
from routervc.precision.entropy import validate

MAGIC = b'RVLWPAC1'
INNER_MAGIC = b'RVLWCHAN'
HEADER = struct.Struct('<8sBI32s')
INNER_HEADER = struct.Struct('<8sI')


def pack_inner(meta, intra, chunks):
    meta = dict(meta, I_size=len(intra), chunks=[dict(z_size=len(z), y_size=len(y)) for z, y in chunks])
    text = json.dumps(meta, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    body = INNER_HEADER.pack(INNER_MAGIC, len(text))+text+intra+b''.join(z+y for z, y in chunks)
    result = body+digest(body)
    parse_inner(result)
    return result


def parse_inner(data):
    if len(data) < INNER_HEADER.size: raise ValueError('short base header')
    magic, size = INNER_HEADER.unpack_from(data)
    if magic != INNER_MAGIC or not 10 <= size <= 65536 or len(data) < INNER_HEADER.size+size:
        raise ValueError('invalid base header')
    meta = json.loads(data[INNER_HEADER.size:INNER_HEADER.size+size])
    expected = {'profile', 'identities', 'frames', 'height', 'width', 'q_star',
                'bin_width', 'I_qp', 'I_parallel', 'I_size', 'chunks'}
    if not isinstance(meta, dict) or set(meta) != expected or meta['profile'] != 'RVLW1':
        raise ValueError('unsupported precision profile')
    if any(type(meta[k]) is not int for k in ('frames','height','width','q_star','bin_width','I_qp','I_parallel','I_size')):
        raise ValueError('noninteger config')
    validate(meta['bin_width'])
    if meta['q_star'] != 48 or meta['I_qp'] != 32: raise ValueError('unsupported working point')
    if not 2 <= meta['frames'] <= 65 or any(v < 64 or v > 2048 or v % 16 for v in (meta['height'], meta['width'])):
        raise ValueError('unsupported geometry')
    if not 1 <= meta['I_parallel'] <= 8 or not 4 <= meta['I_size'] <= 64 << 20:
        raise ValueError('invalid bootstrap')
    ids = meta['identities']
    if not isinstance(ids, list) or len(ids) != 5 or any(type(h) is not str or len(h) != 64 for h in ids):
        raise ValueError('invalid identities')
    for identity in ids: bytes.fromhex(identity)
    if not isinstance(meta['chunks'], list) or len(meta['chunks']) != math.ceil((meta['frames']-1)/8):
        raise ValueError('wrong P8 count')
    start = INNER_HEADER.size+size; cursor = start+meta['I_size']; chunks = []
    for chunk in meta['chunks']:
        if not isinstance(chunk, dict) or set(chunk) != {'z_size', 'y_size'} or any(
                type(v) is not int or not 4 <= v <= 64 << 20 for v in chunk.values()):
            raise ValueError('invalid chunk lengths')
        zend = cursor+chunk['z_size']; yend = zend+chunk['y_size']
        chunks.append((data[cursor:zend], data[zend:yend])); cursor = yend
    if cursor+32 != len(data) or data[cursor:] != digest(data[:cursor]):
        raise ValueError('incomplete/corrupt/trailing base')
    return dict(meta=meta, intra=data[start:start+meta['I_size']], chunks=chunks)


def base_stream(inner, profile):
    parse_inner(inner)
    if type(profile) is not str or len(profile) != 64: raise ValueError('invalid profile')
    return HEADER.pack(MAGIC, 2, len(inner), bytes.fromhex(profile))+inner


def parse(data, *, allow_incomplete_tail=False):
    if len(data) < HEADER.size: raise ValueError('short B header')
    magic, kind, size, profile = HEADER.unpack_from(data)
    if magic != MAGIC or kind != 2 or not 190 <= size <= 128 << 20:
        raise ValueError('invalid B header')
    end = HEADER.size+size
    if end > len(data): raise ValueError('incomplete B')
    inner = data[HEADER.size:end]; p = parse_inner(inner); m = p['meta']
    info = dict(frames=m['frames'], height=m['height'], width=m['width'],
                bin_width=m['bin_width'], chunks=len(p['chunks']))
    identity = digest(data[:end]); cursor = end; packets = {}; duplicates = ignored = 0
    while cursor < len(data):
        remain = len(data)-cursor
        if remain < EHEADER.size:
            if not allow_incomplete_tail: raise ValueError('incomplete E header')
            ignored = remain; break
        chunk, region, length, check = EHEADER.unpack_from(data, cursor)
        if chunk >= info['chunks'] or region >= 16 or not 4 <= length <= 64 << 20:
            raise ValueError('invalid E header')
        if remain < EHEADER.size+length:
            if not allow_incomplete_tail: raise ValueError('incomplete E payload')
            ignored = remain; break
        body = data[cursor+EHEADER.size:cursor+EHEADER.size+length]
        if packet(chunk, region, body, identity) != data[cursor:cursor+EHEADER.size+length]:
            raise ValueError('corrupt E or wrong B identity')
        key = chunk, region
        if key in packets:
            if packets[key] != body: raise ValueError('conflicting duplicate E')
            duplicates += 1
        packets[key] = body; cursor += EHEADER.size+length
    return dict(base=inner, kind=kind, info=info, profile=profile.hex(), packets=packets,
                base_end=end, ignored_tail_bytes=ignored, duplicates=duplicates)


def subset(bank, selected):
    parsed = parse(bank)
    if len(set(selected)) != len(selected) or any(type(i) is not int or not 0 <= i < 16 for i in selected):
        raise ValueError('invalid region selection')
    base = bank[:parsed['base_end']]; parts = [base]
    for region in selected:
        for chunk in range(parsed['info']['chunks']):
            if (chunk, region) not in parsed['packets']: raise ValueError('incomplete candidate bundle')
            parts.append(packet(chunk, region, parsed['packets'][chunk, region], digest(base)))
    return b''.join(parts)


def bundle_bytes(bank):
    p = parse(bank)
    return [sum(EHEADER.size+len(p['packets'][j, i]) for j in range(p['info']['chunks'])) for i in range(16)]
