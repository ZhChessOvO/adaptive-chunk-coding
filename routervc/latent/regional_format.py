"""RVLR1: unchanged RVL1 base wrapped once, followed by addressed ROI E.

Each ROI refines all channels of one fixed 4x4 spatial latent tile. Packet IDs
are the coverage: no additional action mask. Payload integrity binds the base,
position and length. Order is arbitrary; identical duplicates are idempotent.
This first single-P8 format is deliberately separate from completed RVL1/RVLC1.
"""
import hashlib
import struct

from routervc.latent import format as base_format

MAGIC = b'RVLROI01'
HEADER = struct.Struct('<8sI32s')
EHEADER = struct.Struct('<4sBBI16s')
GRID = 4


def digest(data):
    return hashlib.sha256(data).digest()


def region_slice(shape, index):
    if type(index) is not int or not 0 <= index < GRID*GRID:
        raise ValueError('invalid region ID')
    if len(shape) != 4 or shape[:2] != (1, 256) or min(shape[2:]) < GRID:
        raise ValueError('invalid compact latent shape')
    height, width = shape[2:]
    row, col = divmod(index, GRID)
    return (slice(None), slice(None), slice(row*height//GRID, (row+1)*height//GRID),
            slice(col*width//GRID, (col+1)*width//GRID))


def base_stream(base, profile):
    parsed = base_format.parse(base)
    if parsed.fine is not None or parsed.qp != 48 or parsed.width != 3:
        raise ValueError('RVLR1 requires an unenhanced RVL1 QP48/width3 base')
    if len(profile) != 64:
        raise ValueError('invalid regional profile identity')
    return HEADER.pack(MAGIC, len(base), bytes.fromhex(profile)) + base


def packet(index, payload, base_digest):
    if type(index) is not int or not 0 <= index < GRID*GRID or len(payload) < 8:
        raise ValueError('invalid regional payload')
    if len(base_digest) != 32:
        raise ValueError('invalid B identity')
    check = digest(base_digest + struct.pack('<BI', index, len(payload)) + payload)[:16]
    return EHEADER.pack(b'RE01', index, 0, len(payload), check) + payload


def parse(data, *, allow_incomplete_tail=False):
    if len(data) < HEADER.size:
        raise ValueError('short regional header')
    magic, size, profile = HEADER.unpack_from(data)
    if magic != MAGIC or size < 190 or size > 128 << 20:
        raise ValueError('invalid regional header')
    end = HEADER.size + size
    if end > len(data):
        raise ValueError('incomplete regional B')
    base = data[HEADER.size:end]
    parsed = base_format.parse(base)
    if parsed.fine is not None or parsed.width != 3 or parsed.qp != 48:
        raise ValueError('unsupported regional B')
    identity = digest(data[:end])
    cursor, ignored, duplicates = end, 0, 0
    regions = {}
    while cursor < len(data):
        remaining = len(data) - cursor
        if remaining < EHEADER.size:
            if not allow_incomplete_tail:
                raise ValueError('partial regional header')
            ignored = remaining
            break
        tag, index, reserved, length, check = EHEADER.unpack_from(data, cursor)
        if tag != b'RE01' or index >= GRID*GRID or reserved or not 8 <= length <= 64 << 20:
            raise ValueError('invalid regional packet')
        if remaining < EHEADER.size + length:
            if not allow_incomplete_tail:
                raise ValueError('partial regional payload')
            ignored = remaining
            break
        payload = data[cursor+EHEADER.size:cursor+EHEADER.size+length]
        if packet(index, payload, identity) != data[cursor:cursor+EHEADER.size+length]:
            raise ValueError('regional packet corrupt or bound to another B')
        if index in regions:
            if regions[index] != payload:
                raise ValueError('conflicting regional duplicate')
            duplicates += 1
        regions[index] = payload
        cursor += EHEADER.size + length
    return dict(base=base, base_end=end, profile=profile.hex(), regions=regions,
                ignored_tail_bytes=ignored, duplicates=duplicates)
