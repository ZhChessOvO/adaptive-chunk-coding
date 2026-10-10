"""RVLCOOP1: 121 global bytes; selection state stays local, fixed F code-bound."""
import hashlib
from pathlib import Path
import struct
import zlib
import numpy as np
from routervc.latent import routing

MAGIC = b'RVLCOOP1'
FIELDS = routing.FIELDS
HEADER_BYTES = routing.HEADER_BYTES


def identity():
    from routervc.cooperation import receiver
    repo = Path(__file__).resolve().parents[2]
    names = ('routervc/cooperation/stream.py', 'routervc/cooperation/render.py',
             'routervc/cooperation/receive.py', 'routervc/fusion/blend.py',
             'routervc/fusion/capture.py')
    return hashlib.sha256((routing.identity()+receiver.identity()+''.join(
        hashlib.sha256((repo/n).read_bytes()).hexdigest() for n in names)).encode()).hexdigest()


def wrap(inner, model, assets, max_g=8, seed=routing.SEED):
    routing.wrap(inner, model, assets, max_g=max_g, seed=seed, boundary=0.)
    header = FIELDS.pack(MAGIC, bytes.fromhex(identity()), bytes.fromhex(model),
                         bytes.fromhex(assets), seed, max_g, 0.)
    return header+struct.pack('<I', zlib.crc32(header))+inner


def parse(data):
    if len(data) < HEADER_BYTES: raise ValueError('incomplete cooperative header')
    magic, code, model, assets, seed, max_g, reserved = FIELDS.unpack_from(data)
    if (magic != MAGIC or code.hex() != identity() or reserved != 0
            or not np.isfinite(reserved) or struct.unpack_from('<I', data, FIELDS.size)[0]
            != zlib.crc32(data[:FIELDS.size])): raise ValueError('invalid cooperative header')
    inner = data[HEADER_BYTES:]
    if wrap(inner, model.hex(), assets.hex(), max_g, seed) != data:
        raise ValueError('noncanonical cooperative stream')
    return inner, dict(receiver_sha256=model.hex(), assets_sha256=assets.hex(), max_g=max_g, seed=seed)
