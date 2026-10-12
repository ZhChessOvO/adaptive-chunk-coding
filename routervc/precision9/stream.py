"""121-byte receiving profile, no width/action map; width is in B's manifest."""
import hashlib
from pathlib import Path
import struct
import zlib
import numpy as np
from routervc.latent import routing
from routervc.precision9 import format as packets

MAGIC = b'RVL9CO01'
FIELDS = routing.FIELDS
HEADER_BYTES = routing.HEADER_BYTES


def identity():
    from routervc.cooperation import stream
    from routervc.precision9.codec import identity as codec_identity
    names = ('stream.py', 'receive.py')
    return hashlib.sha256((stream.identity()+codec_identity()+''.join(
        hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names)).encode()).hexdigest()


def wrap(inner, model, assets, max_g=8, seed=routing.SEED):
    p = packets.parse(inner)
    if p['info']['frames'] != 17: raise ValueError('Router profile requires I + two P8')
    if (type(max_g) is not int or not 0 <= max_g <= 16 or type(seed) is not int or not 0 <= seed < 2**48
            or any(type(h) is not str or len(h) != 64 for h in (model, assets))):
        raise ValueError('invalid receiver config')
    header = FIELDS.pack(MAGIC, bytes.fromhex(identity()), bytes.fromhex(model), bytes.fromhex(assets), seed, max_g, 0.)
    return header+struct.pack('<I', zlib.crc32(header))+inner


def parse(data):
    if len(data) < HEADER_BYTES: raise ValueError('incomplete receiving header')
    magic, code, model, assets, seed, max_g, reserved = FIELDS.unpack_from(data)
    if (magic != MAGIC or code.hex() != identity() or reserved != 0 or not np.isfinite(reserved)
            or struct.unpack_from('<I', data, FIELDS.size)[0] != zlib.crc32(data[:FIELDS.size])):
        raise ValueError('invalid receiving profile')
    inner = data[HEADER_BYTES:]
    if wrap(inner, model.hex(), assets.hex(), max_g, seed) != data: raise ValueError('noncanonical stream')
    return inner, dict(receiver_sha256=model.hex(), assets_sha256=assets.hex(), max_g=max_g, seed=seed)
