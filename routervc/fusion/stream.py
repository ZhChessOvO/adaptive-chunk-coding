"""Charged shared fusion profile around the original literal-prefix stream.

No spatial maps are serialized. The 77-byte envelope identifies code, a shared
checkpoint when needed, and the display mode. Baseline controls pay it too.
"""
import hashlib
from pathlib import Path
import struct
import zlib

from demo.scalable_codec import file_hash
from routervc.latent import routing

MAGIC = b'RVLFUS01'
FIELDS = struct.Struct('<8s32s32sB')
HEADER_BYTES = FIELDS.size+4
MODES = ('current', 'overlap', 'multiband', 'learned')


def identity():
    folder = Path(__file__).resolve().parent
    files = ('stream.py', 'model.py', 'inputs.py', 'blend.py', 'capture.py', 'receive.py')
    return hashlib.sha256((routing.identity()+''.join(file_hash(folder/n) for n in files)).encode()).hexdigest()


def wrap(original, mode, model_sha256='0'*64):
    routing.parse(original)
    if mode not in MODES or len(model_sha256) != 64:
        raise ValueError('invalid fusion configuration')
    if (mode == 'learned') == (model_sha256 == '0'*64):
        raise ValueError('only learned fusion requires a shared checkpoint identity')
    head = FIELDS.pack(MAGIC, bytes.fromhex(identity()), bytes.fromhex(model_sha256), MODES.index(mode))
    return head+struct.pack('<I', zlib.crc32(head))+original


def parse(data):
    if len(data) < HEADER_BYTES: raise ValueError('incomplete fusion header')
    tag, code, model, index = FIELDS.unpack_from(data)
    if tag != MAGIC or code.hex() != identity() or index >= len(MODES):
        raise ValueError('unknown fusion profile')
    original = data[HEADER_BYTES:]
    if wrap(original, MODES[index], model.hex()) != data:
        raise ValueError('corrupt/noncanonical fusion envelope')
    return original, dict(mode=MODES[index], model_sha256=model.hex())
