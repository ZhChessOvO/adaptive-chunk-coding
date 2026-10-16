"""121-byte shared decoder profile; no E/G/protection mask is transmitted."""

import struct
import zlib
from routervc.assets import code_identity
from routervc.codec import format as packets

MAGIC = b"RVC20261"
FIELDS = struct.Struct("<8s32s32s32sQBf")
HEADER_BYTES = FIELDS.size + 4


def wrap(inner, receiver_sha256, assets_sha256, max_g=8, seed=20261008):
    p = packets.parse(inner)
    if p["info"]["frames"] != 17:
        raise ValueError("dual-router release currently requires 17 frames")
    if (
        type(max_g) is not int
        or not 0 <= max_g <= 16
        or type(seed) is not int
        or not 0 <= seed < 2**48
        or any(
            type(v) is not str or len(v) != 64 for v in (receiver_sha256, assets_sha256)
        )
    ):
        raise ValueError("invalid shared decoder configuration")
    header = FIELDS.pack(
        MAGIC,
        bytes.fromhex(code_identity("pipeline")),
        bytes.fromhex(receiver_sha256),
        bytes.fromhex(assets_sha256),
        seed,
        max_g,
        0.0,
    )
    return header + struct.pack("<I", zlib.crc32(header)) + inner


def parse(data):
    if len(data) < HEADER_BYTES:
        raise ValueError("incomplete RouterVC header")
    magic, code, model, assets, seed, max_g, reserved = FIELDS.unpack_from(data)
    if (
        magic != MAGIC
        or code.hex() != code_identity("pipeline")
        or reserved != 0
        or struct.unpack_from("<I", data, FIELDS.size)[0]
        != zlib.crc32(data[: FIELDS.size])
    ):
        raise ValueError("wrong or corrupted RouterVC release")
    inner = data[HEADER_BYTES:]
    if wrap(inner, model.hex(), assets.hex(), max_g, seed) != data:
        raise ValueError("noncanonical stream")
    return inner, dict(
        receiver_sha256=model.hex(), assets_sha256=assets.hex(), max_g=max_g, seed=seed
    )
