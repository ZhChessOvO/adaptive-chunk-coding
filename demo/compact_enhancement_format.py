"""ACSE v2: lossless compact framing for the single-layer neural codecs.

Only the declared schema is supported; unknown fields are rejected, never
silently dropped. Codec constants are versioned here, SHA256s remain complete,
and a float64 qstep preserves the exact receiver setting. No packet count or
dependency on previous enhancement packets: complete boundaries are prefixes.
"""
import struct
import zlib

from demo.scalable_format import (
    HEADER, MAGIC, PACKET, MAX_PAYLOAD, Container, Packet, sha256,
    validate_metadata, _positive,
)

VERSION = 2
PACKET_MAGIC = b"ERP2"
GLOBAL = struct.Struct("<HHIB")
LOCAL = struct.Struct("<IIB4Hd")
CRC = struct.Struct("<I")
CODECS = ("uf_chunk_single_v1", "uf_feature_head_single_v1", "uf_feature_head_pad16_v2")
HASHES = ("base_sha256", "base_rgb_sha256", "model_i_sha256", "model_p_sha256",
          "enhancement_model_sha256")
CONSTANTS = dict(base_codec="dcvc_uf_hts_scalar", base_qp=8,
                 display_format="rgb_u8", skip_thres=0.0,
                 feature_format="uf_hts_F_ctx_v1", generation="disabled")
GLOBAL_KEYS = set(CONSTANTS) | set(HASHES) | {"width", "height", "frame_count", "enhancement_codec"}
LOCAL_KEYS = {"packet_id", "start", "count", "roi", "qstep", "codec"}


def base_container(base, metadata):
    meta = dict(metadata, base_sha256=sha256(base))
    validate_metadata(meta)
    if set(meta) != GLOBAL_KEYS or any(meta[k] != v for k, v in CONSTANTS.items()):
        raise ValueError("compact framing supports only the declared neural schema")
    if not base or meta["enhancement_codec"] not in CODECS:
        raise ValueError("empty base or unsupported compact codec")
    hashes = []
    for key in HASHES:
        value = meta[key]
        if not isinstance(value, str) or len(value) != 64 or value != value.lower():
            raise ValueError(f"invalid canonical SHA256: {key}")
        raw = bytes.fromhex(value)
        if len(raw) != 32:
            raise ValueError("invalid hash length")
        hashes.append(raw)
    body = GLOBAL.pack(meta["width"], meta["height"], meta["frame_count"],
                       CODECS.index(meta["enhancement_codec"])) + b"".join(hashes)
    packed = body + CRC.pack(zlib.crc32(body))
    return HEADER.pack(MAGIC, VERSION, len(packed), len(base)) + packed + base


def packet_bytes(meta, payload, *, codec):
    import math
    if set(meta) != LOCAL_KEYS or codec not in CODECS or meta["codec"] != codec:
        raise ValueError("unsupported compact packet schema/codec")
    _positive(meta["packet_id"], "packet_id", 2**31-1)
    _positive(meta["count"], "count", 8)
    if type(meta["start"]) is not int or not 0 <= meta["start"] < 100000:
        raise ValueError("invalid compact frame start")
    roi, q = meta["roi"], meta["qstep"]
    if (not isinstance(roi, list) or len(roi) != 4
            or any(type(v) is not int or not 0 <= v <= 8192 for v in roi)
            or min(roi[2:]) < 1 or type(q) not in (int, float)
            or not math.isfinite(q) or not 0.125 <= q <= 8):
        raise ValueError("invalid compact ROI/quality")
    if not 0 < len(payload) <= MAX_PAYLOAD:
        raise ValueError("invalid compact payload length")
    local = LOCAL.pack(meta["packet_id"], meta["start"], meta["count"], *roi, q)
    body = local + payload
    return PACKET.pack(PACKET_MAGIC, len(local), len(payload), zlib.crc32(body)) + body


def parse(data, *, allow_incomplete_tail=False):
    if len(data) < HEADER.size:
        raise ValueError("truncated compact header")
    magic, version, nm, nb = HEADER.unpack_from(data)
    expected = GLOBAL.size + 32*len(HASHES) + CRC.size
    if magic != MAGIC or version != VERSION or nm != expected or nb == 0:
        raise ValueError("invalid compact header")
    base_end = HEADER.size + nm + nb
    if base_end > len(data):
        raise ValueError("truncated compact base")
    packed = data[HEADER.size:HEADER.size+nm]
    if zlib.crc32(packed[:-CRC.size]) != CRC.unpack_from(packed, nm-CRC.size)[0]:
        raise ValueError("compact metadata checksum mismatch")
    width, height, frames, codec_index = GLOBAL.unpack_from(packed)
    if codec_index >= len(CODECS):
        raise ValueError("unknown compact enhancement codec")
    meta = dict(CONSTANTS, width=width, height=height, frame_count=frames,
                enhancement_codec=CODECS[codec_index])
    for i, key in enumerate(HASHES):
        offset = GLOBAL.size+i*32
        meta[key] = packed[offset:offset+32].hex()
    validate_metadata(meta)
    base = data[HEADER.size+nm:base_end]
    if sha256(base) != meta["base_sha256"]:
        raise ValueError("base checksum mismatch")
    pos, packets, ids = base_end, [], set()
    while pos < len(data):
        if len(data)-pos < PACKET.size:
            if allow_incomplete_tail:
                break
            raise ValueError("truncated compact packet header")
        magic, nm, npayload, crc = PACKET.unpack_from(data, pos)
        if magic != PACKET_MAGIC or nm != LOCAL.size or not 0 < npayload <= MAX_PAYLOAD:
            raise ValueError("invalid compact packet header")
        end = pos+PACKET.size+nm+npayload
        if end > len(data):
            if allow_incomplete_tail:
                break
            raise ValueError("truncated compact enhancement packet")
        body = data[pos+PACKET.size:end]
        if zlib.crc32(body) != crc:
            raise ValueError("compact enhancement checksum mismatch")
        pid, start, count, x, y, w, h, q = LOCAL.unpack_from(body)
        pm = dict(packet_id=pid, start=start, count=count, roi=[x, y, w, h],
                  qstep=q, codec=meta["enhancement_codec"])
        # Validate against the exact same schema as the writer.
        packet_bytes(pm, body[nm:], codec=pm["codec"])
        if pid in ids:
            raise ValueError("duplicate packet ID")
        ids.add(pid)
        packets.append(Packet(pm, body[nm:], data[pos:end], end))
        pos = end
    return Container(meta, base, tuple(packets), base_end, pos, len(data)-pos)


def repack(data):
    from demo.scalable_format import parse as parse_any
    parsed = parse_any(data)
    prefix = base_container(parsed.base, parsed.meta)
    return prefix+b"".join(packet_bytes(p.meta, p.payload, codec=parsed.meta["enhancement_codec"])
                            for p in parsed.packets)
