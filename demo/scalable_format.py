"""Append-only regional enhancement container and deterministic mechanism codec.

ACSE v1 embeds an ordinary scalar DCVC-UF stream. It has no total packet count,
so every complete packet boundary is a valid prefix. JSON metadata is explicitly
charged. This unoptimized framing and integer-Haar/zlib payload are a mechanism
baseline, NOT the proposed conditional neural enhancement codec.

Display corrections never enter the base codec's decoded-picture buffer.
"""

from __future__ import annotations

import hashlib
import json
import struct
import zlib
from dataclasses import dataclass

import numpy as np


MAGIC = b"ACSE"
VERSION = 1
HEADER = struct.Struct("<4sBIQ")
PACKET = struct.Struct("<4sIII")
PACKET_MAGIC = b"ERP1"
MAX_METADATA = 1 << 20
MAX_PAYLOAD = 256 << 20
CODEC = "integer_haar_zlib_v1"


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def frame_hash(frames: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(frames).tobytes())


def _positive(value, name: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"invalid {name}: {value}")
    return value


def validate_metadata(meta: dict) -> None:
    if meta.get("base_codec") != "dcvc_uf_hts_scalar":
        raise ValueError("unsupported base codec")
    if meta.get("display_format") != "rgb_u8" or meta.get("base_qp") != 8:
        raise ValueError("v1 mechanism requires RGB8 display and scalar QP8")
    for key, maximum in (("width", 8192), ("height", 8192),
                         ("frame_count", 100000)):
        _positive(meta.get(key), key, maximum)
    for key in ("base_sha256", "base_rgb_sha256", "model_i_sha256", "model_p_sha256"):
        value = meta.get(key, "")
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"missing/invalid {key}")
        int(value, 16)
    if meta.get("skip_thres") != 0.0:
        raise ValueError("mechanism v1 fixes the base skip threshold to zero")


@dataclass(frozen=True)
class Packet:
    meta: dict
    payload: bytes
    wire: bytes
    end_offset: int = 0


@dataclass(frozen=True)
class Container:
    meta: dict
    base: bytes
    packets: tuple[Packet, ...]
    base_end: int
    consumed_bytes: int
    incomplete_tail_bytes: int


def base_container(base: bytes, metadata: dict) -> bytes:
    meta = dict(metadata, base_sha256=sha256(base))
    validate_metadata(meta)
    packed = canonical_json(meta)
    if not base or len(packed) > MAX_METADATA:
        raise ValueError("empty base or oversized metadata")
    return HEADER.pack(MAGIC, VERSION, len(packed), len(base)) + packed + base


def packet_bytes(meta: dict, payload: bytes) -> bytes:
    packed = canonical_json(meta)
    if len(packed) > MAX_METADATA or not 0 < len(payload) <= MAX_PAYLOAD:
        raise ValueError("invalid packet length")
    body = packed + payload
    return PACKET.pack(PACKET_MAGIC, len(packed), len(payload),
                       zlib.crc32(body)) + body


def parse(data: bytes, *, allow_incomplete_tail: bool = False) -> Container:
    if len(data) >= 5 and data[:4] == MAGIC and data[4] == 2:
        from demo.compact_enhancement_format import parse as parse_compact
        return parse_compact(data, allow_incomplete_tail=allow_incomplete_tail)
    if len(data) < HEADER.size:
        raise ValueError("truncated global header")
    magic, version, nmeta, nbase = HEADER.unpack_from(data)
    if magic != MAGIC or version != VERSION:
        raise ValueError("unsupported container magic/version")
    if not 0 < nmeta <= MAX_METADATA or nbase == 0:
        raise ValueError("invalid base lengths")
    base_end = HEADER.size + nmeta + nbase
    if base_end > len(data):
        raise ValueError("truncated base layer")
    meta = json.loads(data[HEADER.size:HEADER.size + nmeta])
    validate_metadata(meta)
    base = data[HEADER.size + nmeta:base_end]
    if sha256(base) != meta["base_sha256"]:
        raise ValueError("base checksum mismatch")
    packets = []
    pos = base_end
    ids = set()
    while pos < len(data):
        if len(data) - pos < PACKET.size:
            if allow_incomplete_tail:
                break
            raise ValueError("truncated packet header")
        magic, nm, npayload, crc = PACKET.unpack_from(data, pos)
        if magic != PACKET_MAGIC:
            raise ValueError("bad enhancement packet magic")
        if not 0 < nm <= MAX_METADATA or not 0 < npayload <= MAX_PAYLOAD:
            raise ValueError("invalid enhancement lengths")
        end = pos + PACKET.size + nm + npayload
        if end > len(data):
            if allow_incomplete_tail:
                break
            raise ValueError("truncated enhancement packet")
        body = data[pos + PACKET.size:end]
        if zlib.crc32(body) != crc:
            raise ValueError("enhancement checksum mismatch")
        pm = json.loads(body[:nm])
        packet_id = _positive(pm.get("packet_id"), "packet_id", 2**31 - 1)
        if packet_id in ids:
            raise ValueError("duplicate packet ID")
        ids.add(packet_id)
        packets.append(Packet(pm, body[nm:], data[pos:end], end))
        pos = end
    return Container(meta, base, tuple(packets), base_end, pos, len(data) - pos)


def _lift(x: np.ndarray, axis: int, inverse: bool) -> np.ndarray:
    x = np.moveaxis(x, axis, -1)
    half = x.shape[-1] // 2
    if inverse:
        low, high = x[..., :half], x[..., half:]
        a = low - np.floor_divide(high, 2)
        out = np.empty_like(x)
        out[..., 0::2] = a
        out[..., 1::2] = a + high
    else:
        a, b = x[..., 0::2], x[..., 1::2]
        high = b - a
        low = a + np.floor_divide(high, 2)
        out = np.concatenate((low, high), axis=-1)
    return np.moveaxis(out, -1, axis)


def haar(values: np.ndarray, levels: int, *, inverse: bool = False) -> np.ndarray:
    """Integer lifting on THWC; negative residuals are never clipped."""
    if values.ndim != 4 or values.shape[-1] != 3:
        raise ValueError("expected THWC RGB residual")
    _positive(levels, "haar_levels", 4)
    out = np.asarray(values, dtype=np.int32).copy()
    height, width = out.shape[1:3]
    if height % (1 << levels) or width % (1 << levels):
        raise ValueError("Haar input must be padded to the transform alignment")
    order = reversed(range(levels)) if inverse else range(levels)
    for level in order:
        h, w = height >> level, width >> level
        block = out[:, :h, :w, :]
        if inverse:
            block = _lift(_lift(block, 1, True), 2, True)
        else:
            block = _lift(_lift(block, 2, False), 1, False)
        out[:, :h, :w, :] = block
    return out


def encode_residual(residual: np.ndarray, qstep: int, levels: int = 2) -> bytes:
    _positive(qstep, "qstep", 4096)
    _positive(levels, "haar_levels", 4)
    residual = np.asarray(residual)
    if residual.ndim != 4 or residual.shape[-1] != 3 or residual.size == 0:
        raise ValueError("expected nonempty THWC residual")
    if not np.issubdtype(residual.dtype, np.integer) or np.max(np.abs(
            residual.astype(np.int64))) > 255:
        raise ValueError("expected signed RGB8 reconstruction residual")
    height, width = residual.shape[1:3]
    alignment = 1 << levels
    padded = np.pad(residual.astype(np.int32),
                    ((0, 0), (0, -height % alignment),
                     (0, -width % alignment), (0, 0)), mode="edge")
    coeff = haar(padded, levels)
    symbols = np.sign(coeff) * ((np.abs(coeff) + qstep // 2) // qstep)
    if np.min(symbols) < -32768 or np.max(symbols) > 32767:
        raise ValueError("transform symbols do not fit int16")
    return zlib.compress(symbols.astype("<i2").tobytes(), level=6)


def decode_residual(payload: bytes, shape: tuple, qstep: int, levels: int) -> np.ndarray:
    _positive(qstep, "qstep", 4096)
    _positive(levels, "haar_levels", 4)
    t, h, w, channels = shape
    if channels != 3:
        raise ValueError("expected three channels")
    alignment = 1 << levels
    ph, pw = h + (-h % alignment), w + (-w % alignment)
    expected = t * ph * pw * 3 * 2
    if not 0 < expected <= MAX_PAYLOAD:
        raise ValueError("decoded packet too large")
    decoder = zlib.decompressobj()
    raw = decoder.decompress(payload, expected + 1)
    if len(raw) != expected or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError("invalid residual compressed length or trailing bytes")
    coeff = np.frombuffer(raw, dtype="<i2").astype(np.int32).reshape(t, ph, pw, 3)
    return haar(coeff * qstep, levels, inverse=True)[:, :h, :w]


def packet_region(meta: dict, shape: tuple) -> tuple:
    if meta.get("codec") != CODEC:
        raise ValueError("unknown residual codec")
    t, h, w, _ = shape
    start, count = meta.get("start"), meta.get("count")
    roi = meta.get("roi")
    if type(start) is not int or not isinstance(roi, list) or len(roi) != 4:
        raise ValueError("invalid frame range or ROI")
    _positive(count, "count", t)
    if start < 0 or start + count > t or any(type(v) is not int for v in roi):
        raise ValueError("invalid packet coordinates")
    x, y, rw, rh = roi
    if min(x, y) < 0 or min(rw, rh) <= 0 or x + rw > w or y + rh > h:
        raise ValueError("ROI falls outside image")
    _positive(meta.get("layer"), "layer", 255)
    _positive(meta.get("qstep"), "qstep", 4096)
    _positive(meta.get("haar_levels"), "haar_levels", 4)
    if type(meta.get("parent_id")) is not int or meta["parent_id"] < 0:
        raise ValueError("invalid parent_id")
    return start, count, x, y, rw, rh


def regions_overlap(a: tuple, b: tuple) -> bool:
    ta, na, xa, ya, wa, ha = a
    tb, nb, xb, yb, wb, hb = b
    return (max(ta, tb) < min(ta + na, tb + nb)
            and max(xa, xb) < min(xa + wa, xb + wb)
            and max(ya, yb) < min(ya + ha, yb + hb))


def apply_packets(base: np.ndarray, packets: tuple | list) -> tuple[np.ndarray, dict]:
    """Missing parent packets are skipped. No source or codec state is read."""
    if base.dtype != np.uint8 or base.ndim != 4 or base.shape[-1] != 3:
        raise ValueError("base must be THWC RGB8")
    output = base.copy()
    last = {}
    applied, skipped = [], []
    ids = set()
    for packet in packets:
        meta = packet.meta
        packet_id = _positive(meta.get("packet_id"), "packet_id", 2**31 - 1)
        if packet_id in ids:
            raise ValueError("duplicate packet ID")
        ids.add(packet_id)
        region = packet_region(meta, base.shape)
        for known in last:
            if known != region and regions_overlap(known, region):
                raise ValueError("overlapping enhancement regions create hidden dependencies")
        previous = last.get(region, (0, 0))
        if (meta["parent_id"], meta["layer"] - 1) != previous:
            skipped.append({"packet_id": packet_id, "reason": "missing_or_stale_parent"})
            continue
        start, count, x, y, width, height = region
        correction = decode_residual(packet.payload, (count, height, width, 3),
                                     meta["qstep"], meta["haar_levels"])
        target = output[start:start + count, y:y + height, x:x + width]
        target[:] = np.clip(target.astype(np.int32) + correction, 0, 255).astype(np.uint8)
        last[region] = (packet_id, meta["layer"])
        applied.append(packet_id)
    return output, {"applied_packet_ids": applied, "skipped_packets": skipped}


def make_packet(source: np.ndarray, current: np.ndarray, *, packet_id: int,
                start: int, count: int, roi: list[int], layer: int,
                parent_id: int, qstep: int, levels: int = 2) -> bytes:
    meta = {"packet_id": packet_id, "start": start, "count": count, "roi": roi,
            "layer": layer, "parent_id": parent_id, "qstep": qstep,
            "haar_levels": levels, "codec": CODEC}
    packet_region(meta, source.shape)
    if source.dtype != np.uint8 or current.dtype != np.uint8 or source.shape != current.shape:
        raise ValueError("source/current must have matching RGB8 arrays")
    x, y, width, height = roi
    part = (slice(start, start + count), slice(y, y + height),
            slice(x, x + width), slice(None))
    residual = source[part].astype(np.int32) - current[part].astype(np.int32)
    return packet_bytes(meta, encode_residual(residual, qstep, levels))
