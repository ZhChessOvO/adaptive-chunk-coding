"""Packet. Extracted from the validated RouterVC research implementation."""

from __future__ import annotations
import hashlib
import struct

EHEADER = struct.Struct("<BBI16s")


def digest(data):
    return hashlib.sha256(data).digest()


def packet(chunk, region, payload, identity):
    if (
        type(chunk) is not int
        or not 0 <= chunk < 8
        or type(region) is not int
        or (not 0 <= region < 16)
    ):
        raise ValueError("bad E address")
    if not 4 <= len(payload) <= 64 << 20 or len(identity) != 32:
        raise ValueError("bad E size/identity")
    check = digest(
        identity + struct.pack("<BBI", chunk, region, len(payload)) + payload
    )[:16]
    return EHEADER.pack(chunk, region, len(payload), check) + payload
