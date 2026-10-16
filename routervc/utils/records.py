"""Records. Extracted from the validated RouterVC research implementation."""

from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def immutable(path, value):
    if Path(path).exists():
        if read(path) != value:
            raise ValueError(
                f"changed bound inputs/configuration: {path}; use a new output"
            )
    else:
        save(path, value)
