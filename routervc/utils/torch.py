"""Torch. Extracted from the validated RouterVC research implementation."""

from __future__ import annotations
import io
from pathlib import Path
import torch
from routervc.utils.io import atomic_bytes


def configure_torch():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def atomic_torch(path, data):
    buffer = io.BytesIO()
    torch.save(data, buffer)
    atomic_bytes(Path(path), buffer.getvalue())
