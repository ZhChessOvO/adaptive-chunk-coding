"""Resources. Extracted from the validated RouterVC research implementation."""

from __future__ import annotations
from datetime import datetime
from datetime import timezone
import shutil
import os
from pathlib import Path


def snapshot():
    """Resource receipt with cgroup usage separate from the host's cache view."""
    import psutil
    import torch

    paths = os.environ.get(
        "ROUTERVC_STORAGE_PATHS", "/root:/root/autodl-tmp:/root/autodl-fs"
    ).split(":")
    disks = {}
    for path in paths:
        if Path(path).exists():
            d = shutil.disk_usage(path)
            disks[path] = dict(total=d.total, used=d.used, free=d.free)
    memory = {}
    for name in (
        "/sys/fs/cgroup/memory.current",
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        "/sys/fs/cgroup/memory/memory.failcnt",
    ):
        if Path(name).is_file():
            memory[Path(name).name] = Path(name).read_text().strip()
    return dict(
        utc=now(),
        disks=disks,
        process_rss=psutil.Process().memory_info().rss,
        cgroup=memory,
        cuda_allocated=(
            torch.cuda.memory_allocated() if torch.cuda.is_initialized() else 0
        ),
        cuda_peak_allocated=(
            torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0
        ),
    )


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def check_space() -> None:
    paths = os.environ.get(
        "ROUTERVC_STORAGE_PATHS", "/root:/root/autodl-tmp:/root/autodl-fs"
    ).split(":")
    for path in paths:
        if not os.path.exists(path):
            continue
        usage = shutil.disk_usage(path)
        if usage.used / usage.total >= 0.8:
            raise RuntimeError(f"80% disk stop: {path}")
