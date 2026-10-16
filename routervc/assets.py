"""Portable shared assets and explicit identities for the cleaned release.

Model weights are not downloaded implicitly. Paths are relative to assets.json,
not to a previous server or experiment directory. This release intentionally has
a new wire/code identity; archived streams use the archived source revision.
"""

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def code_identity(component):
    groups = {
        "codec": ["codec"],
        "routers": ["routers"],
        "pipeline": ["codec", "routers", "fusion", "generation", "utils"],
    }
    explicit = {
        "fusion": ["composition.py", "feather.py", "multiband.py", "render.py"],
        "generation": ["runtime.py", "runner.py", "lora.py", "engine.py"],
    }
    files = [REPO / "routervc/assets.py"]
    for folder in groups[component]:
        root = REPO / "routervc" / folder
        (
            files.extend(root / name for name in explicit[folder])
            if folder in explicit
            else files.extend(
                p
                for p in root.iterdir()
                if p.suffix in (".py", ".cpp") and p.name != "__init__.py"
            )
        )
    if component == "pipeline":
        files.extend(REPO / "routervc" / name for name in ("stream.py", "pipeline.py"))
    h = hashlib.sha256()
    for p in sorted(files):
        h.update(str(p.relative_to(REPO)).encode() + b"\0" + p.read_bytes())
    return h.hexdigest()


@lru_cache(maxsize=64)
def _hash(path, size, modified):
    from routervc.utils.io import file_hash

    return file_hash(Path(path))


class Assets:
    def __init__(self, manifest=None):
        self.manifest = Path(
            manifest or os.environ.get("ROUTERVC_ASSETS", REPO / "assets/assets.json")
        ).resolve()
        self.config = json.loads(self.manifest.read_text())
        if self.config.get("format") != "routervc_assets_v1":
            raise ValueError("unknown asset manifest")

    def path(self, name):
        spec = self.config["files"][name]
        p = (self.manifest.parent / spec["path"]).resolve()
        stat = p.stat()
        if _hash(str(p), stat.st_size, stat.st_mtime_ns) != spec["sha256"]:
            raise ValueError("asset hash mismatch: " + name)
        return p

    def sha(self, name):
        self.path(name)
        return self.config["files"][name]["sha256"]

    def generator_identity(self, variant="formal"):
        names = ("dit", "vae", "positive", "negative", "g_" + variant)
        return hashlib.sha256(
            json.dumps({n: self.sha(n) for n in names}, sort_keys=True).encode()
        ).hexdigest()

    def upstream(self):
        return (
            self.manifest.parent
            / self.config.get("seedvr2_source", "../third_party/SeedVR2")
        ).resolve()
