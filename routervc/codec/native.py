"""Native. Extracted from the validated RouterVC research implementation."""

from __future__ import annotations
from pathlib import Path
import hashlib
import os

REPO = Path(__file__).resolve().parents[2]


def cuda_includes():
    import sysconfig
    from torch.utils.cpp_extension import CUDA_HOME

    if not CUDA_HOME:
        raise RuntimeError("CUDA toolkit/nvcc required; set CUDA_HOME before compiling")
    root = Path(CUDA_HOME)
    packages = Path(sysconfig.get_paths()["purelib"])
    paths = [
        packages / "nvidia/cu13/include",
        packages / "nvidia/cuda_runtime/include",
        root / "targets/x86_64-linux/include",
        root / "include",
    ]
    return [str(p) for p in paths if p.is_dir()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_bridge():
    import torch
    import inference_extensions_cuda as native
    from torch.utils.cpp_extension import load

    folder = REPO / "src/layers/extensions/inference"
    source = Path(__file__).with_name("native_bridge.cpp")
    binary = Path(native.__file__).resolve()
    pins = {str(p.relative_to(REPO)): sha(p) for p in sorted(folder.rglob("*.h"))}
    pins.update(native_library_sha256=sha(binary), bridge_sha256=sha(source))
    identity = hashlib.sha256(repr(sorted(pins.items())).encode()).hexdigest()[:16]
    os.environ.setdefault("MAX_JOBS", "1")
    module = load(
        name="routervc_latent_native_" + identity,
        sources=[str(source)],
        extra_include_paths=[
            str(folder),
            str(REPO / "src/cpp/py_rans"),
            *cuda_includes(),
        ],
        extra_cflags=["-O1", "-Wno-deprecated-declarations"],
        extra_ldflags=[str(binary), "-Wl,-rpath," + str(binary.parent)],
        verbose=True,
    )
    return (module, pins)
