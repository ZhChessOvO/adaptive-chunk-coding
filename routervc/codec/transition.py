"""Transition. Extracted from the validated RouterVC research implementation."""

from __future__ import annotations
from pathlib import Path
import os
from routervc.codec.native import REPO
from routervc.codec.native import sha
from routervc.codec.native import cuda_includes


def load_transition():
    import torch
    import inference_extensions_cuda as native
    from torch.utils.cpp_extension import load

    source = Path(__file__).with_name("transition_bridge.cpp")
    binary = Path(native.__file__).resolve()
    folder = REPO / "src/layers/extensions/inference"
    identity = sha(source)[:12] + sha(binary)[:8]
    return load(
        name="routervc_latent_transition_" + identity,
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
