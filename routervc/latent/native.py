"""Version-bound sidecar to the unchanged native UF CUDA library.

This probe exposes copies of native symbols and replays the SAME native kernels.
It neither substitutes a PyTorch approximation nor rebuilds the legacy library.
The private-buffer bridge is intentionally a research-only, single-process API.
"""
from pathlib import Path
import hashlib
import os

REPO = Path(__file__).resolve().parents[2]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_bridge():
    import torch  # Load libtorch before the separately installed CUDA library.
    import inference_extensions_cuda as native
    from torch.utils.cpp_extension import load
    folder = REPO / 'src/layers/extensions/inference'
    source = Path(__file__).with_name('native_bridge.cpp')
    binary = Path(native.__file__).resolve()
    pins = {str(p.relative_to(REPO)): sha(p) for p in sorted(folder.rglob('*.h'))}
    pins.update(native_library_sha256=sha(binary), bridge_sha256=sha(source))
    identity = hashlib.sha256(repr(sorted(pins.items())).encode()).hexdigest()[:16]
    os.environ.setdefault('MAX_JOBS', '1')
    cuda_root = Path(os.environ['CUDA_HOME'])
    module = load(name='routervc_latent_native_' + identity, sources=[str(source)],
        extra_include_paths=[str(folder), str(REPO/'src/cpp/py_rans'),
                             str(cuda_root/'lib/python3.12/site-packages/nvidia/cu13/include'),
                             str(cuda_root/'targets/x86_64-linux/include')],
        extra_cflags=['-O1', '-Wno-deprecated-declarations'],
        extra_ldflags=[str(binary), '-Wl,-rpath,' + str(binary.parent)], verbose=True)
    return module, pins


if __name__ == '__main__':
    _, bindings = load_bridge()
    print('Native latent bridge built:', bindings['native_library_sha256'], flush=True)
