"""Native B-only reference transition, independent of completed probe modules."""
from pathlib import Path
import os
from routervc.latent.native import REPO,sha


def load_transition():
    import torch
    import inference_extensions_cuda as native
    from torch.utils.cpp_extension import load
    source=Path(__file__).with_name('transition_bridge.cpp')
    binary=Path(native.__file__).resolve()
    folder=REPO/'src/layers/extensions/inference'
    cuda=Path(os.environ['CUDA_HOME'])
    identity=sha(source)[:12]+sha(binary)[:8]
    return load(name='routervc_latent_transition_'+identity,sources=[str(source)],
        extra_include_paths=[str(folder),str(REPO/'src/cpp/py_rans'),
            str(cuda/'lib/python3.12/site-packages/nvidia/cu13/include'),str(cuda/'targets/x86_64-linux/include')],
        extra_cflags=['-O1','-Wno-deprecated-declarations'],
        extra_ldflags=[str(binary),'-Wl,-rpath,'+str(binary.parent)],verbose=True)
