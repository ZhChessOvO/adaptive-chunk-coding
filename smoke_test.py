"""Run current CPU regressions; use --gpu for a fresh-process codec round trip."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--assets", type=Path)
    a = p.parse_args()
    subprocess.run([sys.executable, "-m", "unittest", "tests.test_release"], check=True)
    if not a.gpu:
        return
    if not a.assets:
        p.error("--gpu needs --assets")
    os.environ["ROUTERVC_ASSETS"] = str(a.assets.resolve())
    from routervc.codec.precision import PrecisionCodec
    from routervc.codec import format as wire
    from routervc.utils.precision import codec_precision
    from routervc.utils.io import atomic_bytes, atomic_npz

    x = np.random.default_rng(4).integers(0, 256, (17, 256, 256, 3), dtype=np.uint8)
    with tempfile.TemporaryDirectory(prefix="routervc-smoke-") as folder:
        folder = Path(folder)
        with codec_precision():
            codec = PrecisionCodec()
            bank, base, full, _ = codec.encode(x, 3)
        del codec
        import gc, torch

        gc.collect()
        torch.cuda.empty_cache()
        atomic_bytes(folder / "bank.bin", bank)
        atomic_npz(folder / "expected.npz", base=base, full=full)
        code = """from pathlib import Path
import numpy as np
from routervc.codec.precision import decode
from routervc.codec.format import subset
folder=Path(__import__('sys').argv[1]);bank=(folder/'bank.bin').read_bytes()
with np.load(folder/'expected.npz') as z:
 b,y,r=decode(bank);np.testing.assert_array_equal(b,z['base']);np.testing.assert_array_equal(y,z['full'])
 b,y,r=decode(subset(bank,[]));np.testing.assert_array_equal(b,y)
print('FRESH_DECODE_EXACT')
"""
        subprocess.run([sys.executable, "-c", code, str(folder)], check=True)


if __name__ == "__main__":
    main()
