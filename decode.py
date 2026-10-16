"""Fresh RouterVC decoder: stream and shared assets only, never source frames."""

import argparse
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--g-variant", choices=("formal", "candidate800"), default="formal")
    p.add_argument("--disable-generation", action="store_true")
    a = p.parse_args()
    os.environ["ROUTERVC_ASSETS"] = str(a.assets.resolve())
    from routervc.pipeline import decode
    from routervc.data.frames import save_frames
    from routervc.utils.io import atomic_json

    _, _, pixels, report = decode(
        a.input.read_bytes(),
        a.g_variant,
        a.disable_generation,
        progress=lambda **kw: print(kw, flush=True),
    )
    save_frames(a.output, pixels)
    atomic_json(a.output / "report.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
