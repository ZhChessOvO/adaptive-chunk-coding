"""Encode one 17-frame RouterVC window with real entropy-coded B/E bytes."""

import argparse
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input", type=Path, required=True, help="ordered RGB frame directory"
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--width", type=int, choices=(3, 5, 7, 9), default=3)
    p.add_argument(
        "--e-ratio", type=float, default=0.5, help="fraction of full E bundle bytes"
    )
    p.add_argument("--max-g", type=int, default=8)
    p.add_argument("--g-variant", choices=("formal", "candidate800"), default="formal")
    p.add_argument(
        "--all-enhancements",
        action="store_true",
        help="diagnostic raster order, bypass R_s",
    )
    a = p.parse_args()
    os.environ["ROUTERVC_ASSETS"] = str(a.assets.resolve())
    from routervc.data.frames import read_frames
    from routervc.pipeline import encode
    from routervc.utils.io import atomic_bytes, atomic_json

    pixels = read_frames(a.input, a.start)
    wire, report = encode(
        pixels,
        a.width,
        a.e_ratio,
        a.max_g,
        a.g_variant,
        all_enhancements=a.all_enhancements,
        progress=lambda **kw: print(kw, flush=True),
    )
    atomic_bytes(a.output, wire)
    if a.output.stat().st_size != report["actual_bytes"]:
        raise RuntimeError("on-disk byte mismatch")
    atomic_json(a.output.with_suffix(a.output.suffix + ".json"), report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
