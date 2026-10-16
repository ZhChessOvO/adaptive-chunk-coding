"""Evaluate actual decoded PNGs, with optional unmodified DCVC-UF baseline."""

import argparse
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--decoded", type=Path)
    p.add_argument("--stream", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--assets", type=Path)
    p.add_argument("--frames", type=int, default=17)
    p.add_argument("--start", type=int, default=0)
    p.add_argument(
        "--uf-qp", type=int, help="encode/fresh native decode baseline at this QP"
    )
    a = p.parse_args()
    if a.assets:
        os.environ["ROUTERVC_ASSETS"] = str(a.assets.resolve())
    from routervc.data.frames import read_frames
    from routervc.codec.uf import LPIPSAlex, evaluate_variant
    from routervc.utils.io import atomic_bytes, atomic_json
    from routervc.codec.uf import BaseCodec
    from routervc.assets import Assets
    from routervc.utils.precision import codec_precision

    source = read_frames(a.source, a.start, a.frames)
    if a.uf_qp is not None:
        if not 0 <= a.uf_qp <= 63:
            p.error("QP outside [0,63]")
        assets = Assets()
        with codec_precision():
            codec = BaseCodec(assets.path("uf_i"), assets.path("uf_p"))
            wire, _ = codec.encode(source, a.uf_qp)
            received = codec.decode(wire, len(source))
        stream_path = a.output.with_suffix(".bin")
        atomic_bytes(stream_path, wire)
        del codec
    else:
        if not a.decoded or not a.stream:
            p.error("--decoded and --stream are required")
        received = read_frames(a.decoded, frames=a.frames)
        stream_path = a.stream
    metric = LPIPSAlex(True)
    result = evaluate_variant(list(source), list(received), metric)
    result.update(
        actual_bytes=stream_path.stat().st_size,
        bpp=stream_path.stat().st_size
        * 8
        / (source.shape[0] * source.shape[1] * source.shape[2]),
    )
    atomic_json(a.output, result)
    print(result)


if __name__ == "__main__":
    main()
