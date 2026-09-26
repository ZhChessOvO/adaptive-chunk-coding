"""Fill two native-UF reference points on the same four development videos."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from demo.chunk_enhancement_codec import configure_torch
from demo.chunk_enhancement_experiment import Run, MECHANISM, codec, read
from demo.chunk_enhancement_evaluate import roi_psnr
from demo.scalable_experiment import load_source, quality
from demo.scalable_codec import atomic_bytes, atomic_json, file_hash
from demo.stage_c_three_path_roi_probe import LPIPSAlex


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-hours", type=float, default=2)
    args = p.parse_args()
    args.command = "uf_references"
    run = Run(args)
    run.thread.start()
    try:
        configure_torch()
        base_codec, metric = codec(), LPIPSAlex(True)
        entries = read(MECHANISM / "summary.json")["results"]
        protocol = {"qps": [16, 24], "samples": [e["sample"] for e in entries],
                    "data_role": "same previously used development videos",
                    "models": base_codec.models, "code_sha256": file_hash(Path(__file__))}
        if (args.output / "protocol.json").exists() and read(args.output / "protocol.json") != protocol:
            raise RuntimeError("reference protocol changed")
        atomic_json(args.output / "protocol.json", protocol)
        rows = []
        for entry in entries:
            sample, rois = entry["sample"], entry["rois"]
            source = load_source(sample)
            points = {}
            for qp in protocol["qps"]:
                run.check()
                root = args.output / sample["sample_id"] / f"qp{qp}"
                root.mkdir(parents=True, exist_ok=True)
                result_path = root / "result.json"
                if result_path.exists():
                    result = read(result_path)
                    for relative, digest in result["artifacts"].items():
                        if file_hash(root / relative) != digest:
                            raise RuntimeError("reference artifact checksum mismatch")
                else:
                    stream, _ = base_codec.encode(source, qp)
                    atomic_bytes(root / "stream.dcvc", stream)
                    expected = base_codec.decode(stream, len(source))
                    command = [sys.executable, str(REPO / "demo/stage_c_a800_scalar_fresh_decode.py"),
                               "--stream", str(root / "stream.dcvc"), "--frame-count", str(len(source)),
                               "--decode-repeats", "1", "--output", str(root / "decode.json"),
                               "--output-frames-dir", str(root / "frames")]
                    with (root / "decode.log").open("w") as log:
                        subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                       timeout=600, check=True)
                    actual = np.stack([np.asarray(Image.open(f).convert("RGB")) for f in sorted((root / "frames").glob("*.png"))])
                    np.testing.assert_array_equal(actual, expected)
                    result = {"bytes": (root / "stream.dcvc").stat().st_size, "roi_psnr": roi_psnr(source, actual, rois),
                              "quality": quality(source, actual, metric), "fresh_decode_exact": True,
                              "artifacts": {str(f.relative_to(root)): file_hash(f) for f in root.rglob("*") if f.is_file()}}
                    atomic_json(result_path, result)
                points[f"uf{qp}"] = result
                print(json.dumps({"sample": sample["sample_id"], "qp": qp, "bytes": result["bytes"], "roi_psnr": result["roi_psnr"]}), flush=True)
                run.update(completed=len(rows)*2+len(points), total=8)
            rows.append({"sample": sample, "points": points})
        atomic_json(args.output / "summary.json", {"protocol": protocol, "results": rows, "seconds": time.monotonic()-run.started})
        atomic_json(args.output / "uf_references.complete", {"passed": True})
        run.log_resources()
    finally:
        run.stop.set()
        run.thread.join(timeout=2)
        run.lock.close()


if __name__ == "__main__":
    main()
