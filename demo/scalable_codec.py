#!/usr/bin/env python3
"""Scalar UF base encode/decode and standalone ACSE receiver.

Receiver input is only a saved container and pre-shared UF model weights.
Enhancement is applied after base decoding, never to the UF reference chain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from demo.scalable_format import apply_packets, frame_hash, parse


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, ensure_ascii=False,
                                  allow_nan=False) + "\n").encode())


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class BaseCodec:
    def __init__(self, model_i: Path, model_p: Path):
        import torch
        from demo.stage_c_three_path_roi_probe import load_codecs
        from demo.stage_c_a800_scalar_fresh_decode import initialize_intra_decoder_proxy
        from src.utils.common import set_torch_env

        set_torch_env()
        torch.set_num_threads(4)
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
        self.stream = torch.cuda.Stream(device=self.device)
        torch.cuda.set_stream(self.stream)
        self.models = {"model_i_sha256": file_hash(model_i),
                       "model_p_sha256": file_hash(model_p)}
        args = SimpleNamespace(model_path_i=model_i, model_path_p=model_p, skip_thres=0.0)
        self.i_net, self.p_net = load_codecs(args, self.device)
        initialize_intra_decoder_proxy(self.i_net)
        torch.cuda.synchronize()

    def encode(self, frames: np.ndarray, qp: int = 8) -> tuple[bytes, dict]:
        import torch
        from demo.stage_c_three_path_roi_probe import encode_dcvc_stream

        torch.cuda.set_stream(self.stream)
        with torch.inference_mode():
            return encode_dcvc_stream(list(frames), qp, qp, self.i_net,
                                      self.p_net, self.device, reset_interval=32)

    def decode(self, data: bytes, count: int) -> np.ndarray:
        import torch
        from demo.stage_c_three_path_roi_probe import decode_dcvc_stream

        torch.cuda.set_stream(self.stream)
        with torch.inference_mode():
            frames = decode_dcvc_stream(data, count, self.i_net, self.p_net, self.device)
        torch.cuda.synchronize()
        return np.stack(frames)

    def metadata(self, frames: np.ndarray, base: np.ndarray) -> dict:
        count, height, width, _ = frames.shape
        return {"base_codec": "dcvc_uf_hts_scalar", "base_qp": 8, "skip_thres": 0.0,
                "width": width, "height": height, "frame_count": count,
                "display_format": "rgb_u8", "base_rgb_sha256": frame_hash(base),
                "generation": "disabled", **self.models}


def decode_main(args) -> None:
    started = time.perf_counter()
    data = args.stream.read_bytes()
    parsed = parse(data, allow_incomplete_tail=args.allow_incomplete_tail)
    for key, path in (("model_i_sha256", args.model_i), ("model_p_sha256", args.model_p)):
        if file_hash(path) != parsed.meta[key]:
            raise ValueError(f"pre-shared model fingerprint mismatch: {key}")
    codec = BaseCodec(args.model_i, args.model_p)
    setup_seconds = time.perf_counter() - started
    base = codec.decode(parsed.base, parsed.meta["frame_count"])
    shape = (parsed.meta["frame_count"], parsed.meta["height"], parsed.meta["width"], 3)
    if base.shape != shape or frame_hash(base) != parsed.meta["base_rgb_sha256"]:
        raise RuntimeError("fresh base reconstruction mismatch")
    output, status = apply_packets(base, parsed.packets)
    # Save base only for validation; it is produced by this decoder, not read
    # from the encoder's cache.
    atomic_npz(args.output, base=base, reconstruction=output)
    import torch
    report = {
        "source_frames_read": False, "independent_process_pid": os.getpid(),
        "actual_on_disk_bytes": args.stream.stat().st_size,
        "base_stream_bytes": len(parsed.base),
        "container_header_bytes": parsed.base_end - len(parsed.base),
        "enhancement_packet_bytes": sum(len(p.wire) for p in parsed.packets),
        "incomplete_tail_bytes": parsed.incomplete_tail_bytes,
        "base_sha256": parsed.meta["base_sha256"],
        "base_rgb_sha256": frame_hash(base), "output_rgb_sha256": frame_hash(output),
        "setup_seconds": setup_seconds, "complete_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        **status,
    }
    if (report["base_stream_bytes"] + report["container_header_bytes"]
            + report["enhancement_packet_bytes"] + report["incomplete_tail_bytes"]
            != report["actual_on_disk_bytes"]):
        raise RuntimeError("byte accounting mismatch")
    atomic_json(args.report, report)
    print(json.dumps(report), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    decoder = sub.add_parser("decode")
    decoder.add_argument("--stream", type=Path, required=True)
    decoder.add_argument("--output", type=Path, required=True)
    decoder.add_argument("--report", type=Path, required=True)
    decoder.add_argument("--allow-incomplete-tail", action="store_true")
    decoder.add_argument("--model-i", type=Path, default=REPO / "checkpoints/cvpr2026_image.pth.tar")
    decoder.add_argument("--model-p", type=Path, default=REPO / "checkpoints/cvpr2026_video_hts.pth.tar")
    args = parser.parse_args()
    decode_main(args)


if __name__ == "__main__":
    main()
