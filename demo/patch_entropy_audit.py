"""Read-only cost audit of actual decoded symbols (no source frames or training)."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo.chunk_enhancement_codec import configure_torch, decode_enhancement, load_model
from demo.chunk_enhancement_experiment import codec, Run
from demo.chunk_enhancement_model import GaussianStreams
from demo.scalable_codec import atomic_json, file_hash
from demo.scalable_format import frame_hash
from src.models.entropy_models import GaussianEncoder


class AuditStreams(GaussianStreams):
    def __init__(self):
        super().__init__()
        self.records = []

    def decode(self, data, scales):
        symbols = super().decode(data, scales)
        indexes = self.indexes(scales).astype(np.int64)
        values = symbols.cpu().numpy().reshape(-1).astype(np.int64)
        cdf, lengths = self.gaussian.get_cdf_info()
        # Exactly the symbol mapping and bypass coding from rans.cpp.
        mapped = np.abs(values)*2-(values > 0)
        maximum = lengths[indexes]-2
        escaped = mapped >= maximum
        used = np.minimum(mapped, maximum)
        probabilities = (cdf[indexes,used+1]-cdf[indexes,used]).astype(np.float64)/65536
        cdf_bits = -np.log2(probabilities).sum()
        bypass_bits = 0
        for raw in (mapped-maximum)[escaped]:
            n = (int(raw).bit_length()+1)//2
            bypass_bits += 2*(n+n//3+1)
        proxy = -GaussianEncoder.get_prob_train(symbols, scales).log2().sum().item()
        self.records.append(dict(kind="z" if len(self.records)%2 == 0 else "y",
            actual_bytes=len(data), symbols=int(len(values)), nonzero=int(np.count_nonzero(values)),
            escaped=int(escaped.sum()), scales_above_16=int((scales > 16).sum().item()),
            max_scale=float(scales.max()), max_abs_symbol=int(np.abs(values).max()),
            gaussian_proxy_bits=proxy, native_cdf_bits=float(cdf_bits), bypass_bits=int(bypass_bits),
            coder_framing_rounding_bits=float(8*len(data)-cdf_bits-bypass_bits)))
        return symbols


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--max-hours", type=float, default=2)
    args = p.parse_args()
    root = args.root
    args.command, args.output = "entropy_audit", root / "entropy_audit"
    run = Run(args)
    run.thread.start()
    configure_torch()
    try:
        model_path = root / "pad16_initial.pt"
        model, base_codec = load_model(model_path), codec()
        rows = []
        samples = json.loads((root / "padding_evaluation/summary.json").read_text())["results"]
        for row in samples:
            run.check()
            sid = row["sample"]["sample_id"]
            model.entropy = AuditStreams()
            stream_path = root / "padding_evaluation" / sid / "q1.acse"
            output, _ = decode_enhancement(model, model_path, base_codec, stream_path.read_bytes())
            assert frame_hash(output) == row["points"]["q1"]["fresh_decode"]["output_hash"]
            records = model.entropy.records
            totals = {k:sum(r[k] for r in records) for k in ("actual_bytes", "symbols", "nonzero", "escaped",
                      "scales_above_16", "gaussian_proxy_bits", "native_cdf_bits", "bypass_bits", "coder_framing_rounding_bits")}
            saved = dict(sample_id=sid, totals=totals, entropy_streams=records,
                         stream_sha256=file_hash(stream_path), checkpoint_sha256=file_hash(model_path),
                         actual_decode_unchanged=True, no_source_read=True)
            atomic_json(args.output / f"{sid}.json", saved)
            rows.append(saved)
            print(json.dumps(dict(sample=sid, **totals)), flush=True)
        atomic_json(args.output / "summary.json", dict(results=rows, code_sha256=file_hash(Path(__file__))))
        run.log_resources()
    finally:
        run.stop.set()
        run.thread.join(timeout=2)
        run.lock.close()


if __name__ == "__main__":
    main()
