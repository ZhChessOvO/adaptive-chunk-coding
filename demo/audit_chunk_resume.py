"""Compare a split/resumed run with its uninterrupted training control."""
import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo.scalable_codec import atomic_json, file_hash


def same(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(v, b[k]) for k, v in a.items())
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("resumed", type=Path)
    p.add_argument("continuous", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    a, b = [torch.load(v, map_location="cpu", weights_only=False) for v in (args.resumed, args.continuous)]
    result = {"inputs": [str(args.resumed), str(args.continuous)],
              "sha256": [file_hash(args.resumed), file_hash(args.continuous)],
              "exact_fields": {k: same(a[k], b[k]) for k in
                               ("step", "config", "model", "optimizer", "random_state", "torch_rng", "cuda_rng")}}
    result["passed"] = all(result["exact_fields"].values())
    atomic_json(args.output, result)
    print(result)
    if not result["passed"]:
        raise RuntimeError("resume differs from uninterrupted training")


if __name__ == "__main__":
    main()
