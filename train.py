"""Train independent RouterVC routers or the SeedVR2 LoRA, with checkpoint resume."""

import argparse
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=("sender", "receiver", "generator"))
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="compact labels or G training protocol",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=20261011)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument(
        "--stop-after", type=int, default=0, help="checkpoint/restart test only"
    )
    p.add_argument("--arm", choices=("local", "mixed"), default="mixed")
    a = p.parse_args()
    if not os.environ.get("TMUX"):
        p.error("run long training inside tmux")
    os.environ["ROUTERVC_ASSETS"] = str(a.assets.resolve())
    if a.kind == "generator":
        from routervc.training.generator import train

        a.protocol = a.manifest
    else:
        from routervc.training.routers import train
    train(a)


if __name__ == "__main__":
    main()
