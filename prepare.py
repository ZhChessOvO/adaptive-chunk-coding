"""Prepare measured, restartable labels/received G inputs from a frame manifest."""

import argparse
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=("sender", "receiver", "generator"))
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--width", type=int, choices=(3, 5, 7, 9), default=3)
    p.add_argument("--g-variant", choices=("formal", "candidate800"), default="formal")
    p.add_argument("--steps", type=int, default=800)
    a = p.parse_args()
    if not os.environ.get("TMUX"):
        p.error("run teacher preparation inside tmux")
    os.environ["ROUTERVC_ASSETS"] = str(a.assets.resolve())
    from routervc.data.prepare import prepare

    prepare(a)


if __name__ == "__main__":
    main()
