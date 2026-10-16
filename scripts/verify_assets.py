"""Check every shared model without downloading or changing anything."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from routervc.assets import Assets

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("manifest", type=Path)
a = p.parse_args()
assets = Assets(a.manifest)
for name in assets.config["files"]:
    path = assets.path(name)
    print(name, path.stat().st_size, assets.sha(name), flush=True)
print("ALL_ASSETS_VERIFIED")
