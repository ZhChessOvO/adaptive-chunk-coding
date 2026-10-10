"""Boundary-centered perceptual diagnostics, separate from whole-frame LPIPS."""
import numpy as np
import torch

from routervc.fusion.boundaries import CATEGORIES


def crop_sites(shape, boundaries, size=128, frames=(0, 8, 16)):
    _, h, w, _ = shape
    sites = {c:[] for c in CATEGORIES}
    for edge in boundaries:
        center = (edge['lo']+edge['hi'])//2
        x, y = (edge['pos'], center) if edge['axis']=='x' else (center, edge['pos'])
        x, y = int(np.clip(x-size//2, 0, w-size)), int(np.clip(y-size//2, 0, h-size))
        for t in frames:
            if t in edge['frames']: sites[edge['category']].append((t, x, y))
    return sites


@torch.no_grad()
def boundary_lpips(source, variants, boundaries, model, *, check=lambda: None):
    sites = crop_sites(source.shape, boundaries)
    unique = sorted({key for values in sites.values() for key in values})
    cache = {name:{} for name in variants}
    size, batch = 128, 12
    for start in range(0, len(unique), batch):
        check(); chosen = unique[start:start+batch]
        def packed(pixels):
            arr = np.stack([pixels[t, y:y+size, x:x+size] for t, x, y in chosen])
            return torch.from_numpy(arr.copy()).permute(0, 3, 1, 2).float()/127.5-1
        target = packed(source)
        for name, values in variants.items():
            scores = model(target, packed(values)).flatten().tolist()
            cache[name].update(zip(chosen, scores))
    return dict(definition='LPIPS on 128px boundary-centered context crops, frames 1/9/17 when eligible; not pure 16px strip LPIPS',
                sites={k:[list(v) for v in values] for k,values in sites.items()},
                unique_crops=len(unique),
                metrics={category:{name:float(np.mean([cache[name][s] for s in values]))
                                   if values else None for name in variants}
                         for category,values in sites.items()})
