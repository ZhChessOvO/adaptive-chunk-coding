"""Native-scale field-of-view geometry and spatially paired diffusion noise."""
import torch


def window(shape, core, size):
    _, height, width, _ = shape
    x, y, w, h = core
    if min(x, y) < 0 or min(w, h) <= 0 or x+w > width or y+h > height:
        raise ValueError('core outside input')
    if size % 16 or size < max(w, h) or size > min(width, height):
        raise ValueError('unsupported native-scale window')
    left = min(max(0, ((x+w//2-size//2)//8)*8), width-size)
    top = min(max(0, ((y+h//2-size//2)//8)*8), height-size)
    if left % 8 or top % 8 or left > x or top > y or left+size < x+w or top+size < y+h:
        raise ValueError('unaligned or uncovered core')
    return [left, top, size, size]


def geometry(shape, old_core):
    x, y, w, h = old_core
    if min(w, h) < 128:
        raise ValueError('original core smaller than comparison area')
    core = [x+(w-128)//2, y+(h-128)//2, 128, 128]
    small, large = window(shape, core, 256), window(shape, core, 512)
    if not (large[0] <= small[0] and large[1] <= small[1]
            and small[0]+256 <= large[0]+512 and small[1]+256 <= large[1]+512):
        raise ValueError('windows must be nested')
    return dict(core=core, small=small, large=large)


def rgb_crop(pixels, crop):
    x, y, w, h = crop
    result = pixels[:, y:y+h, x:x+w]
    if result.shape[1:3] != (h, w):
        raise ValueError('crop outside pixels')
    return result.copy()


def local_core(pixels, crop, core):
    x, y, w, h = core
    return rgb_crop(pixels, [x-crop[0], y-crop[1], w, h])


def master_noise(seed, device='cpu'):
    # SeedVR2: 17 RGB frames -> five latent frames, spatial stride eight.
    generator = torch.Generator(device='cpu').manual_seed(seed)
    return torch.randn((5, 64, 64, 16), generator=generator,
                       dtype=torch.bfloat16).to(device)


def noise_crop(master, crop, outer):
    x, y, w, h = crop
    dx, dy = x-outer[0], y-outer[1]
    if min(dx, dy) < 0 or any(v % 8 for v in (dx, dy, w, h)):
        raise ValueError('unaligned noise coordinates')
    result = master[:, dy//8:(dy+h)//8, dx//8:(dx+w)//8, :].contiguous()
    if result.shape != (5, h//8, w//8, 16):
        raise ValueError('noise crop outside field')
    return result
