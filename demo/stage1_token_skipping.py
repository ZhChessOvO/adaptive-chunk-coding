#!/usr/bin/env python3
"""Stage 1: real HT-S latent-token skipping with separately counted streams.

The routing unit is one 2x2 spatial block of the HT-S main latent ``y`` across
all 256 channels.  At 512x512 this gives a 16x16 route grid.  One route symbol
therefore controls a spatial unit shared by all eight frames in the P chunk.

Skipped symbols are not passed to rANS.  During reconstruction their quantized
residual is zero, so the existing entropy model's conditional mean is used.
This is intentionally a no-training Stage-1 probe, not the final generator.
"""

import argparse
import csv
import json
import math
import struct
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.image_model import DMCI
from src.models.video_model_ht import DMC, g_ch_y, g_ch_z, g_frame_delay
from src.utils.common import ModelStructure, get_state_dict, set_torch_env
from src.utils.transforms import rgb2ycbcr, ycbcr2rgb
from src.utils.video_reader import PNGReader


MAGIC = b"D1SKIP01"
CONTAINER_HEADER = struct.Struct("<8sBBBBHHHHIIII")
ROUTE_HEADER = struct.Struct("<4sBBBBHHII")
ROUTE_MAGIC = b"RTS1"
FLAG_ALL_BASE = 1
FLAG_Z_EC_SHIFT = 4
FLAG_Z_EC_MASK = 0x70
MIN_SYMBOLS_PER_STREAM = 32768
MAX_EC_PARALLEL = 8


@dataclass
class YCodingResult:
    stream: bytes
    ec_parallel: int
    symbol_count: int
    q_dense: torch.Tensor
    y_hat: torch.Tensor
    expected_bits_by_block: np.ndarray


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a real-bitstream HT-S 2x2x256 latent-token skipping probe.")
    parser.add_argument("--source-dir", default="data/test_sequences/PNG/jockey")
    parser.add_argument("--route-mask",
                        default="output/stage05_prepare_safe/grass_mask_blocks.npy",
                        help="Boolean route-grid .npy; true cells are eligible for skipping.")
    parser.add_argument("--output-dir", default="output/stage1_hts_qp32_grass")
    parser.add_argument("--model-path-i", default="checkpoints/cvpr2026_image.pth.tar")
    parser.add_argument("--model-path-p", default="checkpoints/cvpr2026_video_hts.pth.tar")
    parser.add_argument("--qp-i", type=int, default=32)
    parser.add_argument("--qp-p", type=int, default=32)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--start-frame", type=int, default=1,
                        help="One-based I-frame index; the following 8 frames form the P chunk.")
    parser.add_argument("--latent-block-size", type=int, default=2,
                        help="Spatial y block edge. Stage-1 v1 requires 2.")
    parser.add_argument("--max-skip-blocks", type=int, default=0,
                        help="If positive, retain only the highest estimated-rate candidates.")
    parser.add_argument("--skip-thres", type=float, default=0.0)
    parser.add_argument("--cuda-idx", type=int, default=0)
    parser.add_argument("--compare-stock", action="store_true",
                        help="Also run the stock fused HT-S P-stream encoder for reference.")
    return parser.parse_args()


def compute_ec_parallel(symbol_count):
    return max(1, min(MAX_EC_PARALLEL, symbol_count // MIN_SYMBOLS_PER_STREAM))


def scale_to_index(scales):
    scale_min = 0.11
    scale_max = 16.0
    levels = 128
    scales = np.clip(scales.astype(np.float32), scale_min, scale_max)
    step = (math.log(scale_max) - math.log(scale_min)) / (levels - 1)
    indexes = (np.log(scales) - math.log(scale_min)) / step
    return np.clip(indexes, 0, levels - 1).astype(np.uint8)


def tensor_nhwc_numpy(x, dtype=None):
    out = x.detach().permute(0, 2, 3, 1).contiguous().cpu().numpy()
    return out.astype(dtype, copy=False) if dtype is not None else out


def make_rans_encoder(cdf_info, cdf_slot, ec_parallel):
    from MLCodec_extensions_cpp import RansEncoder
    coder = RansEncoder()
    cdf, cdf_length = cdf_info
    coder.set_cdf(np.asarray(cdf, dtype=np.int32),
                  np.asarray(cdf_length, dtype=np.int32), cdf_slot)
    coder.set_entropy_coder_parallel(ec_parallel)
    coder.reset()
    return coder


def make_rans_decoder(cdf_info, cdf_slot, ec_parallel, stream):
    from MLCodec_extensions_cpp import RansDecoder
    coder = RansDecoder()
    cdf, cdf_length = cdf_info
    coder.set_cdf(np.asarray(cdf, dtype=np.int32),
                  np.asarray(cdf_length, dtype=np.int32), cdf_slot)
    coder.set_entropy_coder_parallel(ec_parallel)
    coder.set_stream(np.frombuffer(stream, dtype=np.uint8).copy())
    return coder


def encode_z(z_hat, qp, cdf_info):
    symbols = tensor_nhwc_numpy(z_hat, np.int8).reshape(-1)
    ec_parallel = compute_ec_parallel(symbols.size)
    coder = make_rans_encoder(cdf_info, 0, ec_parallel)
    coder.encode_z(symbols, qp * g_ch_z, g_ch_z)
    coder.flush()
    return np.asarray(coder.get_encoded_stream(), dtype=np.uint8).tobytes(), ec_parallel


def decode_z(stream, qp, shape, cdf_info, ec_parallel):
    coder = make_rans_decoder(cdf_info, 0, ec_parallel, stream)
    count = int(np.prod(shape))
    coder.decode_z(count, qp * g_ch_z, g_ch_z)
    flat = np.asarray(coder.get_decoded_tensor(), dtype=np.int8).copy()
    n, c, h, w = shape
    arr = flat.reshape(n, h, w, c).transpose(0, 3, 1, 2)
    return torch.from_numpy(arr.copy())


def build_route_section(skip_blocks, block_size):
    skip_blocks = np.asarray(skip_blocks, dtype=np.bool_)
    raw = np.packbits(skip_blocks.reshape(-1), bitorder="little").tobytes()
    payload = zlib.compress(raw, level=9)
    header = ROUTE_HEADER.pack(ROUTE_MAGIC, 1, block_size, block_size, 0,
                               skip_blocks.shape[0], skip_blocks.shape[1],
                               len(raw), len(payload))
    return header + payload


def parse_route_section(section):
    if len(section) < ROUTE_HEADER.size:
        raise ValueError("truncated route section")
    fields = ROUTE_HEADER.unpack(section[:ROUTE_HEADER.size])
    magic, version, block_h, block_w, _, grid_h, grid_w, raw_len, payload_len = fields
    if magic != ROUTE_MAGIC or version != 1 or block_h != block_w:
        raise ValueError("unsupported route section")
    payload = section[ROUTE_HEADER.size:]
    if len(payload) != payload_len:
        raise ValueError("route payload length mismatch")
    raw = zlib.decompress(payload)
    if len(raw) != raw_len:
        raise ValueError("route raw length mismatch")
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")
    return bits[:grid_h * grid_w].reshape(grid_h, grid_w).astype(np.bool_), block_h


def expand_keep_mask(skip_blocks, block_size, y_shape, device):
    skip = torch.from_numpy(skip_blocks).to(device=device)
    skip = skip.repeat_interleave(block_size, 0).repeat_interleave(block_size, 1)
    if tuple(skip.shape) != tuple(y_shape[-2:]):
        raise ValueError(f"route grid expands to {tuple(skip.shape)}, expected {tuple(y_shape[-2:])}")
    return (~skip)[None, None].expand(y_shape[0], y_shape[1], -1, -1)


def y_prior_steps(net, common_params, keep_mask, source_y=None, decoded_q=None):
    if (source_y is None) == (decoded_q is None):
        raise ValueError("provide exactly one of source_y or decoded_q")
    q_enc, q_dec, scales, means = net.separate_prior_video(common_params)
    y_scaled = source_y * q_enc if source_y is not None else None
    reduced = net.y_spatial_prior_reduction(common_params)
    reference = source_y if source_y is not None else decoded_q
    masks = net.get_mask_4x(*reference.shape, reference.device)
    q_dense = torch.zeros_like(reference)
    q_source = torch.zeros_like(reference) if source_y is not None else None
    y_hat_so_far = torch.zeros_like(reference)

    adaptors = [None, net.y_spatial_prior_adaptor_1,
                net.y_spatial_prior_adaptor_2, net.y_spatial_prior_adaptor_3]
    for part, mask in enumerate(masks):
        if part:
            means = net.y_spatial_prior(adaptors[part](y_hat_so_far, reduced))
        if decoded_q is None:
            source = torch.round((y_scaled - means * mask) * mask).clamp_(-128, 127)
            source = source * (scales > net.gaussian_encoder.skip_thres)
            q_source = q_source + source
            q_part = source * keep_mask
        else:
            q_part = decoded_q * mask
        q_dense = q_dense + q_part
        y_hat_so_far = y_hat_so_far + (q_part + means * mask) * mask

    return q_dense, q_source, y_hat_so_far * q_dec, scales


def encode_y(net, y, common_params, skip_blocks, block_size, ec_parallel_override=None):
    keep = expand_keep_mask(skip_blocks, block_size, y.shape, y.device)
    q_dense, q_source, y_hat, scales = y_prior_steps(
        net, common_params, keep, source_y=y)
    active = keep & (scales > net.gaussian_encoder.skip_thres)

    q_hwc = tensor_nhwc_numpy(q_dense, np.int8).reshape(-1)
    scales_hwc = tensor_nhwc_numpy(scales, np.float32).reshape(-1)
    active_hwc = tensor_nhwc_numpy(active, np.bool_).reshape(-1)
    indexes = scale_to_index(scales_hwc)
    q_active = q_hwc[active_hwc].astype(np.int16)
    index_active = indexes[active_hwc].astype(np.int16)
    combined = (q_active * 256 + index_active).astype(np.int16)
    symbol_count = int(combined.size)

    if symbol_count:
        ec_parallel = (
            compute_ec_parallel(symbol_count)
            if ec_parallel_override is None else int(ec_parallel_override))
        if not 1 <= ec_parallel <= MAX_EC_PARALLEL:
            raise ValueError("y entropy-coder parallelism must be in [1, 8]")
        coder = make_rans_encoder(net.gaussian_encoder.get_cdf_info(), 1, ec_parallel)
        coder.encode_y(combined)
        coder.flush()
        stream = np.asarray(coder.get_encoded_stream(), dtype=np.uint8).tobytes()
    else:
        ec_parallel = 1
        stream = b""

    probs = net.gaussian_encoder.get_prob_train(q_source, scales)
    expected = -torch.log2(torch.clamp_min(probs.float(), 1e-9))
    expected = expected.view(1, g_ch_y,
                             skip_blocks.shape[0], block_size,
                             skip_blocks.shape[1], block_size)
    expected = expected.sum(dim=(0, 1, 3, 5)).detach().cpu().numpy()
    return YCodingResult(stream, ec_parallel, symbol_count, q_dense, y_hat, expected)


def decode_y(net, stream, ec_parallel, common_params, skip_blocks, block_size):
    shape = (common_params.shape[0], g_ch_y, common_params.shape[2], common_params.shape[3])
    keep = expand_keep_mask(skip_blocks, block_size, shape, common_params.device)
    _, _, scales, _ = net.separate_prior_video(common_params)
    active = keep & (scales > net.gaussian_encoder.skip_thres)
    active_hwc = tensor_nhwc_numpy(active, np.bool_).reshape(-1)
    scales_hwc = tensor_nhwc_numpy(scales, np.float32).reshape(-1)
    indexes = scale_to_index(scales_hwc)[active_hwc]

    dense_hwc = np.zeros(active_hwc.size, dtype=np.int8)
    if indexes.size:
        coder = make_rans_decoder(net.gaussian_encoder.get_cdf_info(), 1,
                                  ec_parallel, stream)
        coder.decode_y(indexes.astype(np.uint8, copy=False))
        decoded = np.asarray(coder.get_decoded_tensor(), dtype=np.int8).copy()
        if decoded.size != indexes.size:
            raise RuntimeError("decoded y symbol count mismatch")
        dense_hwc[active_hwc] = decoded
    n, c, h, w = shape
    dense = dense_hwc.reshape(n, h, w, c).transpose(0, 3, 1, 2)
    decoded_q = torch.from_numpy(dense.copy()).to(
        device=common_params.device, dtype=common_params.dtype)
    _, _, y_hat, _ = y_prior_steps(
        net, common_params, keep, decoded_q=decoded_q)
    return decoded_q, y_hat


def write_container(path, qp, height, width, latent_shape, global_stream, route_section,
                    base_stream, residual_stream, y_ec_parallel, all_base,
                    z_ec_parallel=None):
    """Write one P-chunk container.

    Version 1 did not serialize the rANS parallelism used by the global ``z``
    stream, so those files cannot independently decode ``z``.  Version 2
    stores ``z_ec_parallel - 1`` in the otherwise unused high flag bits while
    preserving the 36-byte header and backwards readability.
    """
    flags = FLAG_ALL_BASE if all_base else 0
    version = 1
    if z_ec_parallel is not None:
        if not 1 <= z_ec_parallel <= MAX_EC_PARALLEL:
            raise ValueError("z entropy-coder parallelism must be in [1, 8]")
        flags |= (z_ec_parallel - 1) << FLAG_Z_EC_SHIFT
        version = 2
    header = CONTAINER_HEADER.pack(
        MAGIC, version, flags, qp, y_ec_parallel, height, width,
        latent_shape[-2], latent_shape[-1], len(global_stream), len(route_section),
        len(base_stream), len(residual_stream))
    data = header + global_stream + route_section + base_stream + residual_stream
    path.write_bytes(data)
    return len(header), len(data)


def parse_container(data, expected_block_size):
    if len(data) < CONTAINER_HEADER.size:
        raise ValueError("truncated Stage-1 container")
    fields = CONTAINER_HEADER.unpack(data[:CONTAINER_HEADER.size])
    (magic, version, flags, qp, y_ec, height, width, latent_h, latent_w,
     global_len, route_len, base_len, residual_len) = fields
    if magic != MAGIC or version not in (1, 2):
        raise ValueError("unsupported Stage-1 container")
    z_ec = None
    if version >= 2:
        z_ec = ((flags & FLAG_Z_EC_MASK) >> FLAG_Z_EC_SHIFT) + 1
    pos = CONTAINER_HEADER.size
    global_stream = data[pos:pos + global_len]
    pos += global_len
    route = data[pos:pos + route_len]
    pos += route_len
    base = data[pos:pos + base_len]
    pos += base_len
    residual = data[pos:pos + residual_len]
    pos += residual_len
    if pos != len(data):
        raise ValueError("container length mismatch")
    grid_shape = (latent_h // expected_block_size, latent_w // expected_block_size)
    if flags & FLAG_ALL_BASE:
        skip_blocks = np.zeros(grid_shape, dtype=np.bool_)
    else:
        skip_blocks, parsed_block = parse_route_section(route)
        if parsed_block != expected_block_size or skip_blocks.shape != grid_shape:
            raise ValueError("route geometry mismatch")
    return {
        "version": version, "qp": qp, "z_ec": z_ec, "y_ec": y_ec,
        "height": height, "width": width,
        "global": global_stream, "route": route, "base": base,
        "residual": residual, "skip_blocks": skip_blocks,
    }


def read_container(path, expected_block_size):
    return parse_container(Path(path).read_bytes(), expected_block_size)


def read_frames(source_dir, width, height, start_frame):
    reader = PNGReader(source_dir, width, height, start_num=start_frame)
    rgbs = []
    for _ in range(1 + g_frame_delay):
        rgb = reader.read_one_frame()
        if rgb is None:
            raise ValueError("not enough source frames for one I frame and one 8-frame chunk")
        rgbs.append(rgb)
    reader.close()
    tensors = []
    for rgb in rgbs:
        x = torch.from_numpy(rgb.copy()).unsqueeze(0).float() / 255.0
        tensors.append(rgb2ycbcr(x))
    return rgbs, tensors


def rgb_from_recon(x_hat):
    rgb = ycbcr2rgb(x_hat + 0.5)
    return torch.clamp(rgb * 255, 0, 255).round().byte().squeeze(0).cpu().numpy()


def psnr(original, reconstructed):
    mse = np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10.0 * math.log10(255.0 * 255.0 / mse)


def masked_psnr(original, reconstructed, mask):
    if not np.any(mask):
        return None
    diff = original.astype(np.float64) - reconstructed.astype(np.float64)
    mse = np.mean(diff[:, mask] ** 2)
    return float("inf") if mse == 0 else 10.0 * math.log10(255.0 * 255.0 / mse)


def save_recon_frames(out_dir, name, frames):
    target = out_dir / name
    target.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames, start=1):
        Image.fromarray(frame.transpose(1, 2, 0)).save(target / f"im{i:05d}.png")


def select_skip_blocks(candidate, expected_bits, max_blocks):
    candidate = np.asarray(candidate, dtype=np.bool_)
    if max_blocks <= 0 or int(candidate.sum()) <= max_blocks:
        return candidate.copy()
    flat_candidates = np.flatnonzero(candidate.reshape(-1))
    order = np.argsort(expected_bits.reshape(-1)[flat_candidates])[::-1]
    selected = np.zeros(candidate.size, dtype=np.bool_)
    selected[flat_candidates[order[:max_blocks]]] = True
    return selected.reshape(candidate.shape)


@torch.inference_mode()
def main():
    args = parse_args()
    if args.latent_block_size != 2:
        raise ValueError("Stage-1 v1 is deliberately restricted to 2x2 latent blocks")
    if args.height % 64 or args.width % 64:
        raise ValueError("Stage-1 v1 currently requires width and height divisible by 64")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    set_torch_env()
    torch.cuda.set_device(args.cuda_idx)
    device = torch.device(f"cuda:{args.cuda_idx}")
    torch.cuda.set_stream(torch.cuda.Stream(device=device))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    i_net = DMCI().eval()
    i_net.load_state_dict(get_state_dict(args.model_path_i))
    i_net.update(max(0.0, args.skip_thres))
    i_net = i_net.half().to(device).to(memory_format=torch.channels_last)

    p_net = DMC(ModelStructure.HTS).eval()
    p_net.load_state_dict(get_state_dict(args.model_path_p))
    p_net.update(max(0.0, args.skip_thres))
    p_net = p_net.half().to(device).to(memory_format=torch.channels_last)

    original_rgb, input_frames = read_frames(
        args.source_dir, args.width, args.height, args.start_frame)
    i_x = (input_frames[0].to(device).half() - 0.5).to(memory_format=torch.channels_last)
    chunk = torch.cat(input_frames[1:], dim=1)
    chunk = (chunk.to(device).half() - 0.5).to(memory_format=torch.channels_last)

    torch.cuda.synchronize(device)
    start = time.perf_counter()
    i_encoded = i_net.compress(i_x, args.qp_i, 0, 0)
    i_hat = i_encoded["x_hat"]
    torch.cuda.synchronize(device)
    i_seconds = time.perf_counter() - start

    p_net.ref_feature = F.pixel_unshuffle(i_hat, 8)
    p_net.apply_feature_adaptor()
    q_encoder = p_net.q_encoder[args.qp_p:args.qp_p + 1, :, None, None]
    q_feature = p_net.q_feature[args.qp_p:args.qp_p + 1, :, None, None]
    q_decoder = p_net.q_decoder[args.qp_p:args.qp_p + 1, :, None, None]
    y = p_net.encoder(chunk, p_net.ctx, q_encoder)
    z = p_net.hyper_encoder(y)
    z_hat = torch.round(z).clamp_(-128, 127)

    global_stream, z_ec_parallel = encode_z(
        z_hat, args.qp_p, p_net.bit_estimator_z.get_cdf_info())
    decoded_z_cpu = decode_z(global_stream, args.qp_p, z_hat.shape,
                             p_net.bit_estimator_z.get_cdf_info(), z_ec_parallel)
    decoded_z = decoded_z_cpu.to(device=device, dtype=z_hat.dtype,
                                 memory_format=torch.channels_last)
    if not torch.equal(decoded_z, z_hat):
        raise RuntimeError("global z rANS round-trip mismatch")
    common_params = p_net.res_prior_param_decoder(decoded_z, p_net.memory, q_feature)

    candidate = np.load(args.route_mask).astype(np.bool_)
    expected_grid = (y.shape[-2] // args.latent_block_size,
                     y.shape[-1] // args.latent_block_size)
    if candidate.shape != expected_grid:
        raise ValueError(f"route mask shape {candidate.shape}, expected {expected_grid}")

    all_base_mask = np.zeros_like(candidate)
    baseline_y = encode_y(p_net, y, common_params, all_base_mask, args.latent_block_size)
    selected = select_skip_blocks(candidate, baseline_y.expected_bits_by_block,
                                  args.max_skip_blocks)
    routed_y = encode_y(p_net, y, common_params, selected, args.latent_block_size)
    route_section = build_route_section(selected, args.latent_block_size)

    baseline_path = out_dir / "baseline_all_base.d1s"
    routed_path = out_dir / "routed_token_skip.d1s"
    baseline_header_bytes, baseline_total = write_container(
        baseline_path, args.qp_p, args.height, args.width, y.shape, global_stream, b"",
        baseline_y.stream, b"", baseline_y.ec_parallel, all_base=True,
        z_ec_parallel=z_ec_parallel)
    routed_header_bytes, routed_total = write_container(
        routed_path, args.qp_p, args.height, args.width, y.shape, global_stream, route_section,
        routed_y.stream, b"", routed_y.ec_parallel, all_base=False,
        z_ec_parallel=z_ec_parallel)

    baseline_file = read_container(baseline_path, args.latent_block_size)
    routed_file = read_container(routed_path, args.latent_block_size)
    decoded_baseline_z = decode_z(
        baseline_file["global"], baseline_file["qp"], z_hat.shape,
        p_net.bit_estimator_z.get_cdf_info(), baseline_file["z_ec"])
    decoded_routed_z = decode_z(
        routed_file["global"], routed_file["qp"], z_hat.shape,
        p_net.bit_estimator_z.get_cdf_info(), routed_file["z_ec"])
    if not torch.equal(decoded_baseline_z, z_hat.cpu()):
        raise RuntimeError("baseline global z container round-trip mismatch")
    if not torch.equal(decoded_routed_z, z_hat.cpu()):
        raise RuntimeError("routed global z container round-trip mismatch")
    decoded_baseline_z = decoded_baseline_z.to(
        device=device, dtype=z_hat.dtype, memory_format=torch.channels_last)
    decoded_routed_z = decoded_routed_z.to(
        device=device, dtype=z_hat.dtype, memory_format=torch.channels_last)
    baseline_common_params = p_net.res_prior_param_decoder(
        decoded_baseline_z, p_net.memory, q_feature)
    routed_common_params = p_net.res_prior_param_decoder(
        decoded_routed_z, p_net.memory, q_feature)
    decoded_baseline_q, decoded_baseline_y = decode_y(
        p_net, baseline_file["base"], baseline_file["y_ec"], baseline_common_params,
        baseline_file["skip_blocks"], args.latent_block_size)
    decoded_routed_q, decoded_routed_y = decode_y(
        p_net, routed_file["base"], routed_file["y_ec"], routed_common_params,
        routed_file["skip_blocks"], args.latent_block_size)
    if not torch.equal(decoded_baseline_q, baseline_y.q_dense):
        raise RuntimeError("baseline y rANS round-trip mismatch")
    if not torch.equal(decoded_routed_q, routed_y.q_dense):
        raise RuntimeError("routed y rANS round-trip mismatch")

    baseline_hat, _ = p_net.get_recon_and_feature(
        decoded_baseline_y, p_net.ctx, q_decoder)
    routed_hat, _ = p_net.get_recon_and_feature(decoded_routed_y, p_net.ctx, q_decoder)
    baseline_rgb = [rgb_from_recon(frame[:, :, :args.height, :args.width])
                    for frame in baseline_hat]
    routed_rgb = [rgb_from_recon(frame[:, :, :args.height, :args.width])
                  for frame in routed_hat]
    save_recon_frames(out_dir, "baseline", baseline_rgb)
    save_recon_frames(out_dir, "routed", routed_rgb)

    projection_stride = 16 * args.latent_block_size
    projection = np.repeat(np.repeat(selected, projection_stride, axis=0),
                           projection_stride, axis=1)
    Image.fromarray((projection.astype(np.uint8) * 255)).save(out_dir / "route_projection.png")

    per_frame = []
    for index, (org, base, routed) in enumerate(
            zip(original_rgb[1:], baseline_rgb, routed_rgb), start=args.start_frame + 1):
        per_frame.append({
            "frame": index,
            "baseline_psnr": psnr(org, base),
            "routed_psnr": psnr(org, routed),
            "baseline_skip_projection_psnr": masked_psnr(org, base, projection),
            "routed_skip_projection_psnr": masked_psnr(org, routed, projection),
            "baseline_keep_projection_psnr": masked_psnr(org, base, ~projection),
            "routed_keep_projection_psnr": masked_psnr(org, routed, ~projection),
        })
    with (out_dir / "per_frame.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_frame[0].keys())
        writer.writeheader()
        writer.writerows(per_frame)

    with (out_dir / "block_rate_map.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "route_row", "route_col", "candidate", "skipped", "estimated_baseline_bits"])
        writer.writeheader()
        for row in range(candidate.shape[0]):
            for col in range(candidate.shape[1]):
                writer.writerow({
                    "route_row": row,
                    "route_col": col,
                    "candidate": int(candidate[row, col]),
                    "skipped": int(selected[row, col]),
                    "estimated_baseline_bits": float(
                        baseline_y.expected_bits_by_block[row, col]),
                })

    influence_probe = None
    if np.any(selected):
        selected_flat = np.flatnonzero(selected.reshape(-1))
        probe_flat = selected_flat[np.argmax(
            baseline_y.expected_bits_by_block.reshape(-1)[selected_flat])]
        probe_row, probe_col = np.unravel_index(probe_flat, selected.shape)
        probe_skip = np.zeros_like(selected)
        probe_skip[probe_row, probe_col] = True
        probe_keep = expand_keep_mask(probe_skip, args.latent_block_size, y.shape, y.device)
        _, _, probe_y_hat, _ = y_prior_steps(
            p_net, common_params, probe_keep, source_y=y)
        probe_hat, _ = p_net.get_recon_and_feature(probe_y_hat, p_net.ctx, q_decoder)
        probe_rgb = [rgb_from_recon(frame[:, :, :args.height, :args.width])
                     for frame in probe_hat]
        heatmaps = []
        frame_support = []
        y0 = probe_row * projection_stride
        y1 = y0 + projection_stride
        x0 = probe_col * projection_stride
        x1 = x0 + projection_stride
        for frame_index, (base, probe) in enumerate(zip(baseline_rgb, probe_rgb)):
            heat = np.mean(np.abs(base.astype(np.float32) - probe.astype(np.float32)), axis=0)
            heatmaps.append(heat)
            support = heat >= 0.5
            if np.any(support):
                ys, xs = np.nonzero(support)
                bbox = [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]
            else:
                bbox = None
            total_energy = float(heat.sum())
            nominal_energy = float(heat[y0:y1, x0:x1].sum())
            frame_support.append({
                "frame_offset": frame_index,
                "bbox_xyxy_at_0_5_level": bbox,
                "changed_pixel_ratio": float(support.mean()),
                "nominal_projection_energy_fraction": (
                    nominal_energy / total_energy if total_energy > 0 else None),
            })
        aggregate_heat = np.mean(np.stack(heatmaps), axis=0)
        scale = float(np.percentile(aggregate_heat, 99.5))
        heat_u8 = np.clip(aggregate_heat / max(scale, 1e-6) * 255, 0, 255).astype(np.uint8)
        Image.fromarray(heat_u8).save(out_dir / "influence_probe.png")
        influence_probe = {
            "route_block_row_col": [int(probe_row), int(probe_col)],
            "selection": "highest estimated-rate skipped block",
            "nominal_projection_xyxy": [int(x0), int(y0), int(x1), int(y1)],
            "threshold_rgb_level": 0.5,
            "frames": frame_support,
            "heatmap": "influence_probe.png",
        }

    stock_payload_bytes = None
    if args.compare_stock:
        p_net.clear_dpb()
        p_net.add_ref_feature_from_frame(i_hat, apply_feature_adaptor=True)
        stock = p_net.compress(chunk, args.qp_p, False, 0, 0)
        stock_payload_bytes = len(stock["bit_stream"])

    pixels = args.width * args.height * g_frame_delay
    summary = {
        "experiment": "stage1_hts_2x2x256_token_skipping_v1",
        "scope": "one 8-frame P chunk; I-frame is a shared conditioning cost",
        "routing_unit": {
            "latent_shape": list(y.shape),
            "block_shape": [args.latent_block_size, args.latent_block_size, g_ch_y],
            "temporal_extent_frames": g_frame_delay,
            "route_grid": list(candidate.shape),
            "nominal_pixel_projection": [projection_stride, projection_stride],
            "current_chunk_encoder_receptive_field": [136, 136],
            "direct_synthesis_support_before_entropy_mean_propagation": [208, 208],
            "note": "Spatial-prior mean propagation can make empirical influence wider.",
        },
        "route": {
            "candidate_blocks": int(candidate.sum()),
            "skipped_blocks": int(selected.sum()),
            "total_blocks": int(selected.size),
            "skipped_ratio": float(selected.mean()),
        },
        "i_frame_condition": {
            "payload_bytes": len(i_encoded["bit_stream"]),
            "encode_seconds": i_seconds,
            "included_in_stage1_totals": False,
        },
        "baseline": {
            "container_header_bytes": baseline_header_bytes,
            "global_z_bytes": len(global_stream),
            "route_bytes": 0,
            "base_y_bytes": len(baseline_y.stream),
            "residual_bytes": 0,
            "total_bytes": baseline_total,
            "total_bpp": baseline_total * 8 / pixels,
            "y_symbol_count": baseline_y.symbol_count,
            "y_ec_parallel": baseline_y.ec_parallel,
            "mean_psnr": float(np.mean([row["baseline_psnr"] for row in per_frame])),
            "mean_skip_projection_psnr": float(np.mean([
                row["baseline_skip_projection_psnr"] for row in per_frame])),
            "mean_keep_projection_psnr": float(np.mean([
                row["baseline_keep_projection_psnr"] for row in per_frame])),
        },
        "routed": {
            "container_header_bytes": routed_header_bytes,
            "global_z_bytes": len(global_stream),
            "route_bytes": len(route_section),
            "base_y_bytes": len(routed_y.stream),
            "residual_bytes": 0,
            "total_bytes": routed_total,
            "total_bpp": routed_total * 8 / pixels,
            "y_symbol_count": routed_y.symbol_count,
            "y_ec_parallel": routed_y.ec_parallel,
            "mean_psnr": float(np.mean([row["routed_psnr"] for row in per_frame])),
            "mean_skip_projection_psnr": float(np.mean([
                row["routed_skip_projection_psnr"] for row in per_frame])),
            "mean_keep_projection_psnr": float(np.mean([
                row["routed_keep_projection_psnr"] for row in per_frame])),
        },
        "delta": {
            "base_y_bytes_saved": len(baseline_y.stream) - len(routed_y.stream),
            "net_bytes_saved": baseline_total - routed_total,
            "net_rate_change_percent": 100.0 * (routed_total / baseline_total - 1.0),
            "mean_psnr_change_db": float(np.mean([
                row["routed_psnr"] - row["baseline_psnr"] for row in per_frame])),
        },
        "validation": {
            "global_rans_roundtrip": True,
            "global_rans_redecoded_from_container": True,
            "baseline_y_rans_roundtrip": True,
            "routed_y_rans_roundtrip": True,
            "containers_reloaded_before_reconstruction": True,
            "decoder_uses_source_y": False,
            "stock_fused_p_payload_bytes": stock_payload_bytes,
        },
        "influence_probe": influence_probe,
        "args": vars(args),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
