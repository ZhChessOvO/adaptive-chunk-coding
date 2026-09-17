#!/usr/bin/env python3
"""Minimal legal one-shot spatial-quality codec for DCVC-UF HT-S.

The released DCVC-UF implementation selects one quality row per coding unit.
This research prototype selects the released rows spatially, while retaining a
single analysis transform, a single latent field, and a single reference-loop
reconstruction.  It therefore does not splice independently encoded latents.

The outer bitstream uses NAL_I_SQ/NAL_P_SQ from ``stream_helper``.  Each coding
unit contains seven independently decodable rANS substreams: three z streams
(one CDF family per Generate/Base/Enhance action) and four causal y-prior
streams.  Splitting z by CDF family is an entropy syntax choice; every symbol
is scattered back into one shared z tensor before synthesis.

This file intentionally uses the PyTorch model path.  It establishes format
legality and fresh-decode correctness before optimized CUDA proxy work or
codec fine-tuning.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.stage_c_evaluate_seedvr2_gate import load_source
from demo.stage_c_three_path_roi_probe import rgb_from_tensor, tensor_from_rgb
from src.models.image_model import DMCI, g_ch_z as g_ch_z_i
from src.models.video_model_ht import DMC, g_ch_z as g_ch_z_p, g_frame_delay
from src.utils.common import ModelStructure, get_state_dict, set_torch_env
from src.utils.stream_helper import (
    NalType,
    read_header,
    read_spatial_ip_remaining,
    read_sps_remaining,
    validate_spatial_actions,
    write_spatial_ip,
    write_sps,
)


ACTION_BASE = 0
ACTION_GENERATE = 1
ACTION_ENHANCE = 2
ACTION_NAMES = {
    ACTION_BASE: "Base",
    ACTION_GENERATE: "Generate",
    ACTION_ENHANCE: "Enhance",
}
QUALITY_ORDER = (ACTION_GENERATE, ACTION_BASE, ACTION_ENHANCE)

INNER_MAGIC = b"SQE1"
INNER_VERSION = 2
# v1: magic, version, three z EC counts, four y EC counts, seven byte lengths.
# v2 adds a normative scale-interpolation id so fresh decoding has no hidden
# command-line dependency.  The v1 reader remains for already registered
# diagnostics, where the caller must provide the expected interpolation.
INNER_HEADER_V1 = struct.Struct("<4s8B7I")
INNER_HEADER = struct.Struct("<4s9B7I")
SCALE_MODE_TO_ID = {"nearest": 0, "bilinear": 1}
SCALE_ID_TO_MODE = {value: key for key, value in SCALE_MODE_TO_ID.items()}
SUBSTREAM_NAMES = (
    "z-generate", "z-base", "z-enhance",
    "y-prior-0", "y-prior-1", "y-prior-2", "y-prior-3",
)
MIN_SYMBOLS_PER_STREAM = 32768
MAX_EC_PARALLEL = 8


def common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cuda-idx", type=int, default=0)
    parser.add_argument(
        "--model-path-i", type=Path,
        default=Path("checkpoints/cvpr2026_image.pth.tar"))
    parser.add_argument(
        "--model-path-p", type=Path,
        default=Path("checkpoints/cvpr2026_video_hts.pth.tar"))
    parser.add_argument("--skip-thres", type=float, default=0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-shot spatial-QP DCVC-UF research codec")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    encode = subparsers.add_parser("encode")
    common_parser(encode)
    encode.add_argument("--gate-summary", type=Path, required=True)
    encode.add_argument("--route-summary", type=Path, required=True)
    encode.add_argument("--output-stream", type=Path, required=True)
    encode.add_argument("--output-dir", type=Path, required=True)
    encode.add_argument("--frame-count", type=int, default=17)
    encode.add_argument("--cell-size", type=int, default=64)
    encode.add_argument("--generate-qp", type=int, default=8)
    encode.add_argument("--base-qp", type=int, default=16)
    encode.add_argument("--enhance-qp", type=int, default=32)
    encode.add_argument(
        "--scale-interpolation", choices=("nearest", "bilinear"),
        default="nearest")

    decode = subparsers.add_parser("decode")
    common_parser(decode)
    decode.add_argument("--input-stream", type=Path, required=True)
    decode.add_argument("--output-dir", type=Path, required=True)
    decode.add_argument(
        "--scale-interpolation", choices=("nearest", "bilinear"),
        help="Optional assertion for legacy/testing; v2 reads this mode from the stream")

    args = parser.parse_args()
    if args.skip_thres < 0:
        parser.error("--skip-thres must be nonnegative")
    if args.mode == "encode":
        if args.frame_count < 1 or (args.frame_count - 1) % g_frame_delay:
            parser.error("frame count must be 1 + an integer number of 8-frame P chunks")
        if not 0 <= args.generate_qp < args.base_qp < args.enhance_qp < 64:
            parser.error("require Generate QP < Base QP < Enhance QP < 64")
    return args


def compute_ec_parallel(symbol_count: int) -> int:
    return max(1, min(MAX_EC_PARALLEL, symbol_count // MIN_SYMBOLS_PER_STREAM))


def tensor_nhwc_numpy(x: torch.Tensor, dtype=None) -> np.ndarray:
    output = x.detach().permute(0, 2, 3, 1).contiguous().cpu().numpy()
    return output.astype(dtype, copy=False) if dtype is not None else output


def scale_to_index(scales: np.ndarray) -> np.ndarray:
    scale_min = 0.11
    scale_max = 16.0
    scale_levels = 128
    values = np.clip(scales.astype(np.float32), scale_min, scale_max)
    step = (math.log(scale_max) - math.log(scale_min)) / (scale_levels - 1)
    indexes = (np.log(values) - math.log(scale_min)) / step
    return np.clip(indexes, 0, scale_levels - 1).astype(np.uint8)


def make_rans_encoder(cdf_info, slot: int, ec_parallel: int):
    from MLCodec_extensions_cpp import RansEncoder

    coder = RansEncoder()
    cdf, cdf_length = cdf_info
    coder.set_cdf(
        np.asarray(cdf, dtype=np.int32),
        np.asarray(cdf_length, dtype=np.int32),
        slot,
    )
    coder.set_entropy_coder_parallel(ec_parallel)
    coder.reset()
    return coder


def make_rans_decoder(cdf_info, slot: int, ec_parallel: int, stream: bytes):
    from MLCodec_extensions_cpp import RansDecoder

    coder = RansDecoder()
    cdf, cdf_length = cdf_info
    coder.set_cdf(
        np.asarray(cdf, dtype=np.int32),
        np.asarray(cdf_length, dtype=np.int32),
        slot,
    )
    coder.set_entropy_coder_parallel(ec_parallel)
    coder.set_stream(np.frombuffer(stream, dtype=np.uint8).copy())
    return coder


def encode_z_stream(symbols: np.ndarray, qp: int, cdf_info, channels: int):
    symbols = np.asarray(symbols, dtype=np.int8).reshape(-1)
    if symbols.size == 0:
        return b"", 1
    if symbols.size % channels:
        raise ValueError("z substream does not contain complete channel vectors")
    ec_parallel = compute_ec_parallel(int(symbols.size))
    coder = make_rans_encoder(cdf_info, 0, ec_parallel)
    coder.encode_z(symbols, qp * channels, channels)
    coder.flush()
    stream = np.asarray(coder.get_encoded_stream(), dtype=np.uint8).tobytes()
    return stream, ec_parallel


def decode_z_stream(
    stream: bytes,
    symbol_count: int,
    qp: int,
    cdf_info,
    channels: int,
    ec_parallel: int,
) -> np.ndarray:
    if symbol_count == 0:
        if stream:
            raise ValueError("nonempty z stream for an unused action")
        return np.empty((0,), dtype=np.int8)
    if not stream:
        raise ValueError("empty z stream for a used action")
    coder = make_rans_decoder(cdf_info, 0, ec_parallel, stream)
    coder.decode_z(symbol_count, qp * channels, channels)
    decoded = np.asarray(coder.get_decoded_tensor(), dtype=np.int8).copy()
    if decoded.size != symbol_count:
        raise RuntimeError("decoded z symbol count mismatch")
    return decoded


def encode_y_stream(
    q_part: torch.Tensor,
    scales: torch.Tensor,
    active: torch.Tensor,
    cdf_info,
):
    active_values = tensor_nhwc_numpy(active, np.bool_).reshape(-1)
    q_values = tensor_nhwc_numpy(q_part, np.int8).reshape(-1)[active_values]
    scale_values = tensor_nhwc_numpy(scales, np.float32).reshape(-1)[active_values]
    indexes = scale_to_index(scale_values).astype(np.int16)
    symbols = q_values.astype(np.int16) * 256 + indexes
    symbols = symbols.astype(np.int16, copy=False)
    if symbols.size == 0:
        return b"", 1, 0
    ec_parallel = compute_ec_parallel(int(symbols.size))
    coder = make_rans_encoder(cdf_info, 1, ec_parallel)
    coder.encode_y(symbols)
    coder.flush()
    stream = np.asarray(coder.get_encoded_stream(), dtype=np.uint8).tobytes()
    return stream, ec_parallel, int(symbols.size)


def decode_y_stream(
    stream: bytes,
    scales: torch.Tensor,
    active: torch.Tensor,
    cdf_info,
    ec_parallel: int,
) -> torch.Tensor:
    active_values = tensor_nhwc_numpy(active, np.bool_).reshape(-1)
    scale_values = tensor_nhwc_numpy(scales, np.float32).reshape(-1)[active_values]
    indexes = scale_to_index(scale_values)
    dense = np.zeros(active_values.size, dtype=np.int8)
    if indexes.size:
        if not stream:
            raise ValueError("empty y stream with active symbols")
        coder = make_rans_decoder(cdf_info, 1, ec_parallel, stream)
        coder.decode_y(indexes.astype(np.uint8, copy=False))
        decoded = np.asarray(coder.get_decoded_tensor(), dtype=np.int8).copy()
        if decoded.size != indexes.size:
            raise RuntimeError("decoded y symbol count mismatch")
        dense[active_values] = decoded
    elif stream:
        raise ValueError("nonempty y stream without active symbols")
    batch, channels, height, width = active.shape
    dense = dense.reshape(batch, height, width, channels).transpose(0, 3, 1, 2)
    return torch.from_numpy(dense.copy()).to(
        device=scales.device, dtype=scales.dtype,
        memory_format=torch.channels_last,
    )


def pack_entropy_payload(
    streams: list[bytes], ec_parallel: list[int], interpolation: str
) -> bytes:
    if len(streams) != 7 or len(ec_parallel) != 7:
        raise ValueError("spatial entropy payload requires seven substreams")
    if any(not 1 <= value <= MAX_EC_PARALLEL for value in ec_parallel):
        raise ValueError("invalid entropy parallelism")
    if interpolation not in SCALE_MODE_TO_ID:
        raise ValueError("unsupported scale interpolation")
    header = INNER_HEADER.pack(
        INNER_MAGIC,
        INNER_VERSION,
        SCALE_MODE_TO_ID[interpolation],
        *ec_parallel,
        *(len(stream) for stream in streams),
    )
    return header + b"".join(streams)


def unpack_entropy_payload(
    payload: bytes, legacy_interpolation: str | None = None
) -> tuple[list[bytes], list[int], str]:
    if len(payload) < 5:
        raise ValueError("truncated spatial entropy payload")
    magic, version = struct.unpack("<4sB", payload[:5])
    if magic != INNER_MAGIC or version not in (1, INNER_VERSION):
        raise ValueError("unsupported spatial entropy payload")
    if version == 1:
        if legacy_interpolation not in SCALE_MODE_TO_ID:
            raise ValueError("v1 payload requires an explicit legacy interpolation")
        if len(payload) < INNER_HEADER_V1.size:
            raise ValueError("truncated version-1 spatial entropy payload")
        fields = INNER_HEADER_V1.unpack(payload[:INNER_HEADER_V1.size])
        header_size = INNER_HEADER_V1.size
        interpolation = legacy_interpolation
        ec_parallel = list(fields[2:9])
        lengths = list(fields[9:16])
    else:
        if len(payload) < INNER_HEADER.size:
            raise ValueError("truncated version-2 spatial entropy payload")
        fields = INNER_HEADER.unpack(payload[:INNER_HEADER.size])
        header_size = INNER_HEADER.size
        if fields[2] not in SCALE_ID_TO_MODE:
            raise ValueError("invalid stored scale interpolation")
        interpolation = SCALE_ID_TO_MODE[fields[2]]
        ec_parallel = list(fields[3:10])
        lengths = list(fields[10:17])
    if any(not 1 <= value <= MAX_EC_PARALLEL for value in ec_parallel):
        raise ValueError("invalid stored entropy parallelism")
    position = header_size
    streams = []
    for length in lengths:
        end = position + length
        if end > len(payload):
            raise ValueError("truncated spatial entropy substream")
        streams.append(payload[position:end])
        position = end
    if position != len(payload):
        raise ValueError("trailing bytes in spatial entropy payload")
    return streams, ec_parallel, interpolation


def select_scale(
    table: torch.Tensor,
    qp_map: torch.Tensor,
    size: tuple[int, int],
    interpolation: str,
) -> torch.Tensor:
    batch, grid_height, grid_width = qp_map.shape
    values = table[qp_map.reshape(-1)]
    values = values.reshape(batch, grid_height, grid_width, table.shape[1])
    values = values.permute(0, 3, 1, 2).contiguous()
    if values.shape[-2:] == size:
        return values
    if interpolation == "nearest":
        return F.interpolate(values, size=size, mode="nearest")
    return F.interpolate(values, size=size, mode="bilinear", align_corners=False)


def quality_maps(
    actions: np.ndarray,
    quality_profile: tuple[int, int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if actions.ndim != 2:
        raise ValueError("action map must be two-dimensional")
    action_map = torch.from_numpy(actions.astype(np.int64, copy=True))[None].to(device)
    generate_qp, base_qp, enhance_qp = quality_profile
    lookup = torch.tensor(
        [base_qp, generate_qp, enhance_qp], device=device, dtype=torch.long)
    return action_map, lookup[action_map]


def z_action_map(action_map: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    resized = F.interpolate(
        action_map[:, None].float(), size=size, mode="nearest")
    return resized[0, 0].long().cpu().numpy()


def encode_z_groups(
    model,
    z_hat: torch.Tensor,
    action_map: torch.Tensor,
    quality_profile: tuple[int, int, int],
) -> tuple[list[bytes], list[int], list[int]]:
    channels = int(z_hat.shape[1])
    if channels not in (g_ch_z_i, g_ch_z_p):
        raise ValueError("unexpected z channel count")
    actions = z_action_map(action_map, z_hat.shape[-2:])
    values = tensor_nhwc_numpy(z_hat, np.int8)[0]
    qp_by_action = {
        ACTION_GENERATE: quality_profile[0],
        ACTION_BASE: quality_profile[1],
        ACTION_ENHANCE: quality_profile[2],
    }
    streams = []
    ec_values = []
    counts = []
    for action in QUALITY_ORDER:
        symbols = values[actions == action].reshape(-1)
        stream, ec_parallel = encode_z_stream(
            symbols, qp_by_action[action], model.bit_estimator_z.get_cdf_info(), channels)
        streams.append(stream)
        ec_values.append(ec_parallel)
        counts.append(int(symbols.size))
    return streams, ec_values, counts


def decode_z_groups(
    model,
    streams: list[bytes],
    ec_values: list[int],
    action_map: torch.Tensor,
    quality_profile: tuple[int, int, int],
    shape: tuple[int, int, int, int],
    dtype: torch.dtype,
) -> torch.Tensor:
    batch, channels, height, width = shape
    if batch != 1:
        raise ValueError("prototype supports batch size one")
    actions = z_action_map(action_map, (height, width))
    values = np.zeros((height, width, channels), dtype=np.int8)
    qp_by_action = {
        ACTION_GENERATE: quality_profile[0],
        ACTION_BASE: quality_profile[1],
        ACTION_ENHANCE: quality_profile[2],
    }
    for stream, ec_parallel, action in zip(streams, ec_values, QUALITY_ORDER):
        position_count = int(np.count_nonzero(actions == action))
        symbol_count = position_count * channels
        decoded = decode_z_stream(
            stream, symbol_count, qp_by_action[action],
            model.bit_estimator_z.get_cdf_info(), channels, ec_parallel)
        if symbol_count:
            values[actions == action] = decoded.reshape(position_count, channels)
    values = values[None].transpose(0, 3, 1, 2)
    return torch.from_numpy(values.copy()).to(
        device=action_map.device, dtype=dtype,
        memory_format=torch.channels_last,
    )


def prior_state(model, common_params: torch.Tensor, image: bool):
    if image:
        scales, means = model.separate_prior_image(common_params)
        q_enc = q_dec = None
    else:
        q_enc, q_dec, scales, means = model.separate_prior_video(common_params)
    reduced = model.y_spatial_prior_reduction(common_params)
    return q_enc, q_dec, scales, means, reduced


def update_prior(model, part: int, y_hat_so_far: torch.Tensor,
                 reduced: torch.Tensor, image: bool):
    adaptor = (
        model.y_spatial_prior_adaptor_1,
        model.y_spatial_prior_adaptor_2,
        model.y_spatial_prior_adaptor_3,
    )[part - 1]
    if image:
        params = torch.cat((y_hat_so_far, reduced), dim=1)
        return model.y_spatial_prior(adaptor(params)).chunk(2, 1)
    means = model.y_spatial_prior(adaptor(y_hat_so_far, reduced))
    return None, means


def encode_y_groups(
    model,
    y: torch.Tensor,
    common_params: torch.Tensor,
    image: bool,
    external_q_enc: torch.Tensor | None = None,
    external_q_dec: torch.Tensor | None = None,
):
    q_enc, q_dec, scales, means, reduced = prior_state(model, common_params, image)
    if image:
        if external_q_enc is None or external_q_dec is None:
            raise ValueError("image path requires spatial y quantization scales")
        q_enc, q_dec = external_q_enc, external_q_dec
    y_scaled = y * q_enc
    masks = model.get_mask_4x(*y.shape, y.device)
    y_hat_so_far = torch.zeros_like(y)
    streams = []
    ec_values = []
    symbol_counts = []
    for part, mask in enumerate(masks):
        if part:
            new_scales, means = update_prior(
                model, part, y_hat_so_far, reduced, image)
            if new_scales is not None:
                scales = new_scales
        residual = (y_scaled - means * mask) * mask
        q_part = torch.round(residual).clamp_(-128, 127)
        active = mask & (scales > model.gaussian_encoder.skip_thres)
        q_part = q_part * active
        stream, ec_parallel, count = encode_y_stream(
            q_part, scales, active, model.gaussian_encoder.get_cdf_info())
        streams.append(stream)
        ec_values.append(ec_parallel)
        symbol_counts.append(count)
        y_hat_so_far = y_hat_so_far + (q_part + means * mask) * mask
    return y_hat_so_far * q_dec, streams, ec_values, symbol_counts


def decode_y_groups(
    model,
    streams: list[bytes],
    ec_values: list[int],
    common_params: torch.Tensor,
    image: bool,
    external_q_dec: torch.Tensor | None = None,
) -> torch.Tensor:
    _q_enc, q_dec, scales, means, reduced = prior_state(model, common_params, image)
    if image:
        if external_q_dec is None:
            raise ValueError("image path requires spatial y dequantization scales")
        q_dec = external_q_dec
    shape = (common_params.shape[0], 256, common_params.shape[2], common_params.shape[3])
    reference = torch.empty(shape, device=common_params.device, dtype=common_params.dtype)
    masks = model.get_mask_4x(*reference.shape, reference.device)
    y_hat_so_far = torch.zeros_like(reference)
    for part, (mask, stream, ec_parallel) in enumerate(
            zip(masks, streams, ec_values)):
        if part:
            new_scales, means = update_prior(
                model, part, y_hat_so_far, reduced, image)
            if new_scales is not None:
                scales = new_scales
        active = mask & (scales > model.gaussian_encoder.skip_thres)
        q_part = decode_y_stream(
            stream, scales, active,
            model.gaussian_encoder.get_cdf_info(), ec_parallel)
        y_hat_so_far = y_hat_so_far + (q_part + means * mask) * mask
    return y_hat_so_far * q_dec


def round_z(z: torch.Tensor) -> torch.Tensor:
    return torch.round(z).clamp_(-128, 127)


def encode_i(
    model: DMCI,
    x: torch.Tensor,
    action_map: torch.Tensor,
    qp_map: torch.Tensor,
    quality_profile: tuple[int, int, int],
    interpolation: str,
):
    feature_size = (x.shape[-2] // 8, x.shape[-1] // 8)
    q_enc = select_scale(model.q_scale_enc, qp_map, feature_size, interpolation)
    y = model.enc(x, q_enc)
    q_y_enc = select_scale(model.q_scale_y_enc, qp_map, y.shape[-2:], interpolation)
    q_y_dec = select_scale(model.q_scale_y_dec, qp_map, y.shape[-2:], interpolation)
    z_hat = round_z(model.hyper_enc(y))
    z_streams, z_ec, z_counts = encode_z_groups(
        model, z_hat, action_map, quality_profile)
    params = model.y_prior_fusion(model.hyper_dec(z_hat))
    params = params[:, :, :y.shape[-2], :y.shape[-1]]
    y_hat, y_streams, y_ec, y_counts = encode_y_groups(
        model, y, params, True, q_y_enc, q_y_dec)
    q_dec = select_scale(model.q_scale_dec, qp_map, feature_size, interpolation)
    x_hat = model.dec(y_hat, q_dec)
    streams = [*z_streams, *y_streams]
    ec_values = [*z_ec, *y_ec]
    payload = pack_entropy_payload(streams, ec_values, interpolation)
    return payload, x_hat, {
        "substream_bytes": dict(zip(SUBSTREAM_NAMES, map(len, streams))),
        "substream_symbols": dict(zip(SUBSTREAM_NAMES, [*z_counts, *y_counts])),
        "inner_header_bytes": INNER_HEADER.size,
    }


def decode_i(
    model: DMCI,
    payload: bytes,
    action_map: torch.Tensor,
    qp_map: torch.Tensor,
    quality_profile: tuple[int, int, int],
    height: int,
    width: int,
    interpolation: str | None,
) -> torch.Tensor:
    streams, ec_values, stored_interpolation = unpack_entropy_payload(
        payload, interpolation)
    if interpolation is not None and stored_interpolation != interpolation:
        raise ValueError(
            f"decoder interpolation {interpolation} differs from stored "
            f"mode {stored_interpolation}")
    interpolation = stored_interpolation
    dtype = next(model.parameters()).dtype
    z_hat = decode_z_groups(
        model, streams[:3], ec_values[:3], action_map, quality_profile,
        (1, g_ch_z_i, height // 64, width // 64), dtype)
    y_height, y_width = height // 16, width // 16
    params = model.y_prior_fusion(model.hyper_dec(z_hat))
    params = params[:, :, :y_height, :y_width]
    q_y_dec = select_scale(
        model.q_scale_y_dec, qp_map, (y_height, y_width), interpolation)
    y_hat = decode_y_groups(
        model, streams[3:], ec_values[3:], params, True,
        external_q_dec=q_y_dec)
    q_dec = select_scale(
        model.q_scale_dec, qp_map, (height // 8, width // 8), interpolation)
    return model.dec(y_hat, q_dec)


def encode_p(
    model: DMC,
    x: torch.Tensor,
    action_map: torch.Tensor,
    qp_map: torch.Tensor,
    quality_profile: tuple[int, int, int],
    interpolation: str,
):
    model.apply_feature_adaptor()
    feature_size = model.ctx.shape[-2:]
    q_enc = select_scale(model.q_encoder, qp_map, feature_size, interpolation)
    y = model.encoder(x, model.ctx, q_enc)
    z_hat = round_z(model.hyper_encoder(y))
    z_streams, z_ec, z_counts = encode_z_groups(
        model, z_hat, action_map, quality_profile)
    q_feature = select_scale(
        model.q_feature, qp_map, model.memory.shape[-2:], interpolation)
    params = model.res_prior_param_decoder(z_hat, model.memory, q_feature)
    y_hat, y_streams, y_ec, y_counts = encode_y_groups(
        model, y, params, False)
    q_dec = select_scale(model.q_decoder, qp_map, feature_size, interpolation)
    x_hat, feature = model.get_recon_and_feature(y_hat, model.ctx, q_dec)
    model.set_ref_feature(feature, False)
    streams = [*z_streams, *y_streams]
    ec_values = [*z_ec, *y_ec]
    payload = pack_entropy_payload(streams, ec_values, interpolation)
    return payload, x_hat, {
        "substream_bytes": dict(zip(SUBSTREAM_NAMES, map(len, streams))),
        "substream_symbols": dict(zip(SUBSTREAM_NAMES, [*z_counts, *y_counts])),
        "inner_header_bytes": INNER_HEADER.size,
    }


def decode_p(
    model: DMC,
    payload: bytes,
    action_map: torch.Tensor,
    qp_map: torch.Tensor,
    quality_profile: tuple[int, int, int],
    height: int,
    width: int,
    interpolation: str | None,
) -> list[torch.Tensor]:
    streams, ec_values, stored_interpolation = unpack_entropy_payload(
        payload, interpolation)
    if interpolation is not None and stored_interpolation != interpolation:
        raise ValueError(
            f"decoder interpolation {interpolation} differs from stored "
            f"mode {stored_interpolation}")
    interpolation = stored_interpolation
    model.apply_feature_adaptor()
    dtype = next(model.parameters()).dtype
    z_hat = decode_z_groups(
        model, streams[:3], ec_values[:3], action_map, quality_profile,
        (1, g_ch_z_p, height // 64, width // 64), dtype)
    q_feature = select_scale(
        model.q_feature, qp_map, model.memory.shape[-2:], interpolation)
    params = model.res_prior_param_decoder(z_hat, model.memory, q_feature)
    y_hat = decode_y_groups(
        model, streams[3:], ec_values[3:], params, False)
    q_dec = select_scale(
        model.q_decoder, qp_map, model.ctx.shape[-2:], interpolation)
    x_hat, feature = model.get_recon_and_feature(y_hat, model.ctx, q_dec)
    model.set_ref_feature(feature, False)
    return x_hat


def load_codecs(args: argparse.Namespace, device: torch.device) -> tuple[DMCI, DMC, float]:
    started = time.perf_counter()
    i_net = DMCI().eval()
    i_net.load_state_dict(get_state_dict(str(args.model_path_i)))
    i_net.update(args.skip_thres)
    i_net = i_net.half().to(device).to(memory_format=torch.channels_last)
    p_net = DMC(ModelStructure.HTS).eval()
    p_net.load_state_dict(get_state_dict(str(args.model_path_p)))
    p_net.update(args.skip_thres)
    p_net = p_net.half().to(device).to(memory_format=torch.channels_last)
    return i_net, p_net, time.perf_counter() - started


def save_reconstruction(path: Path, frames: list[torch.Tensor], height: int, width: int):
    path.mkdir(parents=True, exist_ok=True)
    arrays = []
    for index, frame in enumerate(frames, start=1):
        rgb = rgb_from_tensor(frame, height, width)
        Image.fromarray(rgb).save(path / f"im{index:05d}.png")
        arrays.append(rgb)
    return arrays


def selected_route_variant(route: dict) -> tuple[str, dict]:
    name = route.get("selected_variant", "joint-three-path-oracle")
    variants = route.get("variants", {})
    if name not in variants:
        raise ValueError(f"selected route variant is absent: {name}")
    return name, variants[name]


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)


def load_actions(
    route: dict, height: int, width: int, cell_size: int,
) -> tuple[np.ndarray, str, dict]:
    config = route["configuration"]
    route_height, route_width = config["tile_grid"]
    route_tile_size = config["tile_size"]
    variant_name, variant = selected_route_variant(route)
    values = np.asarray(variant["actions"], dtype=np.int64)
    values = values.reshape(route_height, route_width)
    if route_tile_size % cell_size:
        raise ValueError("route tile size must be divisible by syntax cell size")
    repeat = route_tile_size // cell_size
    actions = np.repeat(np.repeat(values, repeat, axis=0), repeat, axis=1)
    validate_spatial_actions(width, height, cell_size, actions.reshape(-1))
    return actions, variant_name, variant


@torch.inference_mode()
def encode_main(args: argparse.Namespace, device: torch.device) -> None:
    process_started = time.perf_counter()
    gate = json.loads(args.gate_summary.read_text(encoding="utf-8"))
    route = json.loads(args.route_summary.read_text(encoding="utf-8"))
    reference = load_source(gate, args.frame_count)
    height, width = reference[0].shape[:2]
    if height % 64 or width % 64:
        raise ValueError("current spatial-quality prototype requires dimensions divisible by 64")
    actions, route_variant_name, route_variant = load_actions(
        route, height, width, args.cell_size)
    route_kind = route.get("route_kind", "encoder-side-oracle")
    profile = (args.generate_qp, args.base_qp, args.enhance_qp)
    action_map, qp_map = quality_maps(actions, profile, device)
    tensors = [tensor_from_rgb(frame, device) for frame in reference]

    torch.cuda.reset_peak_memory_stats(device)
    i_net, p_net, model_load_seconds = load_codecs(args, device)
    torch.cuda.synchronize(device)
    codec_started = time.perf_counter()
    payload_i, x_hat_i, stats_i = encode_i(
        i_net, tensors[0], action_map, qp_map, profile,
        args.scale_interpolation)
    reconstructions = [x_hat_i]
    units = [(True, payload_i, stats_i)]
    p_net.clear_dpb()
    p_net.ref_feature = F.pixel_unshuffle(x_hat_i, 8)
    for start in range(1, args.frame_count, g_frame_delay):
        p_input = torch.cat(tensors[start:start + g_frame_delay], dim=1)
        payload_p, x_hat_p, stats_p = encode_p(
            p_net, p_input, action_map, qp_map, profile,
            args.scale_interpolation)
        reconstructions.extend(x_hat_p)
        units.append((False, payload_p, stats_p))
    torch.cuda.synchronize(device)
    codec_seconds = time.perf_counter() - codec_started

    args.output_stream.parent.mkdir(parents=True, exist_ok=True)
    unit_bytes = []
    with args.output_stream.open("wb") as output:
        sps_bytes = write_sps(output, {"sps_id": 0, "height": height, "width": width})
        for unit_index, (is_i, payload, stats) in enumerate(units):
            written = write_spatial_ip(
                output, is_i, 0, profile, args.cell_size,
                actions.reshape(-1).tolist(), 0, 0, payload)
            unit_bytes.append({
                "index": unit_index,
                "type": "I" if is_i else "P8",
                "on_disk_bytes": written,
                "outer_syntax_bytes": written - len(payload),
                "entropy_payload_bytes": len(payload),
                **stats,
            })
    stream_bytes = args.output_stream.stat().st_size
    frames_dir = args.output_dir / "encoder_reconstruction"
    save_reconstruction(frames_dir, reconstructions, height, width)
    summary = {
        "mode": "encode",
        "format": "DCVC-UF one-shot spatial quality prototype v2",
        "stream": str(args.output_stream),
        "stream_bytes": stream_bytes,
        "sps_bytes": sps_bytes,
        "unit_bytes": unit_bytes,
        "frames": args.frame_count,
        "width": width,
        "height": height,
        "cell_size": args.cell_size,
        "action_map_shape": list(actions.shape),
        "actions": actions.tolist(),
        "route": {
            "kind": route_kind,
            "selected_variant": route_variant_name,
            "selected_variant_metadata": {
                key: value for key, value in route_variant.items()
                if key != "actions"
            },
        },
        "action_counts": {
            ACTION_NAMES[action]: int(np.count_nonzero(actions == action))
            for action in ACTION_NAMES
        },
        "quality_profile": {
            "Generate": args.generate_qp,
            "Base": args.base_qp,
            "Enhance": args.enhance_qp,
        },
        "scale_interpolation": args.scale_interpolation,
        "model_load_seconds": model_load_seconds,
        "codec_seconds": codec_seconds,
        "total_after_argument_parse_seconds": time.perf_counter() - process_started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "encoder_reconstruction_dir": str(frames_dir),
        "scientific_boundary": {
            "actual_on_disk_entropy_stream": True,
            "all_headers_maps_and_substreams_charged": True,
            "single_spatial_latent_field": True,
            "different_qp_latents_spliced": False,
            "area_prorated_bytes": False,
            "ground_truth_used_by_codec_forward_path": False,
            "route_is_encoder_side_oracle_for_this_probe": (
                "oracle" in route_kind.lower()),
            "route_is_learned_controller": (
                route_kind == "learned-controller"),
            "budget_known_before_encoding": True,
            "training_or_finetuning": False,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "encode_summary.json"
    atomic_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def read_units(path: Path):
    size = path.stat().st_size
    units = []
    sps = None
    with path.open("rb") as source:
        while source.tell() < size:
            header = read_header(source)
            if header["nal_type"] == NalType.NAL_SPS:
                sps = read_sps_remaining(source, header["sps_id"])
                continue
            if header["nal_type"] not in (NalType.NAL_I_SQ, NalType.NAL_P_SQ):
                raise ValueError("spatial decoder encountered a non-spatial coding unit")
            if sps is None or sps["sps_id"] != header["sps_id"]:
                raise ValueError("coding unit references an unavailable SPS")
            record = read_spatial_ip_remaining(source)
            validate_spatial_actions(
                sps["width"], sps["height"], record["cell_size"], record["actions"])
            record["is_i"] = header["nal_type"] == NalType.NAL_I_SQ
            units.append(record)
    if sps is None or not units or not units[0]["is_i"]:
        raise ValueError("stream must contain SPS followed by a spatial I unit")
    if any(unit["is_i"] for unit in units[1:]):
        raise ValueError("current spatial-quality prototype supports exactly one leading I unit")
    return sps, units


@torch.inference_mode()
def decode_main(args: argparse.Namespace, device: torch.device) -> None:
    process_started = time.perf_counter()
    sps, units = read_units(args.input_stream)
    height, width = sps["height"], sps["width"]
    if height % 64 or width % 64:
        raise ValueError("current spatial-quality prototype requires dimensions divisible by 64")
    torch.cuda.reset_peak_memory_stats(device)
    i_net, p_net, model_load_seconds = load_codecs(args, device)
    torch.cuda.synchronize(device)
    decode_started = time.perf_counter()
    reconstructions = []
    decoded_unit_stats = []
    expected_profile = None
    expected_cell_size = None
    expected_interpolation = None
    p_net.clear_dpb()
    for unit_index, unit in enumerate(units):
        profile = (
            unit["qp_generate"], unit["qp_base"], unit["qp_enhance"])
        if expected_profile is None:
            expected_profile = profile
            expected_cell_size = unit["cell_size"]
        elif profile != expected_profile or unit["cell_size"] != expected_cell_size:
            raise ValueError("prototype requires one quality profile and cell size per stream")
        rows = height // unit["cell_size"]
        columns = width // unit["cell_size"]
        actions = np.asarray(unit["actions"], dtype=np.int64).reshape(rows, columns)
        action_map, qp_map = quality_maps(actions, profile, device)
        if unit["is_i"]:
            x_hat = decode_i(
                i_net, unit["bit_stream"], action_map, qp_map, profile,
                height, width, args.scale_interpolation)
            reconstructions.append(x_hat)
            p_net.ref_feature = F.pixel_unshuffle(x_hat, 8)
            frame_count = 1
        else:
            x_hat = decode_p(
                p_net, unit["bit_stream"], action_map, qp_map, profile,
                height, width, args.scale_interpolation)
            reconstructions.extend(x_hat)
            frame_count = len(x_hat)
        streams, ec_values, stored_interpolation = unpack_entropy_payload(
            unit["bit_stream"], args.scale_interpolation)
        if expected_interpolation is None:
            expected_interpolation = stored_interpolation
        elif stored_interpolation != expected_interpolation:
            raise ValueError("coding units use inconsistent interpolation modes")
        decoded_unit_stats.append({
            "index": unit_index,
            "type": "I" if unit["is_i"] else "P8",
            "decoded_frames": frame_count,
            "entropy_payload_bytes": len(unit["bit_stream"]),
            "inner_header_bytes": INNER_HEADER.size,
            "substream_bytes": dict(zip(SUBSTREAM_NAMES, map(len, streams))),
            "entropy_parallelism": dict(zip(SUBSTREAM_NAMES, ec_values)),
        })
    torch.cuda.synchronize(device)
    decode_seconds = time.perf_counter() - decode_started
    frames_dir = args.output_dir / "fresh_decode"
    save_reconstruction(frames_dir, reconstructions, height, width)
    summary = {
        "mode": "decode",
        "format": "DCVC-UF one-shot spatial quality prototype v2",
        "stream": str(args.input_stream),
        "stream_bytes": args.input_stream.stat().st_size,
        "frames": len(reconstructions),
        "width": width,
        "height": height,
        "quality_profile": {
            "Generate": expected_profile[0],
            "Base": expected_profile[1],
            "Enhance": expected_profile[2],
        },
        "cell_size": expected_cell_size,
        "scale_interpolation": expected_interpolation,
        "unit_stats": decoded_unit_stats,
        "model_load_seconds": model_load_seconds,
        "bitstream_decode_seconds": decode_seconds,
        "total_after_argument_parse_seconds": time.perf_counter() - process_started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "fresh_decode_dir": str(frames_dir),
        "source_rgb_read_by_decoder": False,
        "training_or_finetuning": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "decode_summary.json"
    atomic_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("spatial-quality codec requires CUDA")
    set_torch_env()
    torch.cuda.set_device(args.cuda_idx)
    device = torch.device(f"cuda:{args.cuda_idx}")
    if args.mode == "encode":
        encode_main(args, device)
    else:
        decode_main(args, device)


if __name__ == "__main__":
    main()
