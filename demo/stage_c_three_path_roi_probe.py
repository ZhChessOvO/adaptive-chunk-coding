#!/usr/bin/env python3
"""E19: legal Base / Generate / Enhance ROI probe for DCVC-UF.

The released DCVC-UF codec accepts one scalar QP for each I frame or P chunk.
It does not expose a spatial-QP or scalable-layer syntax.  This probe therefore
keeps one ordinary full-frame Base stream and represents every Enhance action
as an independently encoded DCVC-UF tile video.  The outer container stores an
explicit three-action map and all tile descriptors and payloads.  Every byte in
that file is charged.

Generate operates on RGB display output outside the codec reference loop.  The
source-aware selector is an Oracle used only to measure action response; fresh
decode receives only the saved container, codec/restorer weights, and fixed
configuration.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.image_model import DMCI
from src.models.video_model_ht import DMC, g_frame_delay
from src.utils.common import ModelStructure, get_state_dict, set_torch_env
from src.utils.stream_helper import (
    NalType,
    SPSHelper,
    read_header,
    read_ip_remaining,
    read_sps_remaining,
    write_ip,
    write_sps,
)
from src.utils.transforms import rgb2ycbcr, ycbcr2rgb


ACTION_BASE = 0
ACTION_GENERATE = 1
ACTION_ENHANCE = 2
ACTION_NAMES = ("Base", "Generate", "Enhance")

CONTAINER_MAGIC = b"D3ROI001"
CONTAINER_VERSION = 1
CONTAINER_HEADER = struct.Struct("<8sBBBBHHHHIII")
TILE_HEADER = struct.Struct("<HHHHBBI")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E19 legal three-path DCVC-UF ROI validation")
    parser.add_argument("--sequence-name", default="jockey")
    parser.add_argument("--source-dir", type=Path,
                        default=Path("data/test_sequences/PNG/jockey"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("output/e19_three_path_jockey_basicvsrpp"))
    parser.add_argument("--frame-count", type=int, default=17)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--crop-x", type=int, default=0)
    parser.add_argument("--crop-y", type=int, default=0)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--base-qp", type=int, default=32)
    parser.add_argument("--enhance-qp", type=int, default=48)
    parser.add_argument("--baseline-qps", type=int, nargs="+",
                        default=(24, 32, 40, 48))
    parser.add_argument("--enhance-budget-ratio", type=float, default=1.0,
                        help="Extra tile bytes divided by raw Base stream bytes")
    parser.add_argument("--enhance-budget-bytes", type=int, default=-1,
                        help="If nonnegative, overrides --enhance-budget-ratio")
    parser.add_argument("--max-generate-tiles", type=int, default=2)
    parser.add_argument("--restorer", choices=("basicvsrpp", "sdxl", "unsharp"),
                        default="basicvsrpp")
    parser.add_argument(
        "--basicvsrpp-checkpoint", type=Path,
        default=Path("third_party/mmagic/checkpoints/"
                     "basicvsr_plusplus_c128n25_ntire_decompress_track1_"
                     "20210223-7b2eba02.pth"))
    parser.add_argument("--basicvsrpp-spatial-tile", type=int, default=512)
    parser.add_argument("--sdxl-model", type=Path,
                        default=Path("checkpoints/sdxl-inpainting-0.1"))
    parser.add_argument("--sdxl-steps", type=int, default=20)
    parser.add_argument("--sdxl-guidance", type=float, default=5.0)
    parser.add_argument("--sdxl-prompt", default=(
        "high quality natural video frame, faithful realistic texture"))
    parser.add_argument("--sdxl-negative-prompt", default=(
        "text, watermark, deformation, hallucinated objects, artifacts"))
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--decode-repeats", type=int, default=3)
    parser.add_argument("--cuda-idx", type=int, default=0)
    parser.add_argument("--model-path-i", type=Path,
                        default=Path("checkpoints/cvpr2026_image.pth.tar"))
    parser.add_argument("--model-path-p", type=Path,
                        default=Path("checkpoints/cvpr2026_video_hts.pth.tar"))
    parser.add_argument("--skip-thres", type=float, default=0.0)
    parser.add_argument("--reset-interval", type=int, default=32)
    parser.add_argument("--lpips", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--save-frames", action=argparse.BooleanOptionalAction,
                        default=True)
    args = parser.parse_args()

    if not 1 <= args.frame_count:
        parser.error("--frame-count must be positive")
    if args.width % args.tile_size or args.height % args.tile_size:
        parser.error("width and height must be divisible by --tile-size")
    if args.tile_size < 64:
        parser.error("DCVC-UF enhancement tiles must be at least 64 pixels")
    if not 0 <= args.base_qp < 64 or not 0 <= args.enhance_qp < 64:
        parser.error("QP values must be in [0, 63]")
    if args.enhance_qp <= args.base_qp:
        parser.error("DCVC-UF uses larger QP indexes for higher quality")
    if any(not 0 <= value < 64 for value in args.baseline_qps):
        parser.error("baseline QPs must be in [0, 63]")
    if args.max_generate_tiles < 0:
        parser.error("--max-generate-tiles must be nonnegative")
    if args.enhance_budget_ratio < 0:
        parser.error("--enhance-budget-ratio must be nonnegative")
    if args.decode_repeats < 1:
        parser.error("--decode-repeats must be positive")
    return args


def load_source_frames(args: argparse.Namespace) -> tuple[list[Path], list[np.ndarray]]:
    frame_start = int(getattr(args, "frame_start", 0))
    if frame_start < 0:
        raise ValueError("frame start must be nonnegative")
    paths = sorted(args.source_dir.glob("*.png"))[
        frame_start:frame_start + args.frame_count]
    if len(paths) != args.frame_count:
        raise ValueError(
            f"{args.source_dir} has {len(paths)} of {args.frame_count} requested frames")
    frames = []
    for path in paths:
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        crop = rgb[
            args.crop_y:args.crop_y + args.height,
            args.crop_x:args.crop_x + args.width,
        ]
        if crop.shape != (args.height, args.width, 3):
            raise ValueError(f"crop falls outside {path}: got {crop.shape}")
        frames.append(crop.copy())
    return paths, frames


def tensor_from_rgb(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(frame.transpose(2, 0, 1).copy()).unsqueeze(0)
    tensor = tensor.to(device=device, dtype=torch.float32) / 255.0
    return (rgb2ycbcr(tensor).half() - 0.5).to(
        memory_format=torch.channels_last)


def rgb_from_tensor(frame: torch.Tensor, height: int, width: int) -> np.ndarray:
    frame = frame[:, :, :height, :width]
    rgb = ycbcr2rgb(frame + 0.5)
    rgb = torch.clamp(rgb * 255.0, 0, 255).round().byte()
    return rgb.squeeze(0).permute(1, 2, 0).cpu().numpy().copy()


def load_codecs(args: argparse.Namespace, device: torch.device) -> tuple[DMCI, DMC]:
    i_net = DMCI().eval()
    i_net.load_state_dict(get_state_dict(str(args.model_path_i)))
    i_net.update(max(0.0, args.skip_thres))
    i_net = i_net.half().to(device).to(memory_format=torch.channels_last)

    p_net = DMC(ModelStructure.HTS).eval()
    p_net.load_state_dict(get_state_dict(str(args.model_path_p)))
    p_net.update(max(0.0, args.skip_thres))
    p_net = p_net.half().to(device).to(memory_format=torch.channels_last)
    return i_net, p_net


def encode_dcvc_stream(
    frames: list[np.ndarray],
    qp_i: int,
    qp_p: int,
    i_net: DMCI,
    p_net: DMC,
    device: torch.device,
    reset_interval: int,
) -> tuple[bytes, dict]:
    height, width = frames[0].shape[:2]
    padding_r, padding_b = DMCI.get_padding_size(height, width, 16)
    output = io.BytesIO()
    sps_helper = SPSHelper()
    p_net.clear_dpb()
    frame_index = 0
    nal_times = []
    started = time.perf_counter()

    while frame_index < len(frames):
        is_intra = frame_index == 0
        valid_count = 1 if is_intra else min(g_frame_delay, len(frames) - frame_index)
        current = frames[frame_index:frame_index + valid_count]
        if not is_intra and valid_count < g_frame_delay:
            current = current + [current[-1]] * (g_frame_delay - valid_count)
        tensors = [tensor_from_rgb(frame, device) for frame in current]
        source = tensors[0] if is_intra else torch.cat(tensors, dim=1)

        torch.cuda.synchronize(device)
        nal_started = time.perf_counter()
        if is_intra:
            qp = qp_i
            reset = 0
            encoded = i_net.compress(source, qp, padding_b, padding_r)
            p_net.clear_dpb()
            p_net.add_ref_feature_from_frame(encoded["x_hat"])
        else:
            qp = qp_p
            reset = int(
                reset_interval > 0
                and (frame_index + g_frame_delay) % reset_interval == 1)
            encoded = p_net.compress(source, qp, reset, padding_b, padding_r)
        torch.cuda.synchronize(device)
        nal_times.append(time.perf_counter() - nal_started)

        sps = {"sps_id": -1, "height": height, "width": width}
        sps_id, is_new = sps_helper.get_sps_id(sps)
        sps["sps_id"] = sps_id
        if is_new:
            write_sps(output, sps)
        write_ip(
            output, is_intra, sps_id, qp, int(encoded["ec_parallel"]),
            reset, bytes(encoded["bit_stream"]))
        frame_index += valid_count

    torch.cuda.synchronize(device)
    stream = output.getvalue()
    return stream, {
        "seconds": time.perf_counter() - started,
        "nal_seconds": nal_times,
        "bytes": len(stream),
        "frames": len(frames),
        "width": width,
        "height": height,
        "qp_i": qp_i,
        "qp_p": qp_p,
    }


def dcvc_stream_breakdown(data: bytes, frame_count: int) -> dict:
    stream = io.BytesIO(data)
    sps_helper = SPSHelper()
    decoded_frames = 0
    sps_bytes = 0
    nal_header_bytes = 0
    payload_bytes = 0
    nal_count = 0
    while decoded_frames < frame_count:
        start = stream.tell()
        header = read_header(stream)
        if header["nal_type"] == NalType.NAL_SPS:
            sps = read_sps_remaining(stream, header["sps_id"])
            sps_helper.add_sps_by_id(sps)
            sps_bytes += stream.tell() - start
            continue
        _, _, _, payload = read_ip_remaining(stream)
        end = stream.tell()
        payload_bytes += len(payload)
        nal_header_bytes += end - start - len(payload)
        nal_count += 1
        decoded_frames += 1 if header["nal_type"] == NalType.NAL_I else min(
            g_frame_delay, frame_count - decoded_frames)
    if stream.tell() != len(data):
        raise ValueError("trailing bytes in DCVC-UF stream")
    if sps_bytes + nal_header_bytes + payload_bytes != len(data):
        raise RuntimeError("DCVC-UF stream breakdown does not sum to file size")
    return {
        "total_bytes": len(data),
        "sps_bytes": sps_bytes,
        "nal_header_bytes": nal_header_bytes,
        "entropy_payload_bytes": payload_bytes,
        "nal_count": nal_count,
    }


def decode_dcvc_stream(
    data: bytes,
    frame_count: int,
    i_net: DMCI,
    p_net: DMC,
    device: torch.device,
    chunk_observer=None,
) -> list[np.ndarray]:
    stream = io.BytesIO(data)
    sps_helper = SPSHelper()
    p_net.clear_dpb()
    frames = []
    while len(frames) < frame_count:
        header = read_header(stream)
        while header["nal_type"] == NalType.NAL_SPS:
            sps = read_sps_remaining(stream, header["sps_id"])
            sps_helper.add_sps_by_id(sps)
            header = read_header(stream)
        sps = sps_helper.get_sps_by_id(header["sps_id"])
        if sps is None:
            raise ValueError("DCVC-UF NAL references an unknown SPS")
        qp, ec_part, reset, payload = read_ip_remaining(stream)
        if header["nal_type"] == NalType.NAL_I:
            decoded = i_net.decompress(payload, sps, qp, ec_part)
            p_net.clear_dpb()
            p_net.add_ref_feature_from_frame(
                decoded["x_hat"], apply_feature_adaptor=False)
            outputs = [decoded["x_hat"]]
        elif header["nal_type"] == NalType.NAL_P:
            decoded = p_net.decompress(payload, sps, qp, ec_part, reset)
            outputs = decoded["x_hat"]
        else:
            raise ValueError(f"unexpected NAL type {header['nal_type']}")
        if chunk_observer is not None:
            chunk_observer(len(frames), sps, outputs)
        for output in outputs[:frame_count - len(frames)]:
            frames.append(rgb_from_tensor(output, sps["height"], sps["width"]))
    if stream.tell() != len(data):
        raise ValueError("trailing bytes after requested DCVC-UF frames")
    return frames


def tile_boxes(width: int, height: int, tile_size: int) -> list[tuple[int, int, int, int]]:
    return [
        (x, y, tile_size, tile_size)
        for y in range(0, height, tile_size)
        for x in range(0, width, tile_size)
    ]


def pack_actions(actions: np.ndarray) -> bytes:
    flat = np.asarray(actions, dtype=np.uint8).reshape(-1)
    if np.any(flat > ACTION_ENHANCE):
        raise ValueError("invalid action id")
    padded = np.pad(flat, (0, (-flat.size) % 4))
    packed = (
        padded[0::4]
        | (padded[1::4] << 2)
        | (padded[2::4] << 4)
        | (padded[3::4] << 6))
    return packed.tobytes()


def unpack_actions(data: bytes, count: int) -> np.ndarray:
    packed = np.frombuffer(data, dtype=np.uint8)
    flat = np.empty(packed.size * 4, dtype=np.uint8)
    flat[0::4] = packed & 0x03
    flat[1::4] = (packed >> 2) & 0x03
    flat[2::4] = (packed >> 4) & 0x03
    flat[3::4] = (packed >> 6) & 0x03
    result = flat[:count].copy()
    if np.any(result > ACTION_ENHANCE):
        raise ValueError("container action map has a reserved action id")
    return result


@dataclass(frozen=True)
class TilePayload:
    index: int
    x: int
    y: int
    width: int
    height: int
    qp_i: int
    qp_p: int
    stream: bytes


def write_three_path_container(
    path: Path,
    width: int,
    height: int,
    frame_count: int,
    tile_size: int,
    base_qp: int,
    enhance_qp: int,
    actions: np.ndarray,
    base_stream: bytes,
    candidates: list[TilePayload],
) -> dict:
    route = pack_actions(actions)
    chosen = {
        item.index: item for item in candidates
        if actions[item.index] == ACTION_ENHANCE
    }
    header = CONTAINER_HEADER.pack(
        CONTAINER_MAGIC, CONTAINER_VERSION, base_qp, enhance_qp, 0,
        width, height, frame_count, tile_size, actions.size,
        len(route), len(base_stream))
    data = bytearray(header)
    data.extend(route)
    data.extend(base_stream)
    tile_payload_bytes = 0
    for index in sorted(chosen):
        tile = chosen[index]
        descriptor = TILE_HEADER.pack(
            tile.x, tile.y, tile.width, tile.height,
            tile.qp_i, tile.qp_p, len(tile.stream))
        data.extend(descriptor)
        data.extend(tile.stream)
        tile_payload_bytes += len(tile.stream)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    result = {
        "total_bytes": path.stat().st_size,
        "container_header_bytes": len(header),
        "action_map_bytes": len(route),
        "base_stream_bytes": len(base_stream),
        "enhancement_descriptor_bytes": TILE_HEADER.size * len(chosen),
        "enhancement_stream_bytes": tile_payload_bytes,
        "enhancement_tile_count": len(chosen),
    }
    if result["total_bytes"] != sum(
        result[key] for key in (
            "container_header_bytes", "action_map_bytes", "base_stream_bytes",
            "enhancement_descriptor_bytes", "enhancement_stream_bytes")):
        raise RuntimeError("three-path byte accounting does not equal file size")
    return result


def read_three_path_container(path: Path) -> dict:
    data = path.read_bytes()
    if len(data) < CONTAINER_HEADER.size:
        raise ValueError("truncated three-path container")
    fields = CONTAINER_HEADER.unpack(data[:CONTAINER_HEADER.size])
    (magic, version, base_qp, enhance_qp, flags, width, height, frame_count,
     tile_size, action_count, route_len, base_len) = fields
    if magic != CONTAINER_MAGIC or version != CONTAINER_VERSION or flags != 0:
        raise ValueError("unsupported three-path container")
    if not tile_size or width % tile_size or height % tile_size:
        raise ValueError("invalid frame or tile geometry")
    expected_action_count = (width // tile_size) * (height // tile_size)
    if action_count != expected_action_count:
        raise ValueError("action count does not match tile grid")
    if route_len != (action_count + 3) // 4:
        raise ValueError("action-map byte length is not canonical")
    cursor = CONTAINER_HEADER.size
    route = data[cursor:cursor + route_len]
    cursor += route_len
    base = data[cursor:cursor + base_len]
    cursor += base_len
    if len(route) != route_len or len(base) != base_len:
        raise ValueError("truncated action map or Base stream")
    actions = unpack_actions(route, action_count)
    if pack_actions(actions) != route:
        raise ValueError("action-map padding bits are not canonical")
    tiles = {}
    expected = set(np.flatnonzero(actions == ACTION_ENHANCE).tolist())
    while cursor < len(data):
        if cursor + TILE_HEADER.size > len(data):
            raise ValueError("truncated enhancement tile descriptor")
        values = TILE_HEADER.unpack(data[cursor:cursor + TILE_HEADER.size])
        cursor += TILE_HEADER.size
        x, y, tile_width, tile_height, qp_i, qp_p, stream_len = values
        if (tile_width != tile_size or tile_height != tile_size
                or x % tile_size or y % tile_size
                or x + tile_width > width or y + tile_height > height):
            raise ValueError("invalid enhancement tile descriptor")
        if qp_i != enhance_qp or qp_p != enhance_qp or stream_len == 0:
            raise ValueError("enhancement descriptor disagrees with container")
        stream = data[cursor:cursor + stream_len]
        cursor += stream_len
        if len(stream) != stream_len:
            raise ValueError("truncated enhancement tile stream")
        index = (y // tile_size) * (width // tile_size) + (x // tile_size)
        if index in tiles:
            raise ValueError("duplicate enhancement tile")
        tiles[index] = TilePayload(
            index, x, y, tile_width, tile_height, qp_i, qp_p, stream)
    if set(tiles) != expected:
        raise ValueError("action map and enhancement payload list disagree")
    return {
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "tile_size": tile_size,
        "base_qp": base_qp,
        "enhance_qp": enhance_qp,
        "actions": actions,
        "base_stream": base,
        "tiles": tiles,
        "file_bytes": len(data),
    }


class Restorer(Protocol):
    name: str
    local_compute: bool

    def restore(
        self,
        frames: list[np.ndarray],
        selected_tiles: list[int],
        boxes: list[tuple[int, int, int, int]],
    ) -> list[np.ndarray]: ...


class UnsharpRestorer:
    name = "unsharp-deterministic"
    local_compute = False

    def restore(self, frames, selected_tiles, boxes):
        del selected_tiles, boxes
        return [
            np.asarray(
                Image.fromarray(frame).filter(
                    ImageFilter.UnsharpMask(radius=2.0, percent=180, threshold=2)),
                dtype=np.uint8).copy()
            for frame in frames
        ]


def basicvsrpp_forward_restoration(model: torch.nn.Module, frames: torch.Tensor) -> torch.Tensor:
    """Run the same-size RGB output head retained in the released checkpoint."""
    batch, frame_count, channels, height, width = frames.shape
    low = F.interpolate(
        frames.reshape(-1, channels, height, width),
        scale_factor=0.25, mode="bicubic", align_corners=False,
    ).reshape(batch, frame_count, channels, height // 4, width // 4)
    spatial = model.feat_extract(
        frames.reshape(-1, channels, height, width)).reshape(
            batch, frame_count, model.mid_channels, height // 4, width // 4)
    feats = {"spatial": [spatial[:, index] for index in range(frame_count)]}
    forward, backward = model.compute_flow(low)
    for name in model.branch_names:
        feats[name] = []
        model.propagate(feats, backward if "backward" in name else forward, name)
    outputs = []
    for index in range(frame_count):
        merged = [feats["spatial"][index]]
        merged.extend(feats[name][index] for name in model.branch_names)
        output = model.reconstruction(torch.cat(merged, dim=1))
        output = F.leaky_relu(model.upsample1(output), 0.1)
        output = F.leaky_relu(model.upsample2(output), 0.1)
        output = F.leaky_relu(model.conv_hr(output), 0.1)
        output = model.conv_last(output) + frames[:, index]
        outputs.append(output)
    return torch.stack(outputs, dim=1)


class BasicVSRPPRestorer:
    name = "basicvsrpp-ntire-compressed-video-track1"
    local_compute = False

    def __init__(self, checkpoint: Path, device: torch.device, spatial_tile: int):
        from demo.standalone_basicvsrpp import BasicVSRPlusPlusFeatureBackbone

        self.device = device
        self.spatial_tile = spatial_tile
        self.model = BasicVSRPlusPlusFeatureBackbone().eval()
        self.checkpoint_info = self.model.load_released_checkpoint(checkpoint)
        self.model = self.model.to(device)

    @torch.inference_mode()
    def restore(self, frames, selected_tiles, boxes):
        del selected_tiles, boxes
        height, width = frames[0].shape[:2]
        if height % self.spatial_tile or width % self.spatial_tile:
            raise ValueError("BasicVSR++ spatial tiling must divide the frame")
        if self.spatial_tile < 256:
            raise ValueError("BasicVSR++ compressed-video input tiles must be >=256")
        result = np.empty((len(frames), height, width, 3), dtype=np.uint8)
        for y in range(0, height, self.spatial_tile):
            for x in range(0, width, self.spatial_tile):
                batch = np.stack([
                    frame[y:y + self.spatial_tile, x:x + self.spatial_tile]
                    for frame in frames
                ])
                tensor = torch.from_numpy(batch.copy()).permute(0, 3, 1, 2)
                tensor = tensor.unsqueeze(0).to(self.device, dtype=torch.float32) / 255.0
                restored = basicvsrpp_forward_restoration(self.model, tensor)
                restored = torch.clamp(restored * 255.0, 0, 255).round().byte()
                restored = restored.squeeze(0).permute(0, 2, 3, 1).cpu().numpy()
                result[:, y:y + self.spatial_tile, x:x + self.spatial_tile] = restored
        return [frame.copy() for frame in result]


class SDXLTileRestorer:
    name = "sdxl-inpainting-fixed-seed"
    local_compute = True

    def __init__(self, args: argparse.Namespace, device: torch.device):
        from diffusers import AutoPipelineForInpainting

        self.args = args
        self.device = device
        self.pipe = AutoPipelineForInpainting.from_pretrained(
            str(args.sdxl_model), torch_dtype=torch.float16,
            local_files_only=True, use_safetensors=True, variant="fp16")
        self.pipe = self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)

    @torch.inference_mode()
    def restore(self, frames, selected_tiles, boxes):
        outputs = [frame.copy() for frame in frames]
        for frame_index, frame in enumerate(frames):
            base_image = Image.fromarray(frame)
            for tile_index in selected_tiles:
                x, y, width, height = boxes[tile_index]
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                mask[y:y + height, x:x + width] = 255
                generator = torch.Generator(device=self.device).manual_seed(
                    self.args.seed + frame_index * len(boxes) + tile_index)
                generated = self.pipe(
                    prompt=self.args.sdxl_prompt,
                    negative_prompt=self.args.sdxl_negative_prompt,
                    image=base_image,
                    mask_image=Image.fromarray(mask, mode="L"),
                    num_inference_steps=self.args.sdxl_steps,
                    guidance_scale=self.args.sdxl_guidance,
                    generator=generator,
                    height=frame.shape[0],
                    width=frame.shape[1],
                ).images[0].convert("RGB")
                generated = np.asarray(generated, dtype=np.uint8)
                outputs[frame_index][y:y + height, x:x + width] = generated[
                    y:y + height, x:x + width]
        return outputs


def make_restorer(args: argparse.Namespace, device: torch.device) -> Restorer:
    if args.restorer == "unsharp":
        return UnsharpRestorer()
    if args.restorer == "basicvsrpp":
        return BasicVSRPPRestorer(
            args.basicvsrpp_checkpoint, device, args.basicvsrpp_spatial_tile)
    return SDXLTileRestorer(args, device)


def crop_tile_frames(
    frames: list[np.ndarray], box: tuple[int, int, int, int]
) -> list[np.ndarray]:
    x, y, width, height = box
    return [frame[y:y + height, x:x + width].copy() for frame in frames]


def tile_sse(
    reference: list[np.ndarray],
    reconstruction: list[np.ndarray],
    box: tuple[int, int, int, int],
) -> float:
    x, y, width, height = box
    total = 0.0
    for source, decoded in zip(reference, reconstruction):
        delta = (
            source[y:y + height, x:x + width].astype(np.float64)
            - decoded[y:y + height, x:x + width].astype(np.float64))
        total += float(np.sum(delta * delta))
    return total


def select_actions(
    generate_gains: list[float],
    enhance_gains: list[float],
    enhance_costs: list[int],
    byte_budget: int,
    generate_budget: int,
    allow_generate: bool,
    allow_enhance: bool,
) -> np.ndarray:
    # State maps (enhancement bytes, generated tile count) to (gain, actions).
    states: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {
        (0, 0): (0.0, tuple())
    }
    for index in range(len(generate_gains)):
        updated = {}
        for (used_bytes, used_generate), (gain, actions) in states.items():
            choices = [(ACTION_BASE, 0, 0, 0.0)]
            if allow_generate and used_generate < generate_budget:
                choices.append((ACTION_GENERATE, 0, 1, generate_gains[index]))
            if allow_enhance and used_bytes + enhance_costs[index] <= byte_budget:
                choices.append((ACTION_ENHANCE, enhance_costs[index], 0,
                                enhance_gains[index]))
            for action, extra_bytes, extra_generate, extra_gain in choices:
                key = (used_bytes + extra_bytes, used_generate + extra_generate)
                candidate = (gain + extra_gain, actions + (action,))
                incumbent = updated.get(key)
                if incumbent is None or candidate[0] > incumbent[0]:
                    updated[key] = candidate
        states = updated
    best_key, best = max(
        states.items(), key=lambda item: (item[1][0], -item[0][0], -item[0][1]))
    del best_key
    return np.asarray(best[1], dtype=np.uint8)


def composite_actions(
    base: list[np.ndarray],
    generated: list[np.ndarray],
    enhanced_tiles: dict[int, list[np.ndarray]],
    actions: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
) -> list[np.ndarray]:
    outputs = [frame.copy() for frame in base]
    for tile_index, action in enumerate(actions):
        x, y, width, height = boxes[tile_index]
        if action == ACTION_GENERATE:
            for frame_index in range(len(outputs)):
                outputs[frame_index][y:y + height, x:x + width] = generated[
                    frame_index][y:y + height, x:x + width]
        elif action == ACTION_ENHANCE:
            tile_frames = enhanced_tiles[tile_index]
            for frame_index in range(len(outputs)):
                outputs[frame_index][y:y + height, x:x + width] = tile_frames[
                    frame_index]
    return outputs


def rgb_mse(reference: list[np.ndarray], reconstruction: list[np.ndarray]) -> float:
    squared = sum(float(np.sum(
        (source.astype(np.float64) - decoded.astype(np.float64)) ** 2))
        for source, decoded in zip(reference, reconstruction))
    values = sum(source.size for source in reference)
    return squared / values


def psnr_from_mse(value: float) -> float:
    return float("inf") if value == 0 else 10.0 * math.log10(255.0 ** 2 / value)


def temporal_delta_mae(reference: list[np.ndarray], reconstruction: list[np.ndarray]) -> float:
    if len(reference) < 2:
        return 0.0
    values = []
    for index in range(1, len(reference)):
        source_delta = (
            reference[index].astype(np.float32)
            - reference[index - 1].astype(np.float32))
        decoded_delta = (
            reconstruction[index].astype(np.float32)
            - reconstruction[index - 1].astype(np.float32))
        values.append(float(np.mean(np.abs(source_delta - decoded_delta))))
    return float(np.mean(values))


class LPIPSAlex:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.model = None
        if enabled:
            import lpips
            self.model = lpips.LPIPS(net="alex", verbose=False).eval().cpu()

    @torch.inference_mode()
    def evaluate(self, reference, reconstruction) -> float | None:
        if not self.enabled:
            return None
        values = []
        for source, decoded in zip(reference, reconstruction):
            source_tensor = torch.from_numpy(source.copy()).permute(2, 0, 1)
            decoded_tensor = torch.from_numpy(decoded.copy()).permute(2, 0, 1)
            source_tensor = source_tensor.unsqueeze(0).float() / 127.5 - 1.0
            decoded_tensor = decoded_tensor.unsqueeze(0).float() / 127.5 - 1.0
            values.append(float(self.model(source_tensor, decoded_tensor).item()))
        return float(np.mean(values))


def evaluate_variant(reference, reconstruction, lpips_metric: LPIPSAlex) -> dict:
    mse = rgb_mse(reference, reconstruction)
    return {
        "rgb_mse": mse,
        "psnr_db": psnr_from_mse(mse),
        "lpips_alex": lpips_metric.evaluate(reference, reconstruction),
        "temporal_delta_mae": temporal_delta_mae(reference, reconstruction),
    }


def save_frames(root: Path, frames: list[np.ndarray]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames, start=1):
        Image.fromarray(frame).save(root / f"im{index:05d}.png")


def fresh_decode_baseline(
    path: Path,
    frame_count: int,
    repeats: int,
    i_net: DMCI,
    p_net: DMC,
    device: torch.device,
    codec_stream: torch.cuda.Stream,
) -> tuple[list[np.ndarray], dict]:
    times = []
    peaks = []
    canonical = None
    for _ in range(repeats):
        # DCVC-UF's CUDA Graph proxy must run on its dedicated stream.
        torch.cuda.set_stream(codec_stream)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        data = path.read_bytes()
        decoded = decode_dcvc_stream(data, frame_count, i_net, p_net, device)
        torch.cuda.synchronize(device)
        times.append(time.perf_counter() - started)
        peaks.append(int(torch.cuda.max_memory_allocated(device)))
        if canonical is None:
            canonical = decoded
        elif any(not np.array_equal(a, b) for a, b in zip(canonical, decoded)):
            raise RuntimeError("baseline fresh decodes are not deterministic")
    return canonical, {
        "fresh_decode_repeats": repeats,
        "fresh_decode_seconds_median": float(statistics.median(times)),
        "fresh_decode_seconds_all": times,
        "peak_cuda_allocated_bytes": max(peaks),
        "file_io_in_timing": True,
        "model_load_in_timing": False,
        "source_rgb_available_to_decoder": False,
    }


def fresh_decode_container(
    path: Path,
    repeats: int,
    restorer: Restorer,
    base_i_net: DMCI,
    base_p_net: DMC,
    tile_i_net: DMCI,
    tile_p_net: DMC,
    device: torch.device,
    codec_stream: torch.cuda.Stream,
) -> tuple[list[np.ndarray], dict]:
    times = []
    codec_times = []
    restorer_times = []
    peaks = []
    canonical = None
    for _ in range(repeats):
        # Keep DCVC-UF CUDA Graph work isolated from torchvision/diffusers.
        torch.cuda.set_stream(codec_stream)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        parsed = read_three_path_container(path)

        codec_started = time.perf_counter()
        base = decode_dcvc_stream(
            parsed["base_stream"], parsed["frame_count"],
            base_i_net, base_p_net, device)
        enhanced = {}
        for index, tile in sorted(parsed["tiles"].items()):
            enhanced[index] = decode_dcvc_stream(
                tile.stream, parsed["frame_count"],
                tile_i_net, tile_p_net, device)
        torch.cuda.synchronize(device)
        codec_seconds = time.perf_counter() - codec_started

        boxes = tile_boxes(parsed["width"], parsed["height"], parsed["tile_size"])
        generate_tiles = np.flatnonzero(
            parsed["actions"] == ACTION_GENERATE).tolist()
        torch.cuda.set_stream(torch.cuda.default_stream(device))
        restorer_started = time.perf_counter()
        generated = (
            restorer.restore(base, generate_tiles, boxes)
            if generate_tiles else [frame.copy() for frame in base])
        torch.cuda.synchronize(device)
        restorer_seconds = time.perf_counter() - restorer_started
        output = composite_actions(
            base, generated, enhanced, parsed["actions"], boxes)
        torch.cuda.synchronize(device)
        times.append(time.perf_counter() - started)
        codec_times.append(codec_seconds)
        restorer_times.append(restorer_seconds)
        peaks.append(int(torch.cuda.max_memory_allocated(device)))
        if canonical is None:
            canonical = output
        elif any(not np.array_equal(a, b) for a, b in zip(canonical, output)):
            raise RuntimeError("container fresh decodes are not deterministic")
    return canonical, {
        "fresh_decode_repeats": repeats,
        "fresh_decode_seconds_median": float(statistics.median(times)),
        "fresh_decode_seconds_all": times,
        "codec_decode_seconds_median": float(statistics.median(codec_times)),
        "restorer_seconds_median": float(statistics.median(restorer_times)),
        "peak_cuda_allocated_bytes": max(peaks),
        "restorer_local_compute": bool(restorer.local_compute),
        "file_io_in_timing": True,
        "model_load_in_timing": False,
        "source_rgb_available_to_decoder": False,
    }


def action_counts(actions: np.ndarray) -> dict:
    return {
        name: int(np.sum(actions == index))
        for index, name in enumerate(ACTION_NAMES)
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary: dict, region_rows: list[dict]) -> None:
    lines = [
        f"# E19 {summary['sequence']} 合法三路验证",
        "",
        "## 结论",
        "",
        summary["conclusion"],
        "",
        "## 码流合法性",
        "",
        "发布版 DCVC-UF 每个 I/P 单元只使用一个 QP；本实验没有拼接不同 QP latent。"
        "增强区域是独立编码并可单独 fresh decode 的 ROI 视频流，所有头部、action map、Base 与增强 payload 均按落盘文件计费。",
        "",
        "## 结果",
        "",
        "| 方案 | 文件字节 | PSNR | LPIPS | 时序差分误差 | fresh decode 秒 | 峰值显存字节 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, record in summary["variants"].items():
        quality = record["quality"]
        runtime = record["runtime"]
        lines.append(
            f"| {name} | {record['rate']['total_bytes']} | "
            f"{quality['psnr_db']:.5f} | "
            f"{quality['lpips_alex'] if quality['lpips_alex'] is not None else '未测'} | "
            f"{quality['temporal_delta_mae']:.5f} | "
            f"{runtime['fresh_decode_seconds_median']:.5f} | "
            f"{runtime['peak_cuda_allocated_bytes']} |")
    lines.extend([
        "",
        "## 区域响应",
        "",
        "| tile | Generate PSNR收益(dB) | Enhance PSNR收益(dB) | Enhance真实额外字节 |",
        "|---:|---:|---:|---:|",
    ])
    for row in region_rows:
        lines.append(
            f"| {row['tile_index']} | {row['generate_psnr_gain_db']:.5f} | "
            f"{row['enhance_psnr_gain_db']:.5f} | {row['enhance_cost_bytes']} |")
    lines.extend([
        "",
        "## 边界",
        "",
        "- 路由使用原图选择，是编码端 Oracle 诊断，不是已经训练好的控制器。",
        "- Generate 只影响显示输出，不进入 DCVC-UF 参考环路。",
        "- BasicVSR++ 当前按整段/整幅 tile 执行；只融合少量区域不等于局部省算。",
        "- 本轮不使用 true-fill，不读取任何未发送 latent。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("E19 requires CUDA")
    set_torch_env()
    torch.cuda.set_device(args.cuda_idx)
    device = torch.device(f"cuda:{args.cuda_idx}")
    codec_stream = torch.cuda.Stream(device=device)
    torch.cuda.set_stream(codec_stream)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_paths, originals = load_source_frames(args)
    boxes = tile_boxes(args.width, args.height, args.tile_size)
    grid_shape = (args.height // args.tile_size, args.width // args.tile_size)
    base_i_net, base_p_net = load_codecs(args, device)

    # Ordinary codec QP sweep.  The selected Base stream is reused byte-for-byte
    # by every three-path container.
    baseline_records = {}
    base_stream = None
    base_decoded = None
    for qp in sorted(set(args.baseline_qps) | {args.base_qp}):
        stream, encode_stats = encode_dcvc_stream(
            originals, qp, qp,
            base_i_net, base_p_net, device, args.reset_interval)
        stream_path = args.output_dir / "streams" / f"all_base_qp{qp}.dcvc"
        stream_path.parent.mkdir(parents=True, exist_ok=True)
        stream_path.write_bytes(stream)
        decoded, runtime = fresh_decode_baseline(
            stream_path, len(originals), args.decode_repeats,
            base_i_net, base_p_net, device, codec_stream)
        baseline_records[qp] = {
            "path": str(stream_path),
            "stream": stream,
            "decoded": decoded,
            "encode": encode_stats,
            "runtime": runtime,
            "rate": dcvc_stream_breakdown(stream, len(originals)),
        }
        if qp == args.base_qp:
            base_stream = stream
            base_decoded = decoded
    if base_stream is None or base_decoded is None:
        raise RuntimeError("failed to construct selected Base stream")

    # Every enhancement candidate is a complete, independently decodable
    # DCVC-UF tile stream.  Candidate search bytes are recorded but only chosen
    # tile streams enter the transmitted container.
    # A separate decoder context is mandatory here: the released CUDA proxy
    # captures resolution-specific state and must not alternate between the
    # full-frame Base stream and independently coded ROI streams.
    tile_i_net, tile_p_net = load_codecs(args, device)
    tile_payloads = []
    enhanced_candidates = {}
    candidate_encode_seconds = 0.0
    for index, box in enumerate(boxes):
        tile_source = crop_tile_frames(originals, box)
        stream, encode_stats = encode_dcvc_stream(
            tile_source, args.enhance_qp, args.enhance_qp,
            tile_i_net, tile_p_net, device, args.reset_interval)
        x, y, width, height = box
        candidate_path = (
            args.output_dir / "candidate_enhancement_streams"
            / f"tile_{index:03d}_x{x}_y{y}_qp{args.enhance_qp}.dcvc")
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_path.write_bytes(stream)
        decoded, _ = fresh_decode_baseline(
            candidate_path, len(originals), 1,
            tile_i_net, tile_p_net, device, codec_stream)
        dcvc_stream_breakdown(stream, len(originals))
        tile_payloads.append(TilePayload(
            index, x, y, width, height,
            args.enhance_qp, args.enhance_qp, stream))
        enhanced_candidates[index] = decoded
        candidate_encode_seconds += encode_stats["seconds"]

    restorer = make_restorer(args, device)
    all_tiles = list(range(len(boxes)))
    torch.cuda.synchronize(device)
    torch.cuda.set_stream(torch.cuda.default_stream(device))
    candidate_restore_started = time.perf_counter()
    restored_candidates = restorer.restore(base_decoded, all_tiles, boxes)
    torch.cuda.synchronize(device)
    candidate_restore_seconds = time.perf_counter() - candidate_restore_started

    base_sse = []
    generate_sse = []
    enhance_sse = []
    generate_gains = []
    enhance_gains = []
    enhance_costs = []
    region_rows = []
    tile_value_count = len(originals) * args.tile_size * args.tile_size * 3
    for index, (box, payload) in enumerate(zip(boxes, tile_payloads)):
        base_value = tile_sse(originals, base_decoded, box)
        generate_value = tile_sse(originals, restored_candidates, box)
        x, y, width, height = box
        enhanced_full = [frame.copy() for frame in base_decoded]
        for frame_index, tile in enumerate(enhanced_candidates[index]):
            enhanced_full[frame_index][y:y + height, x:x + width] = tile
        enhance_value = tile_sse(originals, enhanced_full, box)
        base_sse.append(base_value)
        generate_sse.append(generate_value)
        enhance_sse.append(enhance_value)
        generate_gains.append(base_value - generate_value)
        enhance_gains.append(base_value - enhance_value)
        enhance_costs.append(TILE_HEADER.size + len(payload.stream))
        base_mse = base_value / tile_value_count
        region_rows.append({
            "tile_index": index,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "base_psnr_db": psnr_from_mse(base_mse),
            "generate_psnr_db": psnr_from_mse(generate_value / tile_value_count),
            "enhance_psnr_db": psnr_from_mse(enhance_value / tile_value_count),
            "generate_psnr_gain_db": (
                psnr_from_mse(generate_value / tile_value_count)
                - psnr_from_mse(base_mse)),
            "enhance_psnr_gain_db": (
                psnr_from_mse(enhance_value / tile_value_count)
                - psnr_from_mse(base_mse)),
            "generate_sse_reduction": base_value - generate_value,
            "enhance_sse_reduction": base_value - enhance_value,
            "enhance_stream_bytes": len(payload.stream),
            "enhance_descriptor_bytes": TILE_HEADER.size,
            "enhance_cost_bytes": enhance_costs[-1],
        })

    enhance_budget = (
        args.enhance_budget_bytes
        if args.enhance_budget_bytes >= 0
        else int(round(len(base_stream) * args.enhance_budget_ratio)))
    actions_by_variant = {
        "base-only": np.full(len(boxes), ACTION_BASE, dtype=np.uint8),
        "base+restorer-all": np.full(len(boxes), ACTION_GENERATE, dtype=np.uint8),
        "generate-only-oracle": select_actions(
            generate_gains, enhance_gains, enhance_costs,
            0, args.max_generate_tiles, True, False),
        "enhance-only-oracle": select_actions(
            generate_gains, enhance_gains, enhance_costs,
            enhance_budget, 0, False, True),
        "joint-three-path-oracle": select_actions(
            generate_gains, enhance_gains, enhance_costs,
            enhance_budget, args.max_generate_tiles, True, True),
    }

    lpips_metric = LPIPSAlex(args.lpips)
    variants = {}
    decoded_outputs = {}
    for name, actions in actions_by_variant.items():
        path = args.output_dir / "containers" / f"{name}.d3r"
        rate = write_three_path_container(
            path, args.width, args.height, len(originals), args.tile_size,
            args.base_qp, args.enhance_qp, actions, base_stream, tile_payloads)
        decoded, runtime = fresh_decode_container(
            path, args.decode_repeats, restorer,
            base_i_net, base_p_net, tile_i_net, tile_p_net,
            device, codec_stream)
        parsed = read_three_path_container(path)
        if parsed["file_bytes"] != rate["total_bytes"]:
            raise RuntimeError("fresh container size differs from accounting")
        decoded_outputs[name] = decoded
        variants[name] = {
            "path": str(path),
            "actions": actions.tolist(),
            "action_counts": action_counts(actions),
            "rate": rate,
            "quality": evaluate_variant(originals, decoded, lpips_metric),
            "runtime": runtime,
        }

    # Add ordinary scalar-QP baselines without changing their stock streams.
    for qp, record in baseline_records.items():
        name = f"ordinary-dcvc-qp{qp}"
        variants[name] = {
            "path": record["path"],
            "actions": None,
            "action_counts": {"Base": len(boxes), "Generate": 0, "Enhance": 0},
            "rate": record["rate"],
            "quality": evaluate_variant(
                originals, record["decoded"], lpips_metric),
            "runtime": record["runtime"],
            "encode": record["encode"],
        }

    base_quality = variants["base-only"]["quality"]
    joint_quality = variants["joint-three-path-oracle"]["quality"]
    generate_positive = sum(value > 0 for value in generate_gains)
    enhance_positive = sum(value > 0 for value in enhance_gains)
    both_positive = sum(
        generate > 0 and enhance > 0
        for generate, enhance in zip(generate_gains, enhance_gains))
    if generate_positive == 0:
        conclusion = (
            "该恢复器在所有区域都未产生正的像素保真收益；联合方案若有提升，"
            "只能归因于真实编码增强，不能记作 Generate 贡献。")
    elif enhance_positive == 0:
        conclusion = (
            "独立 ROI 增强在当前质量档位下没有正收益，需要先检查 tile 边界、"
            "重复 I 帧和 QP选择；暂不训练控制器。")
    else:
        conclusion = (
            f"{len(boxes)} 个区域中 Generate 有 {generate_positive} 个正收益，"
            f"Enhance 有 {enhance_positive} 个正收益，二者同时为正有 {both_positive} 个。"
            f"联合方案相对 Base 的 PSNR 变化为 "
            f"{joint_quality['psnr_db'] - base_quality['psnr_db']:+.5f} dB。"
            "这只证明首轮区域响应，尚不证明学习控制器或最终 Pareto 收益。")

    summary = {
        "experiment": "E19 legal Base/Generate/Enhance ROI probe",
        "sequence": args.sequence_name,
        "source_role": "previously used regression/development data",
        "source_files": [str(path) for path in source_paths],
        "crop": {
            "x": args.crop_x, "y": args.crop_y,
            "width": args.width, "height": args.height,
        },
        "frames": len(originals),
        "codec_audit": {
            "released_spatial_qp_supported": False,
            "released_scalable_layer_supported": False,
            "enhancement_format": (
                "full-frame Base DCVC-UF stream plus independent high-QP "
                "DCVC-UF ROI tile streams"),
            "different_qp_latents_spliced": False,
            "full_high_quality_stream_area_prorated": False,
        },
        "configuration": {
            "base_qp": args.base_qp,
            "enhance_qp": args.enhance_qp,
            "baseline_qps": sorted(baseline_records),
            "tile_size": args.tile_size,
            "tile_grid": list(grid_shape),
            "enhance_budget_bytes": enhance_budget,
            "enhance_budget_ratio_of_base_stream": (
                enhance_budget / len(base_stream)),
            "max_generate_tiles": args.max_generate_tiles,
            "restorer": restorer.name,
            "restorer_local_compute": restorer.local_compute,
            "basicvsrpp_checkpoint": str(args.basicvsrpp_checkpoint),
            "basicvsrpp_spatial_tile": args.basicvsrpp_spatial_tile,
            "sdxl_model": str(args.sdxl_model),
            "sdxl_steps": args.sdxl_steps,
            "sdxl_guidance": args.sdxl_guidance,
            "sdxl_prompt": args.sdxl_prompt,
            "sdxl_negative_prompt": args.sdxl_negative_prompt,
            "seed": args.seed,
            "reset_interval": args.reset_interval,
            "decode_repeats": args.decode_repeats,
        },
        "oracle_warning": (
            "Action selection uses original RGB only at the encoder for response "
            "measurement. Fresh decode receives no source RGB or omitted latent."),
        "candidate_search": {
            "encoded_enhancement_tile_count": len(tile_payloads),
            "enhancement_candidate_encode_seconds": candidate_encode_seconds,
            "restoration_candidate_seconds": candidate_restore_seconds,
            "search_not_in_decode_runtime": True,
        },
        "region_response": {
            "generate_positive_tiles": generate_positive,
            "enhance_positive_tiles": enhance_positive,
            "both_positive_tiles": both_positive,
        },
        "variants": variants,
        "conclusion": conclusion,
        "limitations": [
            "Oracle action selection is not a deployable controller.",
            "Independent ROI streams repeat intra and boundary information.",
            "BasicVSR++ spatial fusion does not reduce compute when fewer tiles are selected.",
            "The SDXL ablation is image inpainting and not temporally trained.",
        ],
    }

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    write_csv(args.output_dir / "per_region.csv", region_rows)
    variant_rows = []
    for name, record in variants.items():
        variant_rows.append({
            "variant": name,
            "total_bytes": record["rate"]["total_bytes"],
            "psnr_db": record["quality"]["psnr_db"],
            "lpips_alex": record["quality"]["lpips_alex"],
            "temporal_delta_mae": record["quality"]["temporal_delta_mae"],
            "fresh_decode_seconds": record["runtime"]["fresh_decode_seconds_median"],
            "codec_decode_seconds": record["runtime"].get(
                "codec_decode_seconds_median",
                record["runtime"]["fresh_decode_seconds_median"]),
            "restorer_seconds": record["runtime"].get(
                "restorer_seconds_median", 0.0),
            "peak_cuda_allocated_bytes": record["runtime"][
                "peak_cuda_allocated_bytes"],
            "base_tiles": record["action_counts"]["Base"],
            "generate_tiles": record["action_counts"]["Generate"],
            "enhance_tiles": record["action_counts"]["Enhance"],
        })
    write_csv(args.output_dir / "per_variant.csv", variant_rows)
    write_report(args.output_dir / "report.md", summary, region_rows)

    if args.save_frames:
        save_frames(args.output_dir / "frames" / "original", originals)
        for name, frames in decoded_outputs.items():
            save_frames(args.output_dir / "frames" / name, frames)

    print(json.dumps({
        "summary": str(summary_path),
        "conclusion": conclusion,
        "base_bytes": len(base_stream),
        "enhance_budget_bytes": enhance_budget,
        "restorer": restorer.name,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
