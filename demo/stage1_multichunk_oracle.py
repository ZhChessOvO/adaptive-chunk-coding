#!/usr/bin/env python3
"""Multi-chunk, real-bitstream Oracle probe for frozen HT-S.

This script deliberately separates three questions that were conflated in the
first Stage-1 probe:

1. Can every stream be decoded from bytes written to disk?
2. Does a routed reconstruction remain synchronized when it is propagated to
   later chunks?
3. How much rate can a distortion-aware, encoder-side search save at a chosen
   per-chunk PSNR-loss allowance?

The search is an achievable greedy-prefix Oracle proxy, not a mathematical
upper bound over all 2^N route maps.  It uses the source chunk to rank
counterfactual single-block removals, then re-encodes and evaluates complete
prefix route maps with actual rANS and route bytes.  Final metrics come from a
fresh decode of the complete on-disk sequence container.
"""

import argparse
import csv
import json
import math
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

from demo.stage1_token_skipping import (  # noqa: E402
    CONTAINER_HEADER,
    build_route_section,
    decode_y,
    decode_z,
    encode_y,
    encode_z,
    parse_container,
    psnr,
    rgb_from_recon,
    write_container,
)
from src.models.image_model import DMCI  # noqa: E402
from src.models.video_model_ht import (  # noqa: E402
    DMC,
    g_ch_z,
    g_frame_delay,
)
from src.utils.common import ModelStructure, get_state_dict, set_torch_env  # noqa: E402
from src.utils.transforms import rgb2ycbcr, ycbcr2rgb  # noqa: E402


SEQ_MAGIC = b"D1MSEQ01"
SEQ_VERSION = 1
SEQ_HEADER = struct.Struct("<8sBBBBB3xHHHHI")
CHUNK_LENGTH = struct.Struct("<I")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Frozen HT-S multi-chunk greedy Oracle with real bitstreams.")
    parser.add_argument(
        "--sequence", action="append", default=[], metavar="NAME=PATH",
        help="Sequence name and PNG directory. Repeat for multiple sequences.")
    parser.add_argument("--output-dir", default="output/stage1_multichunk_oracle_qp32")
    parser.add_argument("--model-path-i", default="checkpoints/cvpr2026_image.pth.tar")
    parser.add_argument("--model-path-p", default="checkpoints/cvpr2026_video_hts.pth.tar")
    parser.add_argument("--qp-i", type=int, default=32)
    parser.add_argument("--qp-p", type=int, default=32)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument(
        "--frame-count", type=int, default=0,
        help="Maximum frames per sequence; 0 uses every available frame.")
    parser.add_argument(
        "--max-chunks", type=int, default=0,
        help="Maximum P chunks after the I frame; 0 includes the final padded chunk.")
    parser.add_argument(
        "--latent-block-size", type=int, choices=(1, 2, 4), default=2,
        help="Spatial edge of an all-256-channel y routing unit.")
    parser.add_argument(
        "--drop-targets", type=float, nargs="+", default=(0.05, 0.10, 0.20),
        help="Allowed local PSNR loss in dB for independent routed trajectories.")
    parser.add_argument(
        "--candidate-pool", type=int, default=32,
        help="Top estimated-rate blocks receiving single-block counterfactuals.")
    parser.add_argument(
        "--prefix-sizes", type=int, nargs="+", default=(1, 2, 4, 8, 12, 16, 24, 32),
        help="Greedy-prefix sizes that are re-encoded with actual streams.")
    parser.add_argument(
        "--y-ec-parallel", type=int, choices=range(0, 9), default=0,
        help="Fixed y rANS parallelism in [1,8]; 0 keeps stock count-adaptive behavior.")
    parser.add_argument("--skip-thres", type=float, default=0.0)
    parser.add_argument("--reset-interval", type=int, default=32)
    parser.add_argument("--cuda-idx", type=int, default=0)
    parser.add_argument("--save-recon", action="store_true")
    return parser.parse_args()


def parse_sequences(items):
    if not items:
        items = [
            "jockey=data/test_sequences/PNG/jockey",
            "shake=data/test_sequences/PNG/shake",
            "sky=data/test_sequences/PNG/sky",
            "uvg=data/test_sequences/PNG/uvg",
        ]
    sequences = []
    seen = set()
    for item in items:
        if "=" not in item:
            raise ValueError(f"sequence must be NAME=PATH, got {item!r}")
        name, path = item.split("=", 1)
        name = name.strip()
        if not name or name in seen:
            raise ValueError(f"invalid or duplicate sequence name {name!r}")
        source = Path(path)
        if not source.is_dir():
            raise FileNotFoundError(source)
        seen.add(name)
        sequences.append((name, source))
    return sequences


def load_rgb_frames(source_dir, width, height, frame_count, max_chunks):
    files = sorted(source_dir.glob("im*.png"))
    if not files:
        files = sorted(source_dir.glob("*.png"))
    if frame_count > 0:
        files = files[:frame_count]
    if max_chunks > 0:
        files = files[:1 + max_chunks * g_frame_delay]
    if len(files) < 2:
        raise ValueError(f"{source_dir} needs at least an I frame and one P frame")
    frames = []
    for path in files:
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        if image.shape != (height, width, 3):
            raise ValueError(f"{path} has shape {image.shape}, expected {(height, width, 3)}")
        frames.append(image.transpose(2, 0, 1).copy())
    return files, frames


def model_frame(rgb, device):
    tensor = torch.from_numpy(rgb).unsqueeze(0).float() / 255.0
    return rgb2ycbcr(tensor).to(device=device, dtype=torch.float16)


def make_chunk(frames, first_index, device):
    valid = frames[first_index:first_index + g_frame_delay]
    valid_count = len(valid)
    padded = list(valid)
    while len(padded) < g_frame_delay:
        padded.append(padded[-1])
    tensors = [model_frame(frame, device) for frame in padded]
    chunk = torch.cat(tensors, dim=1)
    chunk = (chunk - 0.5).to(memory_format=torch.channels_last)
    target = torch.from_numpy(np.stack(valid)).to(
        device=device, dtype=torch.float32) / 255.0
    return chunk, target, valid_count


def load_models(args, device):
    i_net = DMCI().eval()
    i_net.load_state_dict(get_state_dict(args.model_path_i))
    i_net.update(max(0.0, args.skip_thres))
    i_net = i_net.half().to(device).to(memory_format=torch.channels_last)

    p_net = DMC(ModelStructure.HTS).eval()
    p_net.load_state_dict(get_state_dict(args.model_path_p))
    p_net.update(max(0.0, args.skip_thres))
    p_net = p_net.half().to(device).to(memory_format=torch.channels_last)
    return i_net, p_net


def decode_i_stream(i_net, stream, ec_parallel, qp, height, width):
    return i_net.decompress(
        stream, {"height": height, "width": width}, qp, ec_parallel)["x_hat"]


def initialize_p_state(p_net, i_hat):
    p_net.clear_dpb()
    p_net.ref_feature = F.pixel_unshuffle(i_hat, 8)


def should_reset(chunk_index, reset_interval):
    if reset_interval <= 0:
        return False
    frame_index = 1 + chunk_index * g_frame_delay
    return (frame_index + g_frame_delay) % reset_interval == 1


def get_q_params(p_net, qp):
    q_encoder = p_net.q_encoder[qp:qp + 1, :, None, None]
    q_feature = p_net.q_feature[qp:qp + 1, :, None, None]
    q_decoder = p_net.q_decoder[qp:qp + 1, :, None, None]
    return q_encoder, q_feature, q_decoder


def reconstruction_float(p_net, y_hat, q_decoder, valid_count):
    x_hat, feature = p_net.get_recon_and_feature(y_hat, p_net.ctx, q_decoder)
    rgb = torch.cat([
        torch.clamp(ycbcr2rgb(frame + 0.5), 0.0, 1.0)
        for frame in x_hat[:valid_count]
    ], dim=0)
    return rgb, x_hat, feature


def mse_against_target(recon_rgb, target_rgb):
    return float(torch.mean((recon_rgb.float() - target_rgb) ** 2).item())


def psnr_from_unit_mse(mse):
    return float("inf") if mse <= 0 else -10.0 * math.log10(mse)


def prepare_chunk_latents(p_net, chunk, qp):
    p_net.apply_feature_adaptor()
    q_encoder, q_feature, q_decoder = get_q_params(p_net, qp)
    y = p_net.encoder(chunk, p_net.ctx, q_encoder)
    z = p_net.hyper_encoder(y)
    z_hat = torch.round(z).clamp_(-128, 127)
    global_stream, z_ec = encode_z(
        z_hat, qp, p_net.bit_estimator_z.get_cdf_info())
    decoded_z = decode_z(
        global_stream, qp, z_hat.shape,
        p_net.bit_estimator_z.get_cdf_info(), z_ec)
    if not torch.equal(decoded_z, z_hat.cpu()):
        raise RuntimeError("global z in-memory rANS round-trip mismatch")
    decoded_z = decoded_z.to(
        device=y.device, dtype=z_hat.dtype, memory_format=torch.channels_last)
    common_params = p_net.res_prior_param_decoder(decoded_z, p_net.memory, q_feature)
    return {
        "y": y,
        "z_shape": tuple(z_hat.shape),
        "global_stream": global_stream,
        "z_ec": z_ec,
        "common_params": common_params,
        "q_feature": q_feature,
        "q_decoder": q_decoder,
    }


def decode_chunk_bytes(p_net, data, block_size, z_shape, q_feature, q_decoder):
    parsed = parse_container(data, block_size)
    if parsed["z_ec"] is None:
        raise ValueError("multi-chunk decode requires a version-2 chunk with serialized z EC")
    decoded_z = decode_z(
        parsed["global"], parsed["qp"], z_shape,
        p_net.bit_estimator_z.get_cdf_info(), parsed["z_ec"])
    decoded_z = decoded_z.to(
        device=p_net.memory.device, dtype=p_net.memory.dtype,
        memory_format=torch.channels_last)
    common_params = p_net.res_prior_param_decoder(decoded_z, p_net.memory, q_feature)
    decoded_q, decoded_y = decode_y(
        p_net, parsed["base"], parsed["y_ec"], common_params,
        parsed["skip_blocks"], block_size)
    x_hat, feature = p_net.get_recon_and_feature(decoded_y, p_net.ctx, q_decoder)
    return parsed, decoded_q, x_hat, feature


def route_total_bytes(global_stream, route_section, y_stream):
    return CONTAINER_HEADER.size + len(global_stream) + len(route_section) + len(y_stream)


def search_route(p_net, prepared, target_rgb, valid_count, block_size,
                 drop_target, candidate_pool, prefix_sizes, y_ec_parallel):
    y = prepared["y"]
    common_params = prepared["common_params"]
    q_decoder = prepared["q_decoder"]
    grid_shape = (y.shape[-2] // block_size, y.shape[-1] // block_size)
    all_base = np.zeros(grid_shape, dtype=np.bool_)
    base_y = encode_y(
        p_net, y, common_params, all_base, block_size,
        ec_parallel_override=y_ec_parallel)
    base_rgb, _, _ = reconstruction_float(
        p_net, base_y.y_hat, q_decoder, valid_count)
    base_mse = mse_against_target(base_rgb, target_rgb)
    base_psnr = psnr_from_unit_mse(base_mse)
    base_total = route_total_bytes(prepared["global_stream"], b"", base_y.stream)

    rates = np.nan_to_num(
        base_y.expected_bits_by_block.reshape(-1), nan=0.0, posinf=1e12, neginf=0.0)
    pool_size = min(max(1, candidate_pool), rates.size)
    pool = np.argsort(rates, kind="stable")[::-1][:pool_size]
    single = []
    distortion_floor = 1e-12
    for flat_index in pool:
        skip = np.zeros(grid_shape, dtype=np.bool_)
        skip.reshape(-1)[flat_index] = True
        trial = encode_y(
            p_net, y, common_params, skip, block_size,
            ec_parallel_override=y_ec_parallel)
        trial_rgb, _, _ = reconstruction_float(
            p_net, trial.y_hat, q_decoder, valid_count)
        trial_mse = mse_against_target(trial_rgb, target_rgb)
        delta_mse = trial_mse - base_mse
        score = float(rates[flat_index]) / max(delta_mse, distortion_floor)
        single.append({
            "flat_index": int(flat_index),
            "row": int(flat_index // grid_shape[1]),
            "col": int(flat_index % grid_shape[1]),
            "estimated_bits": float(rates[flat_index]),
            "delta_mse": float(delta_mse),
            "single_psnr_drop_db": float(base_psnr - psnr_from_unit_mse(trial_mse)),
            "score": float(score),
        })
    single.sort(key=lambda item: (item["score"], item["estimated_bits"]), reverse=True)
    order = [item["flat_index"] for item in single]

    sizes = {0, len(order)}
    sizes.update(size for size in prefix_sizes if 0 <= size <= len(order))
    frontier = []
    encodings = {}
    for size in sorted(sizes):
        skip = np.zeros(grid_shape, dtype=np.bool_)
        if size:
            skip.reshape(-1)[order[:size]] = True
        trial = base_y if size == 0 else encode_y(
            p_net, y, common_params, skip, block_size,
            ec_parallel_override=y_ec_parallel)
        route = b"" if size == 0 else build_route_section(skip, block_size)
        if size == 0:
            trial_mse = base_mse
        else:
            trial_rgb, _, _ = reconstruction_float(
                p_net, trial.y_hat, q_decoder, valid_count)
            trial_mse = mse_against_target(trial_rgb, target_rgb)
        total = route_total_bytes(prepared["global_stream"], route, trial.stream)
        point = {
            "prefix_blocks": int(size),
            "total_bytes": int(total),
            "global_z_bytes": len(prepared["global_stream"]),
            "route_bytes": len(route),
            "base_y_bytes": len(trial.stream),
            "float_psnr": psnr_from_unit_mse(trial_mse),
            "local_psnr_drop_db": float(base_psnr - psnr_from_unit_mse(trial_mse)),
            "net_bytes_saved": int(base_total - total),
        }
        frontier.append(point)
        encodings[size] = (skip, trial, route, trial_mse)

    feasible = [
        point for point in frontier
        if point["local_psnr_drop_db"] <= drop_target + 1e-7
        and point["total_bytes"] < base_total
    ]
    if feasible:
        chosen_point = min(
            feasible, key=lambda point: (point["total_bytes"], -point["float_psnr"]))
    else:
        chosen_point = frontier[0]
    chosen_size = chosen_point["prefix_blocks"]
    chosen_skip, chosen_y, chosen_route, _ = encodings[chosen_size]
    diagnostics = {
        "oracle_kind": "source-aware greedy single-block ranking plus actual prefix search",
        "mathematical_upper_bound": False,
        "drop_target_db": float(drop_target),
        "local_all_base_float_psnr": float(base_psnr),
        "candidate_pool": int(pool_size),
        "y_ec_parallel_mode": (
            "count_adaptive" if y_ec_parallel is None else f"fixed_{y_ec_parallel}"),
        "single_block_ranking": single,
        "tested_prefixes": frontier,
        "chosen_prefix_blocks": int(chosen_size),
    }
    return chosen_skip, chosen_y, chosen_route, base_y, diagnostics


def write_sequence_container(path, block_size, qp_i, qp_p, i_ec, height, width,
                             frame_count, i_stream, chunk_payloads):
    header = SEQ_HEADER.pack(
        SEQ_MAGIC, SEQ_VERSION, block_size, qp_i, qp_p, i_ec,
        height, width, frame_count, len(chunk_payloads), len(i_stream))
    data = bytearray(header)
    data.extend(i_stream)
    for payload in chunk_payloads:
        data.extend(CHUNK_LENGTH.pack(len(payload)))
        data.extend(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return len(data)


def read_sequence_container(path):
    data = path.read_bytes()
    if len(data) < SEQ_HEADER.size:
        raise ValueError("truncated Stage-1 sequence container")
    fields = SEQ_HEADER.unpack(data[:SEQ_HEADER.size])
    (magic, version, block_size, qp_i, qp_p, i_ec, height, width,
     frame_count, chunk_count, i_len) = fields
    if magic != SEQ_MAGIC or version != SEQ_VERSION:
        raise ValueError("unsupported Stage-1 sequence container")
    pos = SEQ_HEADER.size
    i_stream = data[pos:pos + i_len]
    pos += i_len
    chunks = []
    for _ in range(chunk_count):
        if pos + CHUNK_LENGTH.size > len(data):
            raise ValueError("truncated chunk-length field")
        chunk_len = CHUNK_LENGTH.unpack(data[pos:pos + CHUNK_LENGTH.size])[0]
        pos += CHUNK_LENGTH.size
        chunks.append(data[pos:pos + chunk_len])
        if len(chunks[-1]) != chunk_len:
            raise ValueError("truncated chunk payload")
        pos += chunk_len
    if pos != len(data):
        raise ValueError("sequence container length mismatch")
    return {
        "block_size": block_size, "qp_i": qp_i, "qp_p": qp_p,
        "i_ec": i_ec, "height": height, "width": width,
        "frame_count": frame_count, "i_stream": i_stream, "chunks": chunks,
    }


def encode_sequence(p_net, i_hat, i_stream, i_ec, frames, output_dir, args,
                    drop_target=None):
    initialize_p_state(p_net, i_hat)
    chunk_payloads = []
    search_records = []
    encoded_recon = []
    chunk_dir = output_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    first_index = 1
    chunk_index = 0
    while first_index < len(frames):
        chunk, target_rgb, valid_count = make_chunk(frames, first_index, i_hat.device)
        prepared = prepare_chunk_latents(p_net, chunk, args.qp_p)
        grid_shape = (
            prepared["y"].shape[-2] // args.latent_block_size,
            prepared["y"].shape[-1] // args.latent_block_size,
        )
        if drop_target is None:
            skip = np.zeros(grid_shape, dtype=np.bool_)
            selected_y = encode_y(
                p_net, prepared["y"], prepared["common_params"],
                skip, args.latent_block_size,
                ec_parallel_override=(
                    None if args.y_ec_parallel == 0 else args.y_ec_parallel))
            route = b""
            local_base = selected_y
            search = {
                "oracle_kind": "all-base",
                "mathematical_upper_bound": False,
                "chosen_prefix_blocks": 0,
            }
        else:
            skip, selected_y, route, local_base, search = search_route(
                p_net, prepared, target_rgb, valid_count,
                args.latent_block_size, drop_target,
                args.candidate_pool, args.prefix_sizes,
                None if args.y_ec_parallel == 0 else args.y_ec_parallel)

        chunk_path = chunk_dir / f"chunk_{chunk_index:04d}.d1s"
        write_container(
            chunk_path, args.qp_p, args.height, args.width,
            prepared["y"].shape, prepared["global_stream"], route,
            selected_y.stream, b"", selected_y.ec_parallel,
            all_base=not np.any(skip), z_ec_parallel=prepared["z_ec"])
        payload = chunk_path.read_bytes()
        parsed, decoded_q, x_hat, feature = decode_chunk_bytes(
            p_net, payload, args.latent_block_size, prepared["z_shape"],
            prepared["q_feature"], prepared["q_decoder"])
        if not torch.equal(decoded_q, selected_y.q_dense):
            raise RuntimeError(f"chunk {chunk_index} y container round-trip mismatch")
        recon = [
            rgb_from_recon(frame[:, :, :args.height, :args.width])
            for frame in x_hat[:valid_count]
        ]
        encoded_recon.extend(recon)
        reset = should_reset(chunk_index, args.reset_interval)
        p_net.set_ref_feature(feature, reset)
        search.update({
            "chunk_index": chunk_index,
            "first_frame_one_based": first_index + 1,
            "valid_frames": valid_count,
            "reset_after_chunk": reset,
            "grid_shape": list(grid_shape),
            "skipped_blocks": int(skip.sum()),
            "chunk_bytes": len(payload),
            "container_version": parsed["version"],
            "container_header_bytes": CONTAINER_HEADER.size,
            "global_z_bytes": len(parsed["global"]),
            "route_bytes": len(parsed["route"]),
            "base_y_bytes": len(parsed["base"]),
            "z_ec_parallel": parsed["z_ec"],
            "y_ec_parallel": parsed["y_ec"],
            "local_all_base_y_bytes": len(local_base.stream),
        })
        search_records.append(search)
        chunk_payloads.append(payload)
        first_index += valid_count
        chunk_index += 1

    sequence_path = output_dir / "sequence.d1m"
    total_bytes = write_sequence_container(
        sequence_path, args.latent_block_size, args.qp_i, args.qp_p,
        i_ec, args.height, args.width, len(frames), i_stream, chunk_payloads)
    return sequence_path, total_bytes, search_records, encoded_recon


def decode_sequence(i_net, p_net, sequence_path, original_frames, reset_interval):
    sequence = read_sequence_container(sequence_path)
    i_hat = decode_i_stream(
        i_net, sequence["i_stream"], sequence["i_ec"], sequence["qp_i"],
        sequence["height"], sequence["width"])
    recon = [rgb_from_recon(
        i_hat[:, :, :sequence["height"], :sequence["width"]])]
    initialize_p_state(p_net, i_hat)
    remaining = sequence["frame_count"] - 1
    for chunk_index, payload in enumerate(sequence["chunks"]):
        p_net.apply_feature_adaptor()
        _, q_feature, q_decoder = get_q_params(p_net, sequence["qp_p"])
        parsed = parse_container(payload, sequence["block_size"])
        z_shape = (
            1, g_ch_z, parsed["height"] // 64, parsed["width"] // 64)
        _, _, x_hat, feature = decode_chunk_bytes(
            p_net, payload, sequence["block_size"], z_shape,
            q_feature, q_decoder)
        valid_count = min(g_frame_delay, remaining)
        recon.extend([
            rgb_from_recon(frame[:, :, :sequence["height"], :sequence["width"]])
            for frame in x_hat[:valid_count]
        ])
        remaining -= valid_count
        p_net.set_ref_feature(feature, should_reset(chunk_index, reset_interval))
    if remaining != 0 or len(recon) != len(original_frames):
        raise RuntimeError("decoded frame count mismatch")
    return recon


def stream_breakdown(sequence_path):
    sequence = read_sequence_container(sequence_path)
    parsed = [parse_container(chunk, sequence["block_size"]) for chunk in sequence["chunks"]]
    result = {
        "sequence_header_and_chunk_lengths_bytes": (
            SEQ_HEADER.size + CHUNK_LENGTH.size * len(parsed)),
        "i_payload_bytes": len(sequence["i_stream"]),
        "p_container_header_bytes": CONTAINER_HEADER.size * len(parsed),
        "global_z_bytes": sum(len(chunk["global"]) for chunk in parsed),
        "route_bytes": sum(len(chunk["route"]) for chunk in parsed),
        "base_y_bytes": sum(len(chunk["base"]) for chunk in parsed),
        "residual_bytes": sum(len(chunk["residual"]) for chunk in parsed),
        "chunk_count": len(parsed),
    }
    result["total_bytes"] = sum(
        value for key, value in result.items()
        if key.endswith("_bytes") and key != "total_bytes")
    if result["total_bytes"] != sequence_path.stat().st_size:
        raise RuntimeError("stream breakdown does not equal file size")
    return result


def metric_summary(original, recon, width, height, chunk_records=None):
    per_frame = [psnr(org, rec) for org, rec in zip(original, recon)]
    total_sse = sum(np.sum(
        (org.astype(np.float64) - rec.astype(np.float64)) ** 2)
        for org, rec in zip(original, recon))
    denom = len(original) * 3 * width * height
    aggregate_mse = total_sse / denom
    summary = {
        "mean_frame_psnr": float(np.mean(per_frame)),
        "aggregate_psnr": (
            float("inf") if aggregate_mse == 0
            else 10.0 * math.log10(255.0 * 255.0 / aggregate_mse)),
        "per_frame_psnr": per_frame,
    }
    if chunk_records is not None:
        chunk_metrics = []
        pos = 1
        for record in chunk_records:
            count = record["valid_frames"]
            values = per_frame[pos:pos + count]
            chunk_metrics.append({
                "chunk_index": record["chunk_index"],
                "mean_frame_psnr": float(np.mean(values)),
                "valid_frames": count,
                "reset_after_chunk": record["reset_after_chunk"],
            })
            pos += count
        summary["per_chunk"] = chunk_metrics
    return summary


def compare_trajectories(baseline_metrics, routed_metrics, baseline_recon, routed_recon):
    frame_delta = [
        routed - baseline for routed, baseline in zip(
            routed_metrics["per_frame_psnr"], baseline_metrics["per_frame_psnr"])
    ]
    chunk_delta = []
    for routed, baseline in zip(
            routed_metrics["per_chunk"], baseline_metrics["per_chunk"]):
        chunk_delta.append({
            "chunk_index": routed["chunk_index"],
            "mean_psnr_change_db": (
                routed["mean_frame_psnr"] - baseline["mean_frame_psnr"]),
            "reset_after_chunk": routed["reset_after_chunk"],
        })
    recon_mae = [
        float(np.mean(np.abs(
            routed.astype(np.float32) - baseline.astype(np.float32))))
        for baseline, routed in zip(baseline_recon, routed_recon)
    ]
    return {
        "mean_frame_psnr_change_db": float(np.mean(frame_delta)),
        "aggregate_psnr_change_db": (
            routed_metrics["aggregate_psnr"] - baseline_metrics["aggregate_psnr"]),
        "worst_frame_psnr_change_db": float(np.min(frame_delta)),
        "last_frame_psnr_change_db": float(frame_delta[-1]),
        "per_chunk": chunk_delta,
        "baseline_vs_routed_mean_abs_rgb_level": float(np.mean(recon_mae)),
        "baseline_vs_routed_last_frame_abs_rgb_level": float(recon_mae[-1]),
    }


def save_frames(target, frames):
    target.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames, start=1):
        Image.fromarray(frame.transpose(1, 2, 0)).save(target / f"im{index:05d}.png")


def drop_slug(value):
    return f"drop_{value:.3f}".replace("-", "m").replace(".", "p")


@torch.inference_mode()
def run_sequence(name, source_dir, i_net, p_net, args, device):
    files, frames = load_rgb_frames(
        source_dir, args.width, args.height, args.frame_count, args.max_chunks)
    sequence_root = Path(args.output_dir) / name
    sequence_root.mkdir(parents=True, exist_ok=True)

    i_x = (model_frame(frames[0], device) - 0.5).to(memory_format=torch.channels_last)
    i_encoded = i_net.compress(i_x, args.qp_i, 0, 0)
    i_stream = bytes(i_encoded["bit_stream"])
    i_ec = int(i_encoded["ec_parallel"])
    i_hat = decode_i_stream(
        i_net, i_stream, i_ec, args.qp_i, args.height, args.width)
    if not torch.equal(i_hat, i_encoded["x_hat"]):
        max_error = float(torch.max(torch.abs(i_hat - i_encoded["x_hat"])).item())
        if max_error > 1e-6:
            raise RuntimeError(f"I-frame encoder/decoder reconstruction mismatch: {max_error}")

    baseline_dir = sequence_root / "baseline"
    start = time.perf_counter()
    baseline_path, _, baseline_records, baseline_encode_recon = encode_sequence(
        p_net, i_hat, i_stream, i_ec, frames, baseline_dir, args)
    baseline_seconds = time.perf_counter() - start
    baseline_recon = decode_sequence(
        i_net, p_net, baseline_path, frames, args.reset_interval)
    if any(not np.array_equal(a, b) for a, b in zip(
            [rgb_from_recon(i_hat)] + baseline_encode_recon, baseline_recon)):
        raise RuntimeError("baseline full-container decode differs from encoding trajectory")
    baseline_stream = stream_breakdown(baseline_path)
    baseline_metrics = metric_summary(
        frames, baseline_recon, args.width, args.height, baseline_records)
    baseline_stream["total_bpp"] = (
        baseline_stream["total_bytes"] * 8
        / (len(frames) * args.width * args.height))
    if args.save_recon:
        save_frames(baseline_dir / "recon", baseline_recon)

    policies = []
    for drop_target in args.drop_targets:
        policy_dir = sequence_root / drop_slug(drop_target)
        start = time.perf_counter()
        routed_path, _, records, encode_recon = encode_sequence(
            p_net, i_hat, i_stream, i_ec, frames, policy_dir, args,
            drop_target=drop_target)
        search_seconds = time.perf_counter() - start
        routed_recon = decode_sequence(
            i_net, p_net, routed_path, frames, args.reset_interval)
        if any(not np.array_equal(a, b) for a, b in zip(
                [rgb_from_recon(i_hat)] + encode_recon, routed_recon)):
            raise RuntimeError("routed full-container decode differs from encoding trajectory")
        routed_stream = stream_breakdown(routed_path)
        routed_metrics = metric_summary(
            frames, routed_recon, args.width, args.height, records)
        routed_stream["total_bpp"] = (
            routed_stream["total_bytes"] * 8
            / (len(frames) * args.width * args.height))
        comparison = compare_trajectories(
            baseline_metrics, routed_metrics, baseline_recon, routed_recon)
        comparison.update({
            "net_bytes_saved": baseline_stream["total_bytes"] - routed_stream["total_bytes"],
            "net_rate_change_percent": 100.0 * (
                routed_stream["total_bytes"] / baseline_stream["total_bytes"] - 1.0),
        })
        if args.save_recon:
            save_frames(policy_dir / "recon", routed_recon)
        policy_summary = {
            "drop_target_db": float(drop_target),
            "search_seconds": search_seconds,
            "stream": routed_stream,
            "metrics": routed_metrics,
            "comparison_to_all_base": comparison,
            "chunks": records,
        }
        (policy_dir / "summary.json").write_text(
            json.dumps(policy_summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
        policies.append(policy_summary)

    result = {
        "sequence": name,
        "source_dir": str(source_dir),
        "first_source_file": str(files[0]),
        "last_source_file": str(files[-1]),
        "frame_count": len(frames),
        "p_chunk_count": len(baseline_records),
        "last_chunk_is_padded": ((len(frames) - 1) % g_frame_delay != 0),
        "qp_i": args.qp_i,
        "qp_p": args.qp_p,
        "latent_block_size": args.latent_block_size,
        "routing_unit": [args.latent_block_size, args.latent_block_size, 256, 8],
        "baseline": {
            "encode_seconds": baseline_seconds,
            "stream": baseline_stream,
            "metrics": baseline_metrics,
            "chunks": baseline_records,
        },
        "policies": policies,
        "validation": {
            "i_payload_redecoded": True,
            "z_parallelism_serialized": True,
            "every_chunk_redecoded_during_encoding": True,
            "full_sequence_container_redecoded_from_disk": True,
            "encoding_and_fresh_decode_reconstructions_identical": True,
            "routed_reconstruction_propagated_to_next_chunk": True,
        },
    }
    (sequence_root / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    rows = []
    for policy in policies:
        rows.append({
            "sequence": name,
            "drop_target_db": policy["drop_target_db"],
            "frames": len(frames),
            "chunks": len(baseline_records),
            "baseline_bytes": baseline_stream["total_bytes"],
            "routed_bytes": policy["stream"]["total_bytes"],
            "rate_change_percent": policy["comparison_to_all_base"]["net_rate_change_percent"],
            "baseline_mean_psnr": baseline_metrics["mean_frame_psnr"],
            "routed_mean_psnr": policy["metrics"]["mean_frame_psnr"],
            "mean_psnr_change_db": policy["comparison_to_all_base"]["mean_frame_psnr_change_db"],
            "worst_frame_psnr_change_db": policy["comparison_to_all_base"]["worst_frame_psnr_change_db"],
            "skipped_blocks": sum(chunk["skipped_blocks"] for chunk in policy["chunks"]),
            "route_bytes": policy["stream"]["route_bytes"],
        })
    return result, rows


@torch.inference_mode()
def main():
    args = parse_args()
    if args.height % 64 or args.width % 64:
        raise ValueError("width and height must be divisible by 64")
    if any(value < 0 for value in args.drop_targets):
        raise ValueError("drop targets must be non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    sequences = parse_sequences(args.sequence)
    set_torch_env()
    torch.cuda.set_device(args.cuda_idx)
    device = torch.device(f"cuda:{args.cuda_idx}")
    torch.cuda.set_stream(torch.cuda.Stream(device=device))
    i_net, p_net = load_models(args, device)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    all_results = []
    all_rows = []
    for name, source_dir in sequences:
        result, rows = run_sequence(
            name, source_dir, i_net, p_net, args, device)
        all_results.append(result)
        all_rows.extend(rows)
        print(json.dumps({
            "sequence": name,
            "policies": [
                {
                    "drop_target_db": row["drop_target_db"],
                    "rate_change_percent": row["rate_change_percent"],
                    "mean_psnr_change_db": row["mean_psnr_change_db"],
                    "skipped_blocks": row["skipped_blocks"],
                }
                for row in rows
            ],
        }, ensure_ascii=False), flush=True)

    csv_path = output_root / "aggregate.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    aggregate = {
        "experiment": "stage1_frozen_hts_multichunk_greedy_oracle_v1",
        "scientific_scope": {
            "oracle_is_source_aware": True,
            "oracle_is_deployable": False,
            "mathematical_upper_bound": False,
            "interpretation": (
                "An achievable upper reference for a deployable Router under the tested "
                "greedy-prefix search, not the optimum over all route maps."),
            "required_final_comparison": (
                "Compare routed points against the interpolated all-Base QP RD curve; "
                "same-QP deltas alone do not establish RD improvement."),
        },
        "args": vars(args),
        "sequences": all_results,
    }
    (output_root / "summary.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
