#!/usr/bin/env python3
"""Stage-B decoder-only predictor ladder on fixed routed HT-S bitstreams.

This is a deliberately small, no-training feasibility probe.  It reuses route
maps selected by the Stage-1 mean-fill Oracle, but every tier is re-encoded from
the source while its own routed reconstruction is propagated to later chunks.
The source is used only to encode the video and compute evaluation metrics.

Predictors receive only quantities available after entropy decoding:

* decoded Base residual symbols;
* the route map;
* hyper/temporal ``common_params`` reconstructed from transmitted ``z`` and
  the decoder's propagated state.

The non-parametric kernels predict one 2x2x256 residual vector per skipped
block.  ``spatial-k4`` uses four nearest transmitted blocks.  ``context-k32``
uses 32 transmitted blocks selected by cosine similarity of decoder-known
``common_params``.  A shrink factor is fitted by leave-one-out prediction of
transmitted blocks only, so neither omitted ``y`` nor source pixels leak into
the predictor.  The ``*-additive`` tiers preserve the already decoded Base
latent and add the predicted dequantized correction only at skipped positions.
The unsuffixed tiers retain the earlier whole-prior replay behavior solely for
reproducibility; replay can perturb later Base positions because their symbols
were encoded under the mean-fill spatial-prior trajectory.  These controls are
not claimed to be learned generators.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.analyze_stage1_oracle_rd import interpolate_log_rate  # noqa: E402
from demo.stage1_multichunk_oracle import (  # noqa: E402
    CHUNK_LENGTH,
    decode_i_stream,
    get_q_params,
    initialize_p_state,
    load_models,
    load_rgb_frames,
    make_chunk,
    metric_summary,
    model_frame,
    prepare_chunk_latents,
    read_sequence_container,
    reconstruction_float,
    should_reset,
)
from demo.stage1_token_skipping import (  # noqa: E402
    CONTAINER_HEADER,
    build_route_section,
    decode_y,
    decode_z,
    encode_y,
    expand_keep_mask,
    parse_container,
    rgb_from_recon,
    write_container,
    y_prior_steps,
)
from src.models.video_model_ht import g_ch_z, g_frame_delay  # noqa: E402
from src.utils.common import set_torch_env  # noqa: E402


PRED_MAGIC = b"D2PRED01"
PRED_VERSION = 1
PRED_HEADER = struct.Struct("<8sBBBBBB2xHHHHI")

TIER_IDS = {
    "all-base": 0,
    "mean": 1,
    "spatial-k4": 2,
    "context-k32": 3,
    "spatial-k4-additive": 4,
    "context-k32-additive": 5,
}
ID_TO_TIER = {value: key for key, value in TIER_IDS.items()}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fixed-route, decoder-only predictor ladder with real HT-S streams.")
    parser.add_argument(
        "--sequence", action="append", default=[], metavar="NAME=PATH",
        help="Sequence name and PNG directory. Repeat for multiple sequences.")
    parser.add_argument(
        "--route-root",
        default="output/stage1_multichunk_oracle_qp32_full_candidates",
        help="Stage-1 root providing fixed route maps in NAME/drop_*/sequence.d1m.")
    parser.add_argument("--route-policy", default="drop_0p100")
    parser.add_argument(
        "--output-dir", default="output/stage_b_decoder_predictor_probe_qp32_additive")
    parser.add_argument("--model-path-i", default="checkpoints/cvpr2026_image.pth.tar")
    parser.add_argument("--model-path-p", default="checkpoints/cvpr2026_video_hts.pth.tar")
    parser.add_argument("--qp-i", type=int, default=32)
    parser.add_argument("--qp-p", type=int, default=32)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--frame-count", type=int, default=0)
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--latent-block-size", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument(
        "--tiers", nargs="+", choices=tuple(TIER_IDS),
        default=("all-base", "mean", "spatial-k4-additive",
                 "context-k32-additive"),
        help="Predictor tiers to evaluate; unsuffixed predictors are legacy controls.")
    parser.add_argument("--skip-thres", type=float, default=0.0)
    parser.add_argument("--reset-interval", type=int, default=32)
    parser.add_argument("--decode-repeats", type=int, default=3)
    parser.add_argument("--cuda-idx", type=int, default=0)
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument(
        "--baseline", action="append", default=[
            "30=output/stage1_allbase_qp30_full",
            "31=output/stage1_allbase_qp31_full",
            "32=output/stage1_multichunk_oracle_qp32_full_candidates",
        ], metavar="QP=ROOT", help="All-Base roots used for matched-RD analysis.")
    return parser.parse_args()


def parse_sequences(items):
    if not items:
        items = [
            "jockey=data/test_sequences/PNG/jockey",
            "shake=data/test_sequences/PNG/shake",
            "sky=data/test_sequences/PNG/sky",
            "uvg=data/test_sequences/PNG/uvg",
        ]
    result = []
    seen = set()
    for item in items:
        if "=" not in item:
            raise ValueError(f"sequence must be NAME=PATH, got {item!r}")
        name, path_text = item.split("=", 1)
        name = name.strip()
        path = Path(path_text)
        if not name or name in seen or not path.is_dir():
            raise ValueError(f"invalid sequence {item!r}")
        seen.add(name)
        result.append((name, path))
    return result


def parse_baselines(items):
    result = {}
    for item in items:
        qp_text, root_text = item.split("=", 1)
        result[int(qp_text)] = Path(root_text)
    return result


def tensor_to_blocks(tensor, block_size):
    if tensor.ndim != 4 or tensor.shape[0] != 1:
        raise ValueError("block conversion expects an NCHW tensor with batch size one")
    _, channels, height, width = tensor.shape
    if height % block_size or width % block_size:
        raise ValueError("latent dimensions must be divisible by the route block size")
    grid_h, grid_w = height // block_size, width // block_size
    return tensor[0].reshape(
        channels, grid_h, block_size, grid_w, block_size
    ).permute(1, 3, 0, 2, 4).reshape(
        grid_h * grid_w, channels * block_size * block_size)


def blocks_to_tensor(blocks, shape, block_size):
    batch, channels, height, width = shape
    if batch != 1:
        raise ValueError("block conversion expects batch size one")
    grid_h, grid_w = height // block_size, width // block_size
    return blocks.reshape(
        grid_h, grid_w, channels, block_size, block_size
    ).permute(2, 0, 3, 1, 4).reshape(shape)


def block_descriptors(common_params, block_size):
    _, channels, height, width = common_params.shape
    grid_h, grid_w = height // block_size, width // block_size
    descriptors = common_params.float().reshape(
        1, channels, grid_h, block_size, grid_w, block_size
    ).mean(dim=(3, 5))[0].permute(1, 2, 0).reshape(grid_h * grid_w, channels)
    return F.normalize(descriptors, dim=1, eps=1e-8)


def spatial_neighbors(query_ids, base_ids, grid_width, neighbor_count, exclude_self):
    query_ids = np.asarray(query_ids, dtype=np.int64)
    base_ids = np.asarray(base_ids, dtype=np.int64)
    q_row, q_col = np.divmod(query_ids, grid_width)
    b_row, b_col = np.divmod(base_ids, grid_width)
    distance_sq = (
        (q_row[:, None] - b_row[None, :]) ** 2
        + (q_col[:, None] - b_col[None, :]) ** 2)
    if exclude_self:
        distance_sq[query_ids[:, None] == base_ids[None, :]] = np.iinfo(np.int64).max
    count = min(neighbor_count, len(base_ids) - int(exclude_self))
    if count <= 0:
        raise ValueError("not enough transmitted blocks to form a predictor neighborhood")
    order = np.argsort(distance_sq, axis=1, kind="stable")[:, :count]
    selected = base_ids[order]
    selected_distance = np.take_along_axis(distance_sq, order, axis=1)
    weights = 1.0 / np.sqrt(selected_distance.astype(np.float32) + 1e-3)
    weights /= np.sum(weights, axis=1, keepdims=True)
    return selected, weights


def context_neighbors(query_ids, base_ids, descriptors, neighbor_count, exclude_self):
    query = torch.as_tensor(query_ids, device=descriptors.device, dtype=torch.long)
    base = torch.as_tensor(base_ids, device=descriptors.device, dtype=torch.long)
    similarity = descriptors[query] @ descriptors[base].T
    if exclude_self:
        similarity = similarity.masked_fill(query[:, None] == base[None, :], -torch.inf)
    count = min(neighbor_count, len(base_ids) - int(exclude_self))
    if count <= 0:
        raise ValueError("not enough transmitted blocks to form a predictor neighborhood")
    values, positions = similarity.topk(count, dim=1)
    selected = base[positions]
    weights = torch.softmax(values * 10.0, dim=1)
    return selected, weights


def gather_prediction(base_vectors, neighbor_ids, weights):
    ids = torch.as_tensor(neighbor_ids, device=base_vectors.device, dtype=torch.long)
    weight_tensor = torch.as_tensor(
        weights, device=base_vectors.device, dtype=torch.float32)
    return (base_vectors[ids].float() * weight_tensor[:, :, None]).sum(dim=1)


def predict_residual_blocks(decoded_q, common_params, skip_blocks, block_size, tier):
    residual_blocks = tensor_to_blocks(decoded_q, block_size)
    skip_flat = skip_blocks.reshape(-1)
    missing_ids = np.flatnonzero(skip_flat)
    base_ids = np.flatnonzero(~skip_flat)
    if not len(missing_ids):
        return decoded_q, {
            "tier": tier, "skipped_blocks": 0, "shrink": 0.0,
            "predicted_coefficients": 0, "neighbor_vector_reads": 0,
            "calibration_neighbor_vector_reads": 0, "descriptor_pairs": 0,
        }

    kernel_tier = tier.removesuffix("-additive")
    if kernel_tier == "spatial-k4":
        neighbor_count = 4
        missing_neighbors, missing_weights = spatial_neighbors(
            missing_ids, base_ids, skip_blocks.shape[1], neighbor_count, False)
        calibration_neighbors, calibration_weights = spatial_neighbors(
            base_ids, base_ids, skip_blocks.shape[1], neighbor_count, True)
        missing_prediction = gather_prediction(
            residual_blocks, missing_neighbors, missing_weights)
        calibration_prediction = gather_prediction(
            residual_blocks, calibration_neighbors, calibration_weights)
        descriptor_pairs = 0
    elif kernel_tier == "context-k32":
        neighbor_count = 32
        descriptors = block_descriptors(common_params, block_size)
        missing_neighbors, missing_weights = context_neighbors(
            missing_ids, base_ids, descriptors, neighbor_count, False)
        calibration_neighbors, calibration_weights = context_neighbors(
            base_ids, base_ids, descriptors, neighbor_count, True)
        missing_prediction = gather_prediction(
            residual_blocks, missing_neighbors, missing_weights)
        calibration_prediction = gather_prediction(
            residual_blocks, calibration_neighbors, calibration_weights)
        descriptor_pairs = int((len(missing_ids) + len(base_ids)) * len(base_ids))
    else:
        raise ValueError(f"tier {tier!r} is not a residual-block predictor")

    calibration_target = residual_blocks[
        torch.as_tensor(base_ids, device=decoded_q.device, dtype=torch.long)
    ].float()
    denominator = torch.sum(calibration_prediction.square())
    if float(denominator) <= 1e-12:
        shrink = torch.zeros((), device=decoded_q.device, dtype=torch.float32)
    else:
        shrink = torch.clamp(
            torch.sum(calibration_prediction * calibration_target) / denominator,
            min=0.0, max=1.0)

    output_blocks = residual_blocks.clone()
    output_blocks[
        torch.as_tensor(missing_ids, device=decoded_q.device, dtype=torch.long)
    ] = (missing_prediction * shrink).to(decoded_q.dtype)
    vector_width = residual_blocks.shape[1]
    missing_k = missing_neighbors.shape[1]
    calibration_k = calibration_neighbors.shape[1]
    return blocks_to_tensor(output_blocks, decoded_q.shape, block_size), {
        "tier": tier,
        "skipped_blocks": int(len(missing_ids)),
        "base_blocks": int(len(base_ids)),
        "neighbors_per_missing_block": int(missing_k),
        "neighbors_per_calibration_block": int(calibration_k),
        "shrink": float(shrink),
        "predicted_coefficients": int(len(missing_ids) * vector_width),
        "neighbor_vector_reads": int(len(missing_ids) * missing_k),
        "calibration_neighbor_vector_reads": int(len(base_ids) * calibration_k),
        "descriptor_pairs": descriptor_pairs,
        "vector_width": int(vector_width),
        "uses_only_decoder_available_inputs": True,
    }


def apply_predictor(net, decoded_q, mean_y, common_params, skip_blocks,
                    block_size, tier, measure):
    if tier in ("all-base", "mean") or not np.any(skip_blocks):
        return mean_y, {
            "tier": tier,
            "wall_seconds": 0.0,
            "skipped_blocks": int(np.sum(skip_blocks)),
            "predicted_coefficients": 0,
            "uses_only_decoder_available_inputs": True,
        }
    if measure:
        torch.cuda.synchronize(decoded_q.device)
        start = time.perf_counter()
    predicted_q, profile = predict_residual_blocks(
        decoded_q, common_params, skip_blocks, block_size, tier)
    keep = expand_keep_mask(skip_blocks, block_size, decoded_q.shape, decoded_q.device)
    if tier.endswith("-additive"):
        _, q_dec, _, _ = net.separate_prior_video(common_params)
        missing = ~keep
        predicted_y = mean_y + predicted_q * missing * q_dec
        profile["correction_interface"] = "post_prior_skip_only"
        profile["base_latent_modified"] = False
        profile["corrected_latent_coefficients"] = int(missing.sum().item())
    else:
        _, _, predicted_y, _ = y_prior_steps(
            net, common_params, keep, decoded_q=predicted_q)
        profile["correction_interface"] = "legacy_whole_prior_replay"
        profile["base_latent_modified"] = True
    if measure:
        torch.cuda.synchronize(decoded_q.device)
        profile["wall_seconds"] = time.perf_counter() - start
    else:
        profile["wall_seconds"] = 0.0
    return predicted_y, profile


def write_predictor_sequence(path, predictor_id, block_size, qp_i, qp_p, i_ec,
                             height, width, frame_count, i_stream, chunks):
    header = PRED_HEADER.pack(
        PRED_MAGIC, PRED_VERSION, predictor_id, block_size, qp_i, qp_p, i_ec,
        height, width, frame_count, len(chunks), len(i_stream))
    data = bytearray(header)
    data.extend(i_stream)
    for chunk in chunks:
        data.extend(CHUNK_LENGTH.pack(len(chunk)))
        data.extend(chunk)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return len(data)


def read_predictor_sequence(path):
    data = path.read_bytes()
    if len(data) < PRED_HEADER.size:
        raise ValueError("truncated predictor sequence")
    fields = PRED_HEADER.unpack(data[:PRED_HEADER.size])
    (magic, version, predictor_id, block_size, qp_i, qp_p, i_ec,
     height, width, frame_count, chunk_count, i_len) = fields
    if magic != PRED_MAGIC or version != PRED_VERSION or predictor_id not in ID_TO_TIER:
        raise ValueError("unsupported predictor sequence")
    pos = PRED_HEADER.size
    i_stream = data[pos:pos + i_len]
    pos += i_len
    chunks = []
    for _ in range(chunk_count):
        if pos + CHUNK_LENGTH.size > len(data):
            raise ValueError("truncated predictor chunk length")
        chunk_len = CHUNK_LENGTH.unpack(data[pos:pos + CHUNK_LENGTH.size])[0]
        pos += CHUNK_LENGTH.size
        chunks.append(data[pos:pos + chunk_len])
        if len(chunks[-1]) != chunk_len:
            raise ValueError("truncated predictor chunk")
        pos += chunk_len
    if pos != len(data):
        raise ValueError("predictor sequence length mismatch")
    return {
        "tier": ID_TO_TIER[predictor_id], "predictor_id": predictor_id,
        "block_size": block_size, "qp_i": qp_i, "qp_p": qp_p,
        "i_ec": i_ec, "height": height, "width": width,
        "frame_count": frame_count, "i_stream": i_stream, "chunks": chunks,
    }


def predictor_stream_breakdown(path):
    sequence = read_predictor_sequence(path)
    parsed = [parse_container(chunk, sequence["block_size"])
              for chunk in sequence["chunks"]]
    result = {
        "sequence_header_and_chunk_lengths_bytes": (
            PRED_HEADER.size + CHUNK_LENGTH.size * len(parsed)),
        "predictor_mode_signal_bytes": 1,
        "i_payload_bytes": len(sequence["i_stream"]),
        "p_container_header_bytes": CONTAINER_HEADER.size * len(parsed),
        "global_z_bytes": sum(len(chunk["global"]) for chunk in parsed),
        "route_bytes": sum(len(chunk["route"]) for chunk in parsed),
        "base_y_bytes": sum(len(chunk["base"]) for chunk in parsed),
        "residual_bytes": sum(len(chunk["residual"]) for chunk in parsed),
        "chunk_count": len(parsed),
    }
    result["total_bytes"] = (
        result["sequence_header_and_chunk_lengths_bytes"]
        + result["i_payload_bytes"] + result["p_container_header_bytes"]
        + result["global_z_bytes"] + result["route_bytes"]
        + result["base_y_bytes"] + result["residual_bytes"])
    if result["total_bytes"] != path.stat().st_size:
        raise RuntimeError("predictor stream breakdown does not equal file size")
    return result


def decode_chunk_with_predictor(p_net, payload, block_size, z_shape,
                                q_feature, q_decoder, tier, measure):
    parsed = parse_container(payload, block_size)
    decoded_z = decode_z(
        parsed["global"], parsed["qp"], z_shape,
        p_net.bit_estimator_z.get_cdf_info(), parsed["z_ec"])
    decoded_z = decoded_z.to(
        device=p_net.memory.device, dtype=p_net.memory.dtype,
        memory_format=torch.channels_last)
    common_params = p_net.res_prior_param_decoder(decoded_z, p_net.memory, q_feature)
    decoded_q, mean_y = decode_y(
        p_net, parsed["base"], parsed["y_ec"], common_params,
        parsed["skip_blocks"], block_size)
    predicted_y, profile = apply_predictor(
        p_net, decoded_q, mean_y, common_params, parsed["skip_blocks"],
        block_size, tier, measure)
    x_hat, feature = p_net.get_recon_and_feature(predicted_y, p_net.ctx, q_decoder)
    return parsed, decoded_q, x_hat, feature, profile


def load_fixed_routes(path, expected_block_size, frame_count, qp_i, qp_p):
    source = read_sequence_container(path)
    if source["block_size"] != expected_block_size:
        raise ValueError("route source block size mismatch")
    if source["qp_i"] != qp_i or source["qp_p"] != qp_p:
        raise ValueError("route source QP mismatch")
    if source["frame_count"] < frame_count:
        raise ValueError("route source has fewer frames than the requested probe")
    chunk_count = math.ceil((frame_count - 1) / g_frame_delay)
    routes = [
        parse_container(payload, expected_block_size)["skip_blocks"]
        for payload in source["chunks"][:chunk_count]
    ]
    if len(routes) != chunk_count:
        raise ValueError("route source has too few chunks")
    return routes


def encode_policy(p_net, i_hat, i_stream, i_ec, frames, route_masks,
                  output_dir, args, tier):
    initialize_p_state(p_net, i_hat)
    chunks = []
    encoder_recon = []
    records = []
    chunk_dir = output_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    first_index = 1
    chunk_index = 0
    while first_index < len(frames):
        chunk, _, valid_count = make_chunk(frames, first_index, i_hat.device)
        prepared = prepare_chunk_latents(p_net, chunk, args.qp_p)
        grid_shape = (
            prepared["y"].shape[-2] // args.latent_block_size,
            prepared["y"].shape[-1] // args.latent_block_size)
        skip = (np.zeros(grid_shape, dtype=np.bool_) if tier == "all-base"
                else route_masks[chunk_index].copy())
        encoded_y = encode_y(
            p_net, prepared["y"], prepared["common_params"], skip,
            args.latent_block_size)
        route = b"" if not np.any(skip) else build_route_section(
            skip, args.latent_block_size)
        chunk_path = chunk_dir / f"chunk_{chunk_index:04d}.d1s"
        write_container(
            chunk_path, args.qp_p, args.height, args.width,
            prepared["y"].shape, prepared["global_stream"], route,
            encoded_y.stream, b"", encoded_y.ec_parallel,
            all_base=not np.any(skip), z_ec_parallel=prepared["z_ec"])
        payload = chunk_path.read_bytes()
        parsed, decoded_q, x_hat, feature, profile = decode_chunk_with_predictor(
            p_net, payload, args.latent_block_size, prepared["z_shape"],
            prepared["q_feature"], prepared["q_decoder"], tier, False)
        if not torch.equal(decoded_q, encoded_y.q_dense):
            raise RuntimeError(f"{tier} chunk {chunk_index} y rANS mismatch")
        encoder_recon.extend([
            rgb_from_recon(frame[:, :, :args.height, :args.width])
            for frame in x_hat[:valid_count]
        ])
        reset = should_reset(chunk_index, args.reset_interval)
        p_net.set_ref_feature(feature, reset)
        records.append({
            "chunk_index": chunk_index,
            "valid_frames": valid_count,
            "reset_after_chunk": reset,
            "skipped_blocks": int(skip.sum()),
            "chunk_bytes": len(payload),
            "global_z_bytes": len(parsed["global"]),
            "route_bytes": len(parsed["route"]),
            "base_y_bytes": len(parsed["base"]),
            "predictor_profile": profile,
        })
        chunks.append(payload)
        first_index += valid_count
        chunk_index += 1

    sequence_path = output_dir / "sequence.d2m"
    write_predictor_sequence(
        sequence_path, TIER_IDS[tier], args.latent_block_size,
        args.qp_i, args.qp_p, i_ec, args.height, args.width,
        len(frames), i_stream, chunks)
    return sequence_path, encoder_recon, records


def decode_policy_once(i_net, p_net, path, reset_interval, measure):
    sequence = read_predictor_sequence(path)
    device = next(p_net.parameters()).device
    if measure:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
    i_hat = decode_i_stream(
        i_net, sequence["i_stream"], sequence["i_ec"], sequence["qp_i"],
        sequence["height"], sequence["width"])
    recon = [rgb_from_recon(
        i_hat[:, :, :sequence["height"], :sequence["width"]])]
    initialize_p_state(p_net, i_hat)
    remaining = sequence["frame_count"] - 1
    profiles = []
    for chunk_index, payload in enumerate(sequence["chunks"]):
        p_net.apply_feature_adaptor()
        _, q_feature, q_decoder = get_q_params(p_net, sequence["qp_p"])
        parsed = parse_container(payload, sequence["block_size"])
        z_shape = (
            1, g_ch_z, parsed["height"] // 64, parsed["width"] // 64)
        _, _, x_hat, feature, profile = decode_chunk_with_predictor(
            p_net, payload, sequence["block_size"], z_shape,
            q_feature, q_decoder, sequence["tier"], measure)
        valid_count = min(g_frame_delay, remaining)
        recon.extend([
            rgb_from_recon(frame[:, :, :sequence["height"], :sequence["width"]])
            for frame in x_hat[:valid_count]
        ])
        remaining -= valid_count
        profiles.append(profile)
        p_net.set_ref_feature(feature, should_reset(chunk_index, reset_interval))
    if remaining != 0:
        raise RuntimeError("predictor sequence frame count mismatch")
    if measure:
        torch.cuda.synchronize(device)
        total_seconds = time.perf_counter() - start
        peak_memory = int(torch.cuda.max_memory_allocated(device))
    else:
        total_seconds = 0.0
        peak_memory = 0
    return recon, profiles, total_seconds, peak_memory


def temporal_delta_mae(original, recon):
    if len(original) < 2:
        return 0.0
    values = []
    for index in range(1, len(original)):
        source_delta = original[index].astype(np.float32) - original[index - 1].astype(np.float32)
        recon_delta = recon[index].astype(np.float32) - recon[index - 1].astype(np.float32)
        values.append(float(np.mean(np.abs(source_delta - recon_delta))))
    return float(np.mean(values))


def lpips_alex(original, recon, device, batch_size=8):
    import lpips
    model = lpips.LPIPS(net="alex", verbose=False).eval().to(device)
    values = []
    for start in range(0, len(original), batch_size):
        source = torch.from_numpy(
            np.stack(original[start:start + batch_size])).to(device=device).float()
        decoded = torch.from_numpy(
            np.stack(recon[start:start + batch_size])).to(device=device).float()
        source = source / 127.5 - 1.0
        decoded = decoded / 127.5 - 1.0
        values.extend(model(source, decoded).reshape(-1).detach().cpu().tolist())
    result = float(np.mean(values))
    del model
    torch.cuda.empty_cache()
    return result


def load_baseline_points(baselines, name, metric="mean_frame_psnr"):
    points = []
    for qp, root in baselines.items():
        data = json.loads((root / name / "summary.json").read_text(encoding="utf-8"))
        points.append((
            qp,
            float(data["baseline"]["metrics"][metric]),
            float(data["baseline"]["stream"]["total_bpp"]),
        ))
    return points


def matched_rd(points, quality, routed_bpp):
    interpolation = interpolate_log_rate(points, quality)
    if interpolation is None:
        return {"status": "outside_baseline_quality_range"}
    matched_bpp, lower_qp, upper_qp, weight = interpolation
    return {
        "status": "ok",
        "matched_all_base_bpp": matched_bpp,
        "matched_rate_change_percent": 100.0 * (routed_bpp / matched_bpp - 1.0),
        "lower_qp": lower_qp,
        "upper_qp": upper_qp,
        "interpolation_weight": weight,
    }


def summarize_decode(profiles_by_repeat, seconds, peak_memory, frame_count):
    chunk_counts = [len(profiles) for profiles in profiles_by_repeat]
    predictor_seconds = [
        sum(profile.get("wall_seconds", 0.0) for profile in profiles)
        for profiles in profiles_by_repeat
    ]
    nonempty = [
        profile for profiles in profiles_by_repeat for profile in profiles
        if profile.get("skipped_blocks", 0) > 0
    ]
    shrink = [profile["shrink"] for profile in nonempty if "shrink" in profile]
    exemplar = nonempty[0] if nonempty else {}
    return {
        "decode_repeats": len(seconds),
        "decode_wall_seconds_median": float(np.median(seconds)),
        "decode_wall_seconds_all": seconds,
        "decode_throughput_fps": frame_count / float(np.median(seconds)),
        "predictor_wall_seconds_median": float(np.median(predictor_seconds)),
        "predictor_ms_per_p_chunk": (
            1000.0 * float(np.median(predictor_seconds)) / chunk_counts[0]
            if chunk_counts and chunk_counts[0] else 0.0),
        "peak_cuda_allocated_bytes_max": int(max(peak_memory)),
        "mean_decoder_fitted_shrink": float(np.mean(shrink)) if shrink else 0.0,
        "min_decoder_fitted_shrink": float(np.min(shrink)) if shrink else 0.0,
        "max_decoder_fitted_shrink": float(np.max(shrink)) if shrink else 0.0,
        "per_chunk_activation_example": {
            key: exemplar[key] for key in (
                "skipped_blocks", "base_blocks", "neighbors_per_missing_block",
                "neighbors_per_calibration_block", "predicted_coefficients",
                "neighbor_vector_reads", "calibration_neighbor_vector_reads",
                "descriptor_pairs", "vector_width") if key in exemplar
        },
    }


def run_sequence(name, source_dir, i_net, p_net, args, device, baselines):
    files, frames = load_rgb_frames(
        source_dir, args.width, args.height, args.frame_count, args.max_chunks)
    route_path = Path(args.route_root) / name / args.route_policy / "sequence.d1m"
    routes = load_fixed_routes(
        route_path, args.latent_block_size, len(frames), args.qp_i, args.qp_p)

    i_x = (model_frame(frames[0], device) - 0.5).to(memory_format=torch.channels_last)
    i_encoded = i_net.compress(i_x, args.qp_i, 0, 0)
    i_stream = bytes(i_encoded["bit_stream"])
    i_ec = int(i_encoded["ec_parallel"])
    i_hat = decode_i_stream(
        i_net, i_stream, i_ec, args.qp_i, args.height, args.width)

    sequence_root = Path(args.output_dir) / name
    results = []
    recon_by_tier = {}
    for tier in args.tiers:
        tier_dir = sequence_root / tier.replace("-", "_")
        path, encoder_recon, records = encode_policy(
            p_net, i_hat, i_stream, i_ec, frames, routes, tier_dir, args, tier)
        repeated_profiles = []
        repeated_seconds = []
        repeated_peak = []
        decoded_reference = None
        for _ in range(args.decode_repeats):
            decoded, profiles, seconds, peak = decode_policy_once(
                i_net, p_net, path, args.reset_interval, True)
            if decoded_reference is None:
                decoded_reference = decoded
            elif any(not np.array_equal(a, b) for a, b in zip(decoded_reference, decoded)):
                raise RuntimeError(f"{name}/{tier} repeated decodes are not identical")
            repeated_profiles.append(profiles)
            repeated_seconds.append(seconds)
            repeated_peak.append(peak)
        encoder_full = [rgb_from_recon(i_hat)] + encoder_recon
        if any(not np.array_equal(a, b) for a, b in zip(encoder_full, decoded_reference)):
            raise RuntimeError(f"{name}/{tier} fresh decode differs from encoding trajectory")
        stream = predictor_stream_breakdown(path)
        stream["total_bpp"] = (
            stream["total_bytes"] * 8 / (len(frames) * args.width * args.height))
        metrics = metric_summary(
            frames, decoded_reference, args.width, args.height, records)
        metrics["temporal_delta_mae_rgb_level"] = temporal_delta_mae(
            frames, decoded_reference)
        compute = summarize_decode(
            repeated_profiles, repeated_seconds, repeated_peak, len(frames))
        result = {
            "tier": tier,
            "predictor_id_serialized": TIER_IDS[tier],
            "stream": stream,
            "metrics": metrics,
            "compute": compute,
            "chunks": records,
            "matched_rd_mean_frame_psnr": matched_rd(
                load_baseline_points(baselines, name),
                metrics["mean_frame_psnr"], stream["total_bpp"]),
            "validation": {
                "fixed_route_source": str(route_path),
                "source_used_by_predictor": False,
                "omitted_y_used_by_predictor": False,
                "route_and_predictor_mode_serialized": True,
                "full_sequence_fresh_decode": True,
                "repeated_decodes_identical": True,
                "encoding_and_fresh_decode_identical": True,
                "routed_reconstruction_propagated": True,
            },
        }
        results.append(result)
        recon_by_tier[tier] = decoded_reference

    if args.lpips:
        for result in results:
            result["metrics"]["lpips_alex"] = lpips_alex(
                frames, recon_by_tier[result["tier"]], device)

    mean_result = next((item for item in results if item["tier"] == "mean"), None)
    for result in results:
        if mean_result is None:
            result["comparison_to_mean"] = None
            continue
        result["comparison_to_mean"] = {
            "bytes_change": (
                result["stream"]["total_bytes"] - mean_result["stream"]["total_bytes"]),
            "mean_psnr_change_db": (
                result["metrics"]["mean_frame_psnr"]
                - mean_result["metrics"]["mean_frame_psnr"]),
            "lpips_change": (
                result["metrics"].get("lpips_alex", float("nan"))
                - mean_result["metrics"].get("lpips_alex", float("nan"))),
            "temporal_delta_mae_change": (
                result["metrics"]["temporal_delta_mae_rgb_level"]
                - mean_result["metrics"]["temporal_delta_mae_rgb_level"]),
            "predictor_ms_per_p_chunk_change": (
                result["compute"]["predictor_ms_per_p_chunk"]
                - mean_result["compute"]["predictor_ms_per_p_chunk"]),
        }

    summary = {
        "sequence": name,
        "source_dir": str(source_dir),
        "first_source_file": str(files[0]),
        "last_source_file": str(files[-1]),
        "frame_count": len(frames),
        "p_chunk_count": len(routes),
        "qp_i": args.qp_i,
        "qp_p": args.qp_p,
        "routing_unit": [args.latent_block_size, args.latent_block_size, 256, 8],
        "fixed_route_skipped_blocks": int(sum(route.sum() for route in routes)),
        "tiers": results,
    }
    sequence_root.mkdir(parents=True, exist_ok=True)
    (sequence_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def is_dominated(point, candidates):
    for other in candidates:
        no_worse = (
            other["bpp"] <= point["bpp"]
            and other["psnr"] >= point["psnr"]
            and other["predictor_ms"] <= point["predictor_ms"])
        strictly_better = (
            other["bpp"] < point["bpp"]
            or other["psnr"] > point["psnr"]
            or other["predictor_ms"] < point["predictor_ms"])
        if no_worse and strictly_better:
            return True, other["label"]
    return False, None


def make_rows(results, baselines):
    rows = []
    for sequence in results:
        name = sequence["sequence"]
        candidates = []
        for qp, psnr_value, bpp in load_baseline_points(baselines, name):
            candidates.append({
                "label": f"all-base-qp{qp}", "bpp": bpp,
                "psnr": psnr_value, "predictor_ms": 0.0})
        for tier in sequence["tiers"]:
            candidates.append({
                "label": tier["tier"], "bpp": tier["stream"]["total_bpp"],
                "psnr": tier["metrics"]["mean_frame_psnr"],
                "predictor_ms": tier["compute"]["predictor_ms_per_p_chunk"],
            })
        for tier in sequence["tiers"]:
            point = next(item for item in candidates if item["label"] == tier["tier"])
            dominated, dominator = is_dominated(point, candidates)
            matched = tier["matched_rd_mean_frame_psnr"]
            comparison = tier["comparison_to_mean"] or {}
            rows.append({
                "sequence": name,
                "tier": tier["tier"],
                "total_bytes": tier["stream"]["total_bytes"],
                "total_bpp": tier["stream"]["total_bpp"],
                "container_bytes": (
                    tier["stream"]["sequence_header_and_chunk_lengths_bytes"]
                    + tier["stream"]["p_container_header_bytes"]),
                "i_payload_bytes": tier["stream"]["i_payload_bytes"],
                "global_z_bytes": tier["stream"]["global_z_bytes"],
                "base_y_bytes": tier["stream"]["base_y_bytes"],
                "route_bytes": tier["stream"]["route_bytes"],
                "residual_bytes": tier["stream"]["residual_bytes"],
                "mean_frame_psnr": tier["metrics"]["mean_frame_psnr"],
                "aggregate_psnr": tier["metrics"]["aggregate_psnr"],
                "lpips_alex": tier["metrics"].get("lpips_alex", ""),
                "temporal_delta_mae": tier["metrics"]["temporal_delta_mae_rgb_level"],
                "decode_wall_seconds": tier["compute"]["decode_wall_seconds_median"],
                "decode_throughput_fps": tier["compute"]["decode_throughput_fps"],
                "predictor_ms_per_p_chunk": tier["compute"]["predictor_ms_per_p_chunk"],
                "peak_cuda_allocated_bytes": tier["compute"]["peak_cuda_allocated_bytes_max"],
                "mean_fitted_shrink": tier["compute"]["mean_decoder_fitted_shrink"],
                "psnr_change_vs_mean_db": comparison.get("mean_psnr_change_db", ""),
                "lpips_change_vs_mean": comparison.get("lpips_change", ""),
                "temporal_change_vs_mean": comparison.get("temporal_delta_mae_change", ""),
                "matched_rate_change_percent": matched.get(
                    "matched_rate_change_percent", ""),
                "rdc_dominated_in_tested_points": dominated,
                "dominated_by": dominator or "",
            })
    return rows


def aggregate_tiers(results, width, height):
    tier_names = [item["tier"] for item in results[0]["tiers"]]
    by_sequence = {
        sequence["sequence"]: {item["tier"]: item for item in sequence["tiers"]}
        for sequence in results
    }
    total_frames = sum(sequence["frame_count"] for sequence in results)
    total_pixels = total_frames * width * height
    aggregates = []
    for tier_name in tier_names:
        items = [by_sequence[sequence["sequence"]][tier_name] for sequence in results]
        total_bytes = sum(item["stream"]["total_bytes"] for item in items)
        weighted_psnr = sum(
            item["metrics"]["mean_frame_psnr"] * sequence["frame_count"]
            for item, sequence in zip(items, results)) / total_frames
        weighted_lpips = sum(
            item["metrics"].get("lpips_alex", 0.0) * sequence["frame_count"]
            for item, sequence in zip(items, results)) / total_frames
        temporal_weight = sum(max(0, sequence["frame_count"] - 1) for sequence in results)
        weighted_temporal = sum(
            item["metrics"]["temporal_delta_mae_rgb_level"]
            * max(0, sequence["frame_count"] - 1)
            for item, sequence in zip(items, results)) / temporal_weight
        total_chunks = sum(sequence["p_chunk_count"] for sequence in results)
        predictor_ms = sum(
            item["compute"]["predictor_ms_per_p_chunk"] * sequence["p_chunk_count"]
            for item, sequence in zip(items, results)) / total_chunks
        aggregates.append({
            "tier": tier_name,
            "total_bytes": total_bytes,
            "total_bpp": total_bytes * 8 / total_pixels,
            "frame_weighted_mean_psnr": weighted_psnr,
            "frame_weighted_lpips_alex": weighted_lpips,
            "lpips_reported": all(
                "lpips_alex" in item["metrics"] for item in items),
            "transition_weighted_temporal_delta_mae": weighted_temporal,
            "predictor_ms_per_p_chunk_weighted": predictor_ms,
            "sequential_decode_throughput_fps": (
                total_frames / sum(
                    item["compute"]["decode_wall_seconds_median"] for item in items)),
            "peak_cuda_allocated_bytes_max": max(
                item["compute"]["peak_cuda_allocated_bytes_max"] for item in items),
            "psnr_improved_vs_mean_sequence_count": sum(
                (item["comparison_to_mean"] or {}).get("mean_psnr_change_db", 0.0) > 0.0
                for item in items),
            "lpips_and_temporal_both_improved_vs_mean_sequence_count": sum(
                (item["comparison_to_mean"] or {}).get("lpips_change", 0.0) < 0.0
                and (item["comparison_to_mean"] or {}).get(
                    "temporal_delta_mae_change", 0.0) < 0.0
                for item in items),
        })
    nontrivial = [
        item for item in aggregates if item["tier"] not in ("all-base", "mean")]
    mean_point = next(item for item in aggregates if item["tier"] == "mean")
    passing = []
    for item in nontrivial:
        psnr_gain = (
            item["frame_weighted_mean_psnr"]
            - mean_point["frame_weighted_mean_psnr"])
        lpips_ratio = (
            item["frame_weighted_lpips_alex"]
            / max(mean_point["frame_weighted_lpips_alex"], 1e-12))
        temporal_ratio = (
            item["transition_weighted_temporal_delta_mae"]
            / max(mean_point["transition_weighted_temporal_delta_mae"], 1e-12))
        item["robust_gate_vs_mean"] = {
            "aggregate_psnr_gain_at_least_0p015_db": psnr_gain >= 0.015,
            "psnr_positive_on_at_least_three_of_four_sequences": (
                item["psnr_improved_vs_mean_sequence_count"] >= 3),
            "lpips_not_worse_by_more_than_0p5_percent": lpips_ratio <= 1.005,
            "lpips_reported": (
                item["lpips_reported"] and mean_point["lpips_reported"]),
            "temporal_not_worse_by_more_than_0p5_percent": temporal_ratio <= 1.005,
            "lpips_or_temporal_improves": (
                lpips_ratio < 1.0 or temporal_ratio < 1.0),
        }
        item["robust_gate_vs_mean"]["passes"] = all(
            item["robust_gate_vs_mean"].values())
        if item["robust_gate_vs_mean"]["passes"]:
            passing.append(item["tier"])
    return {
        "tiers": aggregates,
        "decision": {
            "passes_decoder_only_predictor_gate": bool(passing),
            "passing_tiers": passing,
            "reason": (
                "A tier must gain at least 0.015 dB in aggregate, improve PSNR on "
                "at least three of four sequences, keep aggregate LPIPS and temporal "
                "delta MAE within 0.5%, and improve at least one secondary metric."
            ),
        },
    }


def save_plot(results, baselines, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    max_latency = max(
        item["compute"]["predictor_ms_per_p_chunk"]
        for sequence in results for item in sequence["tiers"])
    norm = plt.Normalize(vmin=0.0, vmax=max(1e-6, max_latency))
    cmap = plt.get_cmap("viridis")
    offsets = {
        "all-base": (6, 10),
        "mean": (6, 24),
        "spatial-k4": (6, -24),
        "context-k32": (6, 2),
        "spatial-k4-additive": (6, -24),
        "context-k32-additive": (6, 2),
    }
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for axis, sequence in zip(axes.flat, results):
        name = sequence["sequence"]
        baseline = sorted(load_baseline_points(baselines, name), key=lambda item: item[2])
        axis.plot(
            [item[2] for item in baseline], [item[1] for item in baseline],
            color="0.45", marker="s", label="all-Base QP30/31/32")
        for qp, quality, rate in baseline:
            axis.annotate(
                f"QP{qp}", (rate, quality), xytext=(-4, -14),
                textcoords="offset points", fontsize=7, color="0.35")
        for tier in sequence["tiers"]:
            latency = tier["compute"]["predictor_ms_per_p_chunk"]
            axis.scatter(
                tier["stream"]["total_bpp"], tier["metrics"]["mean_frame_psnr"],
                color=cmap(norm(latency)), s=85, edgecolors="black")
            axis.annotate(
                f"{tier['tier']}\n{latency:.2f} ms/chunk",
                (tier["stream"]["total_bpp"], tier["metrics"]["mean_frame_psnr"]),
                xytext=offsets[tier["tier"]], textcoords="offset points", fontsize=8)
        axis.set_title(name)
        axis.set_xlabel("actual total bpp")
        axis.set_ylabel("mean frame PSNR (dB)")
        axis.margins(x=0.05, y=0.14)
        axis.grid(alpha=0.25)
    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes.ravel().tolist(), shrink=0.78)
    colorbar.set_label("additional predictor wall time (ms/P chunk)")
    figure.suptitle("Fixed-route R-D-C probe (color: additional predictor latency)")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


@torch.inference_mode()
def main():
    args = parse_args()
    if args.decode_repeats < 1:
        raise ValueError("decode-repeats must be positive")
    if args.height % 64 or args.width % 64:
        raise ValueError("width and height must be divisible by 64")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    sequences = parse_sequences(args.sequence)
    baselines = parse_baselines(args.baseline)
    set_torch_env()
    torch.cuda.set_device(args.cuda_idx)
    device = torch.device(f"cuda:{args.cuda_idx}")
    torch.cuda.set_stream(torch.cuda.Stream(device=device))
    i_net, p_net = load_models(args, device)

    results = []
    for name, source_dir in sequences:
        result = run_sequence(
            name, source_dir, i_net, p_net, args, device, baselines)
        results.append(result)
        print(json.dumps({
            "sequence": name,
            "tiers": [{
                "tier": item["tier"],
                "bytes": item["stream"]["total_bytes"],
                "psnr": item["metrics"]["mean_frame_psnr"],
                "predictor_ms_per_chunk": item["compute"]["predictor_ms_per_p_chunk"],
            } for item in result["tiers"]],
        }, ensure_ascii=False), flush=True)

    output_root = Path(args.output_dir)
    rows = make_rows(results, baselines)
    with (output_root / "aggregate.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    save_plot(results, baselines, output_root / "rdc_probe.png")
    aggregate = {
        "experiment": "adaptive_chunk_coding_stage_b_decoder_only_predictor_probe_v1",
        "scientific_scope": {
            "target_scenario": "uplink bandwidth constrained, GPU receiver",
            "route_policy": "fixed source-aware Stage-1 mean-fill Oracle route",
            "route_is_deployable": False,
            "purpose": (
                "Isolate whether decoder-known Base residuals and common parameters "
                "contain useful no-training signal before requesting training data."),
            "predictors_are_learned_generators": False,
            "source_or_omitted_y_available_to_predictor": False,
            "compute_scaling_claim": (
                "Tiers differ by measured work and latency. The two controls operate on "
                "routed latent blocks; no claim is made that calibration cost scales "
                "linearly with the skipped ratio."),
            "decision_rule": (
                "A predictor must improve quality or temporal/perceptual metrics at the "
                "same fixed route enough to avoid domination by mean-fill and adjacent-QP "
                "all-Base points before Controller or Residual work starts."),
        },
        "args": vars(args),
        "aggregate": aggregate_tiers(results, args.width, args.height),
        "sequences": results,
    }
    (output_root / "summary.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
