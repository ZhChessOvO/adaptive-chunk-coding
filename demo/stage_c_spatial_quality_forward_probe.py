#!/usr/bin/env python3
"""E23: no-training forward probe for one-shot spatial quality modulation.

The released checkpoint stores 64 channel-scale rows but selects only one row
per coding unit.  This probe selects those existing rows spatially for the
analysis transform, synthesis transform, temporal prior, and image-y
quantization.  It exercises an I frame plus one eight-frame P chunk.

No entropy payload is written.  Rate is a model probability estimate plus the
exact new map/header syntax size, never presented as an on-disk codec result.
The purpose is to decide whether the spatial design is numerically viable
before implementing map-indexed rANS and starting codec fine-tuning.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.stage_c_evaluate_seedvr2_gate import load_source
from demo.stage_c_three_path_roi_probe import (
    LPIPSAlex,
    evaluate_variant,
    rgb_from_tensor,
    tensor_from_rgb,
)
from src.layers.layers import QuantFunc
from src.models.image_model import DMCI
from src.models.video_model_ht import DMC
from src.utils.common import ModelStructure, get_state_dict, set_torch_env
from src.utils.stream_helper import write_spatial_ip, write_sps


ACTION_TO_QP = {0: 16, 1: 8, 2: 32}
ACTION_NAMES = {0: "Base", 1: "Generate", 2: "Enhance"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-summary", type=Path, required=True)
    parser.add_argument("--route-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-count", type=int, default=9)
    parser.add_argument("--cell-size", type=int, default=64)
    parser.add_argument("--visual-frame", type=int, default=9)
    parser.add_argument("--cuda-idx", type=int, default=0)
    parser.add_argument(
        "--model-path-i", type=Path,
        default=Path("checkpoints/cvpr2026_image.pth.tar"))
    parser.add_argument(
        "--model-path-p", type=Path,
        default=Path("checkpoints/cvpr2026_video_hts.pth.tar"))
    args = parser.parse_args()
    if args.frame_count != 9:
        parser.error("the registered E23 probe is one I plus one 8-frame P chunk")
    if args.cell_size != 64:
        parser.error("the registered E23 syntax cell is 64 pixels")
    return args


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
    return F.interpolate(
        values, size=size, mode="bilinear", align_corners=False)


def mixed_z_bits(model, z_hat: torch.Tensor, qp_map: torch.Tensor) -> torch.Tensor:
    z_qp = F.interpolate(
        qp_map[:, None].float(), size=z_hat.shape[-2:], mode="nearest")
    total = torch.zeros((), device=z_hat.device, dtype=torch.float32)
    for qp in torch.unique(qp_map).tolist():
        index = torch.tensor([int(qp)], device=z_hat.device, dtype=torch.long)
        bits = model.get_z_bits(z_hat, index).float()
        mask = (z_qp == qp).to(bits.dtype)
        total = total + torch.sum(bits * mask)
    return total


def spatial_i_forward(
    model: DMCI,
    x: torch.Tensor,
    qp_map: torch.Tensor,
    interpolation: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    feature_size = (x.shape[-2] // 8, x.shape[-1] // 8)
    q_enc = select_scale(model.q_scale_enc, qp_map, feature_size, interpolation)
    y = model.enc(x, q_enc)
    q_y_enc = select_scale(
        model.q_scale_y_enc, qp_map, y.shape[-2:], interpolation)
    q_y_dec = select_scale(
        model.q_scale_y_dec, qp_map, y.shape[-2:], interpolation)
    z = model.hyper_enc(y)
    z_hat = QuantFunc.apply(z)
    params = model.y_prior_fusion(model.hyper_dec(z_hat))
    params = params[:, :, :y.shape[-2], :y.shape[-1]]
    y_res, _y_q, y_hat, scales = model.forward_prior_4x(
        y, q_y_enc, q_y_dec, params,
        model.y_spatial_prior_reduction,
        model.y_spatial_prior_adaptor_1,
        model.y_spatial_prior_adaptor_2,
        model.y_spatial_prior_adaptor_3,
        model.y_spatial_prior,
    )
    q_dec = select_scale(model.q_scale_dec, qp_map, feature_size, interpolation)
    x_hat = model.dec(y_hat, q_dec)
    bits = torch.sum(model.get_y_bits(y_res, scales).float())
    bits = bits + mixed_z_bits(model, z_hat, qp_map)
    return x_hat, bits


def spatial_p_forward(
    model: DMC,
    x: torch.Tensor,
    reference: torch.Tensor,
    qp_map: torch.Tensor,
    interpolation: str,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    model.clear_dpb()
    model.ref_feature = F.pixel_unshuffle(reference, 8)
    model.apply_feature_adaptor()
    feature_size = model.ctx.shape[-2:]
    q_enc = select_scale(model.q_encoder, qp_map, feature_size, interpolation)
    y = model.encoder(x, model.ctx, q_enc)
    z = model.hyper_encoder(y)
    z_hat = QuantFunc.apply(z)
    q_feature = select_scale(
        model.q_feature, qp_map, model.memory.shape[-2:], interpolation)
    params = model.res_prior_param_decoder(z_hat, model.memory, q_feature)
    y_res, _y_q, y_hat, scales = model.forward_prior_4x(
        y, None, None, params,
        model.y_spatial_prior_reduction,
        model.y_spatial_prior_adaptor_1,
        model.y_spatial_prior_adaptor_2,
        model.y_spatial_prior_adaptor_3,
        model.y_spatial_prior,
        spatial_prior_has_scales=False,
    )
    q_dec = select_scale(model.q_decoder, qp_map, feature_size, interpolation)
    x_hat, feature = model.get_recon_and_feature(y_hat, model.ctx, q_dec)
    model.set_ref_feature(feature, False)
    bits = torch.sum(model.get_y_bits(y_res, scales).float())
    bits = bits + mixed_z_bits(model, z_hat, qp_map)
    return x_hat, bits


def forward_sequence(
    i_net: DMCI,
    p_net: DMC,
    tensors: list[torch.Tensor],
    qp_map: torch.Tensor,
    interpolation: str,
) -> tuple[list[torch.Tensor], dict]:
    torch.cuda.synchronize(qp_map.device)
    torch.cuda.reset_peak_memory_stats(qp_map.device)
    started = time.perf_counter()
    intra, bits_i = spatial_i_forward(
        i_net, tensors[0], qp_map, interpolation)
    p_input = torch.cat(tensors[1:], dim=1)
    predicted, bits_p = spatial_p_forward(
        p_net, p_input, intra, qp_map, interpolation)
    torch.cuda.synchronize(qp_map.device)
    return [intra, *predicted], {
        "forward_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(qp_map.device)),
        "estimated_entropy_bits_i": float(bits_i.item()),
        "estimated_entropy_bits_p": float(bits_p.item()),
        "estimated_entropy_bits_total": float((bits_i + bits_p).item()),
    }


def direct_uniform_sequence(
    i_net: DMCI,
    p_net: DMC,
    tensors: list[torch.Tensor],
    qp: int,
) -> list[torch.Tensor]:
    index = torch.tensor([qp], device=tensors[0].device, dtype=torch.long)
    intra = i_net.forward_one_frame(tensors[0], index, recon_only=True)
    p_net.clear_dpb()
    p_net.ref_feature = F.pixel_unshuffle(intra, 8)
    result = p_net.forward_one_frame(torch.cat(tensors[1:], dim=1), index)
    return [intra, *result["x_hat"]]


def as_rgb(frames: list[torch.Tensor], height: int, width: int) -> list[np.ndarray]:
    return [rgb_from_tensor(frame, height, width) for frame in frames]


def local_action_metrics(
    reference: list[np.ndarray],
    reconstruction: list[np.ndarray],
    actions: np.ndarray,
    tile_size: int,
    lpips: LPIPSAlex,
) -> dict:
    grid_height, grid_width = actions.shape
    records = {name: [] for name in ACTION_NAMES.values()}
    for row in range(grid_height):
        for column in range(grid_width):
            y, x = row * tile_size, column * tile_size
            ref = [frame[y:y + tile_size, x:x + tile_size] for frame in reference]
            rec = [frame[y:y + tile_size, x:x + tile_size] for frame in reconstruction]
            records[ACTION_NAMES[int(actions[row, column])]].append(
                evaluate_variant(ref, rec, lpips))
    return {
        name: {
            "tile_count": len(values),
            "mean_psnr_db": float(np.mean([value["psnr_db"] for value in values])),
            "mean_lpips_alex": float(np.mean([
                value["lpips_alex"] for value in values])),
        }
        for name, values in records.items() if values
    }


def transition_edge_error(
    reference: list[np.ndarray],
    reconstruction: list[np.ndarray],
    actions: np.ndarray,
    cell_size: int,
) -> float:
    values = []
    rows, columns = actions.shape
    for row in range(rows):
        y0, y1 = row * cell_size, (row + 1) * cell_size
        for column in range(1, columns):
            if actions[row, column - 1] == actions[row, column]:
                continue
            x = column * cell_size
            for source, decoded in zip(reference, reconstruction):
                source_edge = source[y0:y1, x].astype(np.float32) - source[
                    y0:y1, x - 1].astype(np.float32)
                decoded_edge = decoded[y0:y1, x].astype(np.float32) - decoded[
                    y0:y1, x - 1].astype(np.float32)
                values.append(float(np.mean(np.abs(source_edge - decoded_edge))))
    for row in range(1, rows):
        y = row * cell_size
        for column in range(columns):
            if actions[row - 1, column] == actions[row, column]:
                continue
            x0, x1 = column * cell_size, (column + 1) * cell_size
            for source, decoded in zip(reference, reconstruction):
                source_edge = source[y, x0:x1].astype(np.float32) - source[
                    y - 1, x0:x1].astype(np.float32)
                decoded_edge = decoded[y, x0:x1].astype(np.float32) - decoded[
                    y - 1, x0:x1].astype(np.float32)
                values.append(float(np.mean(np.abs(source_edge - decoded_edge))))
    return float(np.mean(values)) if values else 0.0


def syntax_bytes(actions: list[int]) -> dict:
    output = io.BytesIO()
    write_sps(output, {"sps_id": 0, "height": 512, "width": 512})
    sps_bytes = output.tell()
    units = []
    for is_i in (True, False):
        before = output.tell()
        write_spatial_ip(
            output, is_i, 0, (8, 16, 32), 64, actions,
            2, 0, b"")
        units.append(output.tell() - before)
    return {
        "sps_bytes": sps_bytes,
        "coding_unit_bytes_without_entropy_payload": units,
        "total_syntax_bytes": output.tell(),
    }


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    suffix = "-Bold" if bold else ""
    return ImageFont.truetype(
        f"/usr/share/fonts/truetype/dejavu/DejaVuSans{suffix}.ttf", size)


def panel(frame: np.ndarray, title: str, quality: dict | None) -> Image.Image:
    image = Image.fromarray(frame)
    banner = 68
    result = Image.new("RGB", (image.width, image.height + banner), "white")
    result.paste(image, (0, banner))
    draw = ImageDraw.Draw(result)
    draw.text((8, 7), title, fill="black", font=font(18, True))
    if quality:
        draw.text(
            (8, 39),
            f"LPIPS {quality['lpips_alex']:.4f} | PSNR {quality['psnr_db']:.3f} | "
            f"T {quality['temporal_delta_mae']:.3f}",
            fill=(50, 50, 50), font=font(14))
    return result


def make_visual(
    path: Path,
    frame_index: int,
    reference: list[np.ndarray],
    outputs: dict[str, list[np.ndarray]],
    metrics: dict[str, dict],
) -> None:
    names = ["uniform-qp8", "uniform-qp16", "uniform-qp32",
             "spatial-nearest", "spatial-bilinear"]
    panels = [panel(reference[frame_index], "GT", None)]
    titles = {
        "uniform-qp8": "Uniform QP8",
        "uniform-qp16": "Uniform QP16",
        "uniform-qp32": "Uniform QP32",
        "spatial-nearest": "Spatial 8/16/32 nearest",
        "spatial-bilinear": "Spatial 8/16/32 smoothed scales",
    }
    for name in names:
        panels.append(panel(outputs[name][frame_index], titles[name], metrics[name]))
    columns = 3
    width = max(item.width for item in panels)
    height = max(item.height for item in panels)
    canvas = Image.new("RGB", (columns * width, 2 * height), (230, 230, 230))
    for index, item in enumerate(panels):
        canvas.paste(item, ((index % columns) * width, (index // columns) * height))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("E23 requires CUDA")
    set_torch_env()
    torch.cuda.set_device(args.cuda_idx)
    device = torch.device(f"cuda:{args.cuda_idx}")
    gate = json.loads(args.gate_summary.read_text(encoding="utf-8"))
    route = json.loads(args.route_summary.read_text(encoding="utf-8"))
    reference = load_source(gate, args.frame_count)
    height, width = reference[0].shape[:2]
    if (height, width) != (512, 512):
        raise ValueError("registered E23 probe expects a 512x512 crop")
    actions_128 = np.asarray(
        route["variants"]["joint-three-path-oracle"]["actions"],
        dtype=np.int64).reshape(4, 4)
    actions = np.repeat(np.repeat(actions_128, 2, axis=0), 2, axis=1)
    qp_values = np.vectorize(ACTION_TO_QP.__getitem__)(actions)
    qp_map = torch.from_numpy(qp_values.copy()).unsqueeze(0).to(device)
    tensors = [tensor_from_rgb(frame, device) for frame in reference]

    i_net = DMCI().eval()
    i_net.load_state_dict(get_state_dict(str(args.model_path_i)))
    i_net = i_net.half().to(device).to(memory_format=torch.channels_last)
    p_net = DMC(ModelStructure.HTS).eval()
    p_net.load_state_dict(get_state_dict(str(args.model_path_p)))
    p_net = p_net.half().to(device).to(memory_format=torch.channels_last)

    direct_q16 = direct_uniform_sequence(i_net, p_net, tensors, 16)
    outputs_tensors = {}
    runtime = {}
    for qp in (8, 16, 32):
        uniform = torch.full_like(qp_map, qp)
        name = f"uniform-qp{qp}"
        outputs_tensors[name], runtime[name] = forward_sequence(
            i_net, p_net, tensors, uniform, "nearest")
    outputs_tensors["spatial-nearest"], runtime["spatial-nearest"] = forward_sequence(
        i_net, p_net, tensors, qp_map, "nearest")
    outputs_tensors["spatial-bilinear"], runtime["spatial-bilinear"] = forward_sequence(
        i_net, p_net, tensors, qp_map, "bilinear")

    uniform_diff = max(
        float(torch.max(torch.abs(first - second)).item())
        for first, second in zip(direct_q16, outputs_tensors["uniform-qp16"])
    )
    if uniform_diff != 0.0:
        raise RuntimeError(
            f"spatial path does not collapse exactly at uniform QP16: {uniform_diff}")

    outputs = {
        name: as_rgb(frames, height, width)
        for name, frames in outputs_tensors.items()
    }
    lpips = LPIPSAlex(True)
    metrics = {
        name: evaluate_variant(reference, frames, lpips)
        for name, frames in outputs.items()
    }
    syntax = syntax_bytes(actions.reshape(-1).tolist())
    for name, record in runtime.items():
        record["exact_spatial_syntax_bytes_if_written"] = (
            syntax["total_syntax_bytes"] if name.startswith("spatial") else None)
        record["estimated_total_bits_with_spatial_syntax"] = (
            record["estimated_entropy_bits_total"]
            + 8 * syntax["total_syntax_bytes"]
            if name.startswith("spatial") else None)
        record["estimated_total_bytes_ceil"] = (
            math.ceil(record["estimated_total_bits_with_spatial_syntax"] / 8)
            if name.startswith("spatial") else None)

    local_metrics = {
        name: local_action_metrics(
            reference, outputs[name], actions_128, 128, lpips)
        for name in ("uniform-qp16", "spatial-nearest", "spatial-bilinear")
    }
    seam = {
        name: transition_edge_error(
            reference, outputs[name], actions, args.cell_size)
        for name in ("uniform-qp16", "spatial-nearest", "spatial-bilinear")
    }

    frame_index = min(max(args.visual_frame, 1), args.frame_count) - 1
    visual = args.output_dir / "visuals" / "spatial_forward_probe.png"
    make_visual(visual, frame_index, reference, outputs, metrics)
    for name, frames in outputs.items():
        target = args.output_dir / "frames" / name
        target.mkdir(parents=True, exist_ok=True)
        for index, frame in enumerate(frames, start=1):
            Image.fromarray(frame).save(target / f"im{index:05d}.png")

    base_local = local_metrics["uniform-qp16"]
    nearest_local = local_metrics["spatial-nearest"]
    response = {
        action: {
            "lpips_delta_vs_uniform_qp16": (
                nearest_local[action]["mean_lpips_alex"]
                - base_local[action]["mean_lpips_alex"]),
            "psnr_delta_db_vs_uniform_qp16": (
                nearest_local[action]["mean_psnr_db"]
                - base_local[action]["mean_psnr_db"]),
        }
        for action in nearest_local
    }
    result = {
        "experiment": "E23 no-training one-shot spatial quality forward probe",
        "status": "forward-path-probe-complete-entropy-stream-pending",
        "source_role": gate["source_role"],
        "source_files": gate["source_files"][:args.frame_count],
        "frames": args.frame_count,
        "quality_profile": {"Generate": 8, "Base": 16, "Enhance": 32},
        "map": {
            "cell_size_pixels": args.cell_size,
            "shape": list(actions.shape),
            "actions": actions.tolist(),
            "derived_from_e21_128px_oracle_by_nearest_repeat": True,
            "budget_known_before_encoding": True,
        },
        "uniform_regression": {
            "qp16_spatial_path_vs_original_python_path_max_abs_tensor_error": uniform_diff,
            "exact": True,
        },
        "metrics": metrics,
        "local_action_metrics": local_metrics,
        "nearest_spatial_response_vs_uniform_qp16": response,
        "transition_edge_delta_mae": seam,
        "runtime_and_rate_estimate": runtime,
        "syntax": syntax,
        "scientific_boundary": {
            "actual_entropy_stream_written": False,
            "rate_is_model_probability_estimate": True,
            "map_and_header_bytes_are_exact": True,
            "result_is_not_a_codec_rd_point": True,
            "ground_truth_used_by_forward_path": False,
            "true_fill_used": False,
            "omitted_latent_prediction_used": False,
        },
        "next_implementation": [
            "map-indexed z CDF selection in encoder and decoder",
            "spatial channel-scale tensors in CUDA DMCI/DMCHTS proxies",
            "one-shot reference-loop-consistent entropy payload",
            "short frozen-backbone pilot, then codec fine-tuning if the pilot passes",
        ],
        "visual": str(visual),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = args.output_dir / "summary.json"
    summary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "summary": str(summary),
        "visual": str(visual),
        "uniform_regression": result["uniform_regression"],
        "metrics": metrics,
        "spatial_response": response,
        "transition_edge_delta_mae": seam,
        "rate_estimate": {
            name: {
                "estimated_entropy_bits": record["estimated_entropy_bits_total"],
                "estimated_total_bytes_ceil": record["estimated_total_bytes_ceil"],
            }
            for name, record in runtime.items() if name.startswith("spatial")
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
