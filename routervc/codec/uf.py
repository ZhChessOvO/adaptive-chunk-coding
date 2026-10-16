"""Uf. Extracted from the validated RouterVC research implementation."""

from __future__ import annotations
import argparse
import io
import math
import time
import numpy as np
import torch
from src.models.image_model import DMCI
from src.models.video_model_ht import DMC
from src.models.video_model_ht import g_frame_delay
from src.utils.common import ModelStructure
from src.utils.common import get_state_dict
from src.utils.stream_helper import NalType
from src.utils.stream_helper import SPSHelper
from src.utils.stream_helper import read_header
from src.utils.stream_helper import read_ip_remaining
from src.utils.stream_helper import read_sps_remaining
from src.utils.stream_helper import write_ip
from src.utils.stream_helper import write_sps
from src.utils.transforms import rgb2ycbcr
from src.utils.transforms import ycbcr2rgb


def tensor_from_rgb(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(frame.transpose(2, 0, 1).copy()).unsqueeze(0)
    tensor = tensor.to(device=device, dtype=torch.float32) / 255.0
    return (rgb2ycbcr(tensor).half() - 0.5).to(memory_format=torch.channels_last)


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
    return (i_net, p_net)


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
        current = frames[frame_index : frame_index + valid_count]
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
                and (frame_index + g_frame_delay) % reset_interval == 1
            )
            encoded = p_net.compress(source, qp, reset, padding_b, padding_r)
        torch.cuda.synchronize(device)
        nal_times.append(time.perf_counter() - nal_started)
        sps = {"sps_id": -1, "height": height, "width": width}
        sps_id, is_new = sps_helper.get_sps_id(sps)
        sps["sps_id"] = sps_id
        if is_new:
            write_sps(output, sps)
        write_ip(
            output,
            is_intra,
            sps_id,
            qp,
            int(encoded["ec_parallel"]),
            reset,
            bytes(encoded["bit_stream"]),
        )
        frame_index += valid_count
    torch.cuda.synchronize(device)
    stream = output.getvalue()
    return (
        stream,
        {
            "seconds": time.perf_counter() - started,
            "nal_seconds": nal_times,
            "bytes": len(stream),
            "frames": len(frames),
            "width": width,
            "height": height,
            "qp_i": qp_i,
            "qp_p": qp_p,
        },
    )


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
                decoded["x_hat"], apply_feature_adaptor=False
            )
            outputs = [decoded["x_hat"]]
        elif header["nal_type"] == NalType.NAL_P:
            decoded = p_net.decompress(payload, sps, qp, ec_part, reset)
            outputs = decoded["x_hat"]
        else:
            raise ValueError(f"unexpected NAL type {header['nal_type']}")
        if chunk_observer is not None:
            chunk_observer(len(frames), sps, outputs)
        for output in outputs[: frame_count - len(frames)]:
            frames.append(rgb_from_tensor(output, sps["height"], sps["width"]))
    if stream.tell() != len(data):
        raise ValueError("trailing bytes after requested DCVC-UF frames")
    return frames


def rgb_mse(reference: list[np.ndarray], reconstruction: list[np.ndarray]) -> float:
    squared = sum(
        (
            float(np.sum((source.astype(np.float64) - decoded.astype(np.float64)) ** 2))
            for source, decoded in zip(reference, reconstruction)
        )
    )
    values = sum((source.size for source in reference))
    return squared / values


def psnr_from_mse(value: float) -> float:
    return float("inf") if value == 0 else 10.0 * math.log10(255.0**2 / value)


def temporal_delta_mae(
    reference: list[np.ndarray], reconstruction: list[np.ndarray]
) -> float:
    if len(reference) < 2:
        return 0.0
    values = []
    for index in range(1, len(reference)):
        source_delta = reference[index].astype(np.float32) - reference[
            index - 1
        ].astype(np.float32)
        decoded_delta = reconstruction[index].astype(np.float32) - reconstruction[
            index - 1
        ].astype(np.float32)
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


from pathlib import Path
from types import SimpleNamespace
from routervc.utils.io import file_hash


class BaseCodec:

    def __init__(self, model_i: Path, model_p: Path):
        import torch
        from routervc.codec.uf import load_codecs
        from routervc.codec.intra import initialize_intra_decoder_proxy
        from src.utils.common import set_torch_env

        set_torch_env()
        torch.set_num_threads(4)
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
        self.stream = torch.cuda.Stream(device=self.device)
        torch.cuda.set_stream(self.stream)
        self.models = {
            "model_i_sha256": file_hash(model_i),
            "model_p_sha256": file_hash(model_p),
        }
        args = SimpleNamespace(
            model_path_i=model_i, model_path_p=model_p, skip_thres=0.0
        )
        self.i_net, self.p_net = load_codecs(args, self.device)
        initialize_intra_decoder_proxy(self.i_net)
        torch.cuda.synchronize()

    def encode(self, frames: np.ndarray, qp: int = 8) -> tuple[bytes, dict]:
        import torch
        from routervc.codec.uf import encode_dcvc_stream

        torch.cuda.set_stream(self.stream)
        with torch.inference_mode():
            return encode_dcvc_stream(
                list(frames),
                qp,
                qp,
                self.i_net,
                self.p_net,
                self.device,
                reset_interval=32,
            )

    def decode(self, data: bytes, count: int) -> np.ndarray:
        import torch
        from routervc.codec.uf import decode_dcvc_stream

        torch.cuda.set_stream(self.stream)
        with torch.inference_mode():
            frames = decode_dcvc_stream(
                data, count, self.i_net, self.p_net, self.device
            )
        torch.cuda.synchronize()
        return np.stack(frames)
