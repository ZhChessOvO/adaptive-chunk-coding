"""Runtime. Extracted from the validated RouterVC research implementation."""

from __future__ import annotations
import argparse
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
from routervc.generation.runner import configure_runner
from routervc.generation.runner import pad_temporal
from routervc.generation.runner import resize_and_normalize
from routervc.generation.runner import resize_output


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def arrays_from_seed_tensor(sample: torch.Tensor, frame_count: int) -> list[np.ndarray]:
    if sample.ndim == 3:
        sample = sample.unsqueeze(1)
    if sample.ndim != 4:
        raise ValueError(
            f"SeedVR2 output must be C,T,H,W (or C,H,W for T=1), got {sample.shape}"
        )
    sample = sample[:, :frame_count]
    sample = sample.permute(1, 2, 3, 0).float().cpu()
    values = sample.clamp(-1, 1).add_(1).mul_(127.5).round_().byte().numpy()
    return [frame.copy() for frame in values]


class PersistentSeedVR2:

    def __init__(self, args: argparse.Namespace) -> None:
        bridge_args = SimpleNamespace(
            upstream_root=resolve(args.upstream_root),
            dit_checkpoint=resolve(args.dit_checkpoint),
            lora_checkpoint=(
                resolve(args.lora_checkpoint)
                if getattr(args, "lora_checkpoint", None) is not None
                else None
            ),
            lora_strength=float(getattr(args, "lora_strength", 1.0)),
            vae_checkpoint=resolve(args.vae_checkpoint),
            positive_embedding=resolve(args.positive_embedding),
            negative_embedding=resolve(args.negative_embedding),
            sample_steps=args.sample_steps,
            cfg_scale=args.cfg_scale,
            dit_dtype=args.dit_dtype,
        )
        started = time.perf_counter()
        self.runner, self.device = configure_runner(bridge_args)
        self.dtype = getattr(torch, args.dit_dtype)
        self.runner.dit.requires_grad_(False).eval().to(
            device=self.device, dtype=self.dtype
        )
        self.runner.vae.to(self.device)
        self.positive = torch.load(
            bridge_args.positive_embedding, map_location=self.device
        )
        self.negative = torch.load(
            bridge_args.negative_embedding, map_location=self.device
        )
        self.cfg_scale = args.cfg_scale
        self.sample_steps = args.sample_steps
        self.model_load_seconds = time.perf_counter() - started

    @torch.inference_mode()
    def restore(
        self,
        frames: list[np.ndarray],
        seed: int,
        processing_height: int | None = None,
        processing_width: int | None = None,
        output_height: int | None = None,
        output_width: int | None = None,
    ) -> tuple[list[np.ndarray], dict]:
        values = np.stack(frames)
        input_height, input_width = values.shape[1:3]
        processing_height = processing_height or input_height
        processing_width = processing_width or input_width
        output_height = output_height or input_height
        output_width = output_width or input_width
        if any((value % 16 for value in (processing_height, processing_width))):
            raise ValueError("SeedVR2 processing dimensions must be divisible by 16")
        tensor = torch.from_numpy(values.copy()).permute(0, 3, 1, 2)
        tensor = tensor.float().div_(255.0)
        condition, original_length = pad_temporal(
            resize_and_normalize(
                tensor, processing_height, processing_width, self.device
            )
        )
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=self.dtype == torch.bfloat16,
        ):
            cond_latents = self.runner.vae_encode([condition])
            noises = [torch.randn_like(cond_latents[0])]
            conditions = [
                self.runner.get_condition(
                    noises[0], task="sr", latent_blur=cond_latents[0]
                )
            ]
            outputs = self.runner.inference(
                noises=noises,
                conditions=conditions,
                texts_pos=[self.positive],
                texts_neg=[self.negative],
                cfg_scale=self.cfg_scale,
                dit_offload=False,
            )
            outputs = [
                resize_output(sample, output_height, output_width) for sample in outputs
            ]
        torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        peak = int(torch.cuda.max_memory_allocated(self.device))
        is_full_frame = (
            input_height == 512
            and input_width == 512
            and (processing_height == 512)
            and (processing_width == 512)
            and (output_height == 512)
            and (output_width == 512)
        )
        return (
            arrays_from_seed_tensor(outputs[0], original_length),
            {
                "seconds_model_load_excluded": elapsed,
                "peak_cuda_allocated_bytes": peak,
                "input_shape": [original_length, input_height, input_width],
                "processing_shape": [
                    original_length,
                    processing_height,
                    processing_width,
                ],
                "output_shape": [original_length, output_height, output_width],
                "seed": seed,
                "sample_steps": self.sample_steps,
                "actual_compute_scope": "full-frame" if is_full_frame else "roi-crop",
                "full_frame_actual_compute": is_full_frame,
                "roi_compute_measured_here": not is_full_frame,
            },
        )
