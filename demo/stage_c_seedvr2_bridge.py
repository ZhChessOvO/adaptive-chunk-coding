#!/usr/bin/env python3
"""Run official SeedVR2-3B weights on a lossless folder of decoded frames.

The upstream prototype requires Apex and FlashAttention.  For the small 4090
diagnostic used here, this bridge keeps the released architecture and weights
but substitutes parameter-compatible PyTorch normalizations and an exact
variable-length SDPA implementation.  It does not train or fine-tune SeedVR2.

Launch with ``torchrun --standalone --nproc-per-node=1`` because the upstream
inference stack initializes a process group even for one GPU.
"""

from __future__ import annotations

import argparse
import gc
import importlib.machinery
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = REPO_ROOT / "third_party" / "SeedVR2"


def install_flash_attention_fallback() -> None:
    """Expose the subset of flash_attn used by upstream SeedVR2."""

    def flash_attn_varlen_func(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int | None = None,
        max_seqlen_k: int | None = None,
        dropout_p: float = 0.0,
        softmax_scale: float | None = None,
        causal: bool = False,
        **_: object,
    ) -> torch.Tensor:
        del max_seqlen_q, max_seqlen_k
        outputs = []
        q_offsets = cu_seqlens_q.tolist()
        k_offsets = cu_seqlens_k.tolist()
        for index in range(len(q_offsets) - 1):
            q_part = q[q_offsets[index]:q_offsets[index + 1]]
            k_part = k[k_offsets[index]:k_offsets[index + 1]]
            v_part = v[k_offsets[index]:k_offsets[index + 1]]
            q_part = q_part.permute(1, 0, 2).unsqueeze(0)
            k_part = k_part.permute(1, 0, 2).unsqueeze(0)
            v_part = v_part.permute(1, 0, 2).unsqueeze(0)
            output = F.scaled_dot_product_attention(
                q_part,
                k_part,
                v_part,
                dropout_p=dropout_p,
                is_causal=causal,
                scale=softmax_scale,
            )
            outputs.append(output.squeeze(0).permute(1, 0, 2))
        return torch.cat(outputs, dim=0)

    module = types.ModuleType("flash_attn")
    module.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None)
    module.flash_attn_varlen_func = flash_attn_varlen_func
    sys.modules.setdefault("flash_attn", module)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lossless PNG bridge from DCVC-UF reconstruction to SeedVR2")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, default=UPSTREAM_ROOT)
    parser.add_argument(
        "--dit-checkpoint", type=Path,
        default=UPSTREAM_ROOT / "ckpts" / "seedvr2_ema_3b.pth")
    parser.add_argument(
        "--vae-checkpoint", type=Path,
        default=UPSTREAM_ROOT / "ckpts" / "ema_vae.pth")
    parser.add_argument(
        "--positive-embedding", type=Path,
        default=UPSTREAM_ROOT / "pos_emb.pt")
    parser.add_argument(
        "--negative-embedding", type=Path,
        default=UPSTREAM_ROOT / "neg_emb.pt")
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument(
        "--output-height", type=int,
        help="Resize restored frames to this height before lossless PNG output")
    parser.add_argument(
        "--output-width", type=int,
        help="Resize restored frames to this width before lossless PNG output")
    parser.add_argument("--sample-steps", type=int, default=1)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument(
        "--dit-dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument(
        "--sequential-offload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep DiT and VAE activations resident on the GPU in sequence")
    args = parser.parse_args()
    if args.sample_steps < 1:
        parser.error("--sample-steps must be positive")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    for name in ("height", "width", "output_height", "output_width"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def load_frames(
    path: Path, max_frames: int | None = None
) -> tuple[list[Path], torch.Tensor]:
    paths = sorted(path.glob("*.png"))
    if max_frames is not None:
        paths = paths[:max_frames]
    if not paths:
        raise ValueError(f"no PNG frames found in {path}")
    arrays = [np.asarray(Image.open(item).convert("RGB"), dtype=np.uint8) for item in paths]
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        raise ValueError("all input PNGs must have the same shape")
    tensor = torch.from_numpy(np.stack(arrays).copy()).permute(0, 3, 1, 2)
    return paths, tensor.float().div_(255.0)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)


def resize_and_normalize(
    frames: torch.Tensor, height: int, width: int, device: torch.device
) -> torch.Tensor:
    frames = frames.to(device)
    if frames.shape[-2:] != (height, width):
        frames = F.interpolate(
            frames,
            size=(height, width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
    if height % 16 or width % 16:
        raise ValueError("SeedVR2 output height and width must be divisible by 16")
    frames = frames.clamp_(0, 1).mul_(2).sub_(1)
    return frames.permute(1, 0, 2, 3).contiguous()


def resize_output(sample: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resize a C,T,H,W result while keeping the temporal axis untouched."""

    if sample.ndim == 3:
        sample = sample.unsqueeze(1)
    if sample.ndim != 4:
        raise ValueError(
            f"SeedVR2 output must be C,T,H,W (or C,H,W for T=1), got {sample.shape}")
    if sample.shape[-2:] == (height, width):
        return sample
    frames = sample.permute(1, 0, 2, 3)
    frames = F.interpolate(
        frames,
        size=(height, width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    return frames.permute(1, 0, 2, 3).contiguous()


def pad_temporal(sample: torch.Tensor) -> tuple[torch.Tensor, int]:
    original_length = sample.shape[1]
    if original_length == 1 or (original_length - 1) % 4 == 0:
        return sample, original_length
    padding = 4 - ((original_length - 1) % 4)
    tail = sample[:, -1:].expand(-1, padding, -1, -1)
    return torch.cat([sample, tail], dim=1), original_length


def configure_safetensors_dit(runner, checkpoint: Path) -> None:
    """Mirror upstream meta-device loading for a safetensors checkpoint."""

    from common.config import create_object
    from common.distributed.meta_init_utils import (
        meta_non_persistent_buffer_init_fn,
    )
    from safetensors.torch import load_file

    with torch.device("meta"):
        runner.dit = create_object(runner.config.dit.model)
    runner.dit.set_gradient_checkpointing(runner.config.dit.gradient_checkpoint)
    state = load_file(str(checkpoint.resolve()), device="cpu")
    loading_info = runner.dit.load_state_dict(state, strict=True, assign=True)
    print(f"Loading pretrained safetensors checkpoint from {checkpoint}")
    print(f"Loading info: {loading_info}")
    runner.dit = meta_non_persistent_buffer_init_fn(runner.dit)
    del state
    gc.collect()

    num_params = sum(
        parameter.numel()
        for parameter in runner.dit.parameters()
        if parameter.requires_grad
    )
    print(f"DiT trainable parameters: {num_params:,}")


def configure_runner(args: argparse.Namespace):
    upstream = args.upstream_root.resolve()
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    install_flash_attention_fallback()

    from common.config import load_config
    from common.distributed import get_device, init_torch
    from omegaconf import OmegaConf
    from projects.video_diffusion_sr.infer import VideoDiffusionInfer

    previous_cwd = Path.cwd()
    try:
        # Upstream inheritance entries are repository-relative rather than
        # relative to the YAML file, so load them from its repository root.
        os.chdir(upstream)
        config = load_config("configs_3b/main.yaml")
    finally:
        os.chdir(previous_cwd)
    OmegaConf.set_readonly(config, False)
    config.dit.init_with_meta_device = True
    config.dit.model.norm = "rms"
    config.dit.model.qk_norm = "rms"
    config.dit.model.txt_in_norm = "layer"
    config.dit.model.vid_out_norm = "rms"
    config.vae.checkpoint = str(args.vae_checkpoint.resolve())
    config.vae.slicing.memory_device = "cpu"
    config.diffusion.cfg.scale = args.cfg_scale
    config.diffusion.cfg.rescale = 0.0
    config.diffusion.timesteps.sampling.steps = args.sample_steps

    init_torch(cudnn_benchmark=False)
    runner = VideoDiffusionInfer(config)
    if args.dit_checkpoint.suffix == ".safetensors":
        configure_safetensors_dit(runner, args.dit_checkpoint)
    else:
        runner.configure_dit_model(
            device="cpu", checkpoint=str(args.dit_checkpoint.resolve()))
    runner.configure_vae_model()
    if hasattr(runner.vae, "set_memory_limit"):
        runner.vae.set_memory_limit(**runner.config.vae.memory_limit)
    runner.configure_diffusion()
    return runner, get_device()


def save_frames(path: Path, sample: torch.Tensor, frame_count: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if sample.ndim == 3:
        sample = sample.unsqueeze(1)
    if sample.ndim != 4:
        raise ValueError(
            f"SeedVR2 output must be C,T,H,W (or C,H,W for T=1), got {sample.shape}")
    sample = sample[:, :frame_count]
    sample = sample.permute(1, 2, 3, 0).float().cpu()
    sample = sample.clamp(-1, 1).add_(1).mul_(127.5).round_().byte().numpy()
    for index, frame in enumerate(sample, start=1):
        Image.fromarray(frame).save(path / f"im{index:05d}.png")


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    process_started = time.perf_counter()
    for path in (
        args.dit_checkpoint,
        args.vae_checkpoint,
        args.positive_embedding,
        args.negative_embedding,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    input_paths, frames = load_frames(args.input_dir, args.max_frames)
    input_height = int(frames.shape[-2])
    input_width = int(frames.shape[-1])
    height = args.height or int(frames.shape[-2])
    width = args.width or int(frames.shape[-1])
    output_height = args.output_height or input_height
    output_width = args.output_width or input_width

    runner, device = configure_runner(args)
    dtype = getattr(torch, args.dit_dtype)
    if args.sequential_offload:
        runner.dit.to(device="cpu", dtype=dtype)
    else:
        runner.dit.to(device=device, dtype=dtype)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()

    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=dtype == torch.bfloat16,
    ):
        condition, original_length = pad_temporal(
            resize_and_normalize(frames, height, width, device))
        runner.vae.to(device)
        cond_latents = runner.vae_encode([condition])
        if args.sequential_offload:
            runner.vae.to("cpu")
            torch.cuda.empty_cache()
            runner.dit.to(device=device, dtype=dtype)
        noises = [torch.randn_like(cond_latents[0])]
        conditions = [runner.get_condition(
            noises[0], task="sr", latent_blur=cond_latents[0])]
        positive = torch.load(args.positive_embedding, map_location=device)
        negative = torch.load(args.negative_embedding, map_location=device)
        if args.sequential_offload:
            original_vae_decode = runner.vae_decode

            def release_dit_then_decode(latents):
                # This bridge processes exactly one clip.  Releasing the DiT
                # avoids upstream's unnecessary GPU -> CPU -> GPU round trip
                # before/after VAE decoding and leaves more room for pixels.
                runner.dit = None
                gc.collect()
                torch.cuda.empty_cache()
                return original_vae_decode(latents)

            runner.vae_decode = release_dit_then_decode
        outputs = runner.inference(
            noises=noises,
            conditions=conditions,
            texts_pos=[positive],
            texts_neg=[negative],
            cfg_scale=args.cfg_scale,
            dit_offload=False,
        )
        outputs = [
            resize_output(sample, output_height, output_width)
            for sample in outputs
        ]
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated(device))
    save_frames(args.output_dir, outputs[0], original_length)

    metadata = {
        "model": (
            "SeedVR2-3B community BF16 conversion of the released prototype checkpoint"
            if args.dit_checkpoint.suffix == ".safetensors"
            else "SeedVR2-3B official prototype weights"
        ),
        "bf16_checkpoint_source": (
            "https://huggingface.co/szwagros/SeedVR2-3B-bf16"
            if args.dit_checkpoint.suffix == ".safetensors"
            else None
        ),
        "dit_checkpoint": str(args.dit_checkpoint),
        "input_dir": str(args.input_dir),
        "input_frames": [str(path) for path in input_paths],
        "output_dir": str(args.output_dir),
        "frame_count": original_length,
        "input_height": input_height,
        "input_width": input_width,
        "processing_height": height,
        "processing_width": width,
        "output_height": output_height,
        "output_width": output_width,
        "seed": args.seed,
        "sample_steps": args.sample_steps,
        "cfg_scale": args.cfg_scale,
        "dit_dtype": args.dit_dtype,
        "sequential_offload": args.sequential_offload,
        "sequential_offload_strategy": (
            "one-shot DiT release before VAE decode"
            if args.sequential_offload else None
        ),
        "runtime_seconds": elapsed,
        "total_after_argument_parse_seconds": time.perf_counter() - process_started,
        "peak_cuda_allocated_bytes": peak,
        "model_load_in_timing": False,
        "input_and_output_png_lossless": True,
        "flash_attention_substitution": (
            "exact per-sequence PyTorch scaled_dot_product_attention"),
        "normalization_substitution": (
            "parameter-compatible PyTorch LayerNorm/RMSNorm; no Apex"),
        "training_or_finetuning": False,
        "color_fix": False,
    }
    metadata_path = args.output_dir / "seedvr2_metadata.json"
    atomic_json(metadata_path, metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("the local diagnostic bridge supports one GPU only")
    try:
        main()
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
