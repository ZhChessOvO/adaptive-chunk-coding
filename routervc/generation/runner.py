"""Runner. Extracted from the validated RouterVC research implementation."""

from __future__ import annotations
import argparse
import gc
import importlib.machinery
import os
import sys
import types
from pathlib import Path
import torch
import torch.nn.functional as F


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
            q_part = q[q_offsets[index] : q_offsets[index + 1]]
            k_part = k[k_offsets[index] : k_offsets[index + 1]]
            v_part = v[k_offsets[index] : k_offsets[index + 1]]
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


def install_compact_mmrope() -> None:
    """Build exact multimodal RoPE prefixes at the requested clip shape.

    Upstream allocates a fixed 1024x128x128 frequency grid and then slices a
    tiny prefix from it.  The language-style axial coordinates are integer
    aranges, so constructing only the largest requested prefix is numerically
    identical while avoiding a roughly 4 GiB final grid and much larger
    broadcast/cat allocation peak.
    """
    from models.dit_v2.rope import NaMMRotaryEmbedding3d

    if getattr(NaMMRotaryEmbedding3d, "_adaptive_chunk_compact", False):
        return

    def compact_get_freqs(self, vid_shape, txt_shape):
        video_shapes = [tuple(map(int, shape)) for shape in vid_shape.tolist()]
        text_lengths = [int(value) for value in txt_shape[:, 0].tolist()]
        if not video_shapes or len(video_shapes) != len(text_lengths):
            raise RuntimeError("SeedVR2 RoPE received inconsistent batch shapes")
        max_t = max(
            (length + shape[0] for shape, length in zip(video_shapes, text_lengths))
        )
        max_h = max((shape[1] for shape in video_shapes))
        max_w = max((shape[2] for shape in video_shapes))
        max_text = max(text_lengths)
        vid_freqs = self.get_axial_freqs(max_t, max_h, max_w)
        txt_freqs = self.get_axial_freqs(max_text)
        video_values = []
        text_values = []
        for (frames, height, width), length in zip(video_shapes, text_lengths):
            video_values.append(
                vid_freqs[length : length + frames, :height, :width].reshape(
                    -1, vid_freqs.size(-1)
                )
            )
            text_values.append(
                txt_freqs[:length].repeat(1, 3).reshape(-1, vid_freqs.size(-1))
            )
        return (torch.cat(video_values, dim=0), torch.cat(text_values, dim=0))

    NaMMRotaryEmbedding3d.get_freqs = compact_get_freqs
    NaMMRotaryEmbedding3d._adaptive_chunk_compact = True


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
            f"SeedVR2 output must be C,T,H,W (or C,H,W for T=1), got {sample.shape}"
        )
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
        return (sample, original_length)
    padding = 4 - (original_length - 1) % 4
    tail = sample[:, -1:].expand(-1, padding, -1, -1)
    return (torch.cat([sample, tail], dim=1), original_length)


def configure_safetensors_dit(runner, checkpoint: Path) -> None:
    """Mirror upstream meta-device loading for a safetensors checkpoint."""
    from common.config import create_object
    from common.distributed.meta_init_utils import meta_non_persistent_buffer_init_fn
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
        (
            parameter.numel()
            for parameter in runner.dit.parameters()
            if parameter.requires_grad
        )
    )
    print(f"DiT trainable parameters: {num_params:,}")


def configure_runner(args: argparse.Namespace):
    upstream = args.upstream_root.resolve()
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    install_flash_attention_fallback()
    from common.config import load_config
    from common.distributed import get_device
    from common.distributed import init_torch
    from omegaconf import OmegaConf
    from projects.video_diffusion_sr.infer import VideoDiffusionInfer

    install_compact_mmrope()
    previous_cwd = Path.cwd()
    try:
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
    # Upstream assumes torchrun. The public RouterVC entrypoints are single-GPU
    # Python commands, so initialize a private local rendezvous when necessary.
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("this release supports one GPU/process only")
    if not torch.distributed.is_initialized():
        import tempfile

        fd, rendezvous = tempfile.mkstemp(prefix="routervc-dist-")
        os.close(fd)
        torch.cuda.set_device(0)
        torch.distributed.init_process_group(
            backend="nccl", init_method="file://" + rendezvous, rank=0, world_size=1
        )
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    runner = VideoDiffusionInfer(config)
    if args.dit_checkpoint.suffix == ".safetensors":
        configure_safetensors_dit(runner, args.dit_checkpoint)
    else:
        runner.configure_dit_model(
            device="cpu", checkpoint=str(args.dit_checkpoint.resolve())
        )
    runner.configure_vae_model()
    if hasattr(runner.vae, "set_memory_limit"):
        runner.vae.set_memory_limit(**runner.config.vae.memory_limit)
    runner.configure_diffusion()
    runner.lora_adapter_info = None
    lora_checkpoint = getattr(args, "lora_checkpoint", None)
    if lora_checkpoint is not None:
        from routervc.generation.lora import load_lora_adapter

        lora_checkpoint = Path(lora_checkpoint).resolve()
        if not lora_checkpoint.is_file():
            raise FileNotFoundError(lora_checkpoint)
        runner.lora_adapter_info = load_lora_adapter(
            runner.dit,
            lora_checkpoint,
            trainable=False,
            strength=float(getattr(args, "lora_strength", 1.0)),
        )
    return (runner, get_device())
