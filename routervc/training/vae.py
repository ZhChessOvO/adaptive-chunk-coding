"""Vae. Extracted from the validated RouterVC research implementation."""

from __future__ import annotations
import json
import os
import torch


def restore_step_log(path, completed):
    """Power-loss recovery: ignore only an incomplete final JSON write.

    Checkpointed steps must still be present; corruption in an earlier or
    newline-terminated record is an error, not silently discarded evidence.
    """
    lines = path.read_text().splitlines(keepends=True) if path.exists() else []
    records = {}
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1 and (not line.endswith("\n")):
                break
            raise
        if int(record["step"]) <= completed:
            if record["step"] in records:
                raise RuntimeError("duplicate checkpointed step record")
            records[record["step"]] = record
    if sorted(records) != list(range(1, completed + 1)):
        raise RuntimeError("missing checkpointed step record")
    ordered = [records[i] for i in range(1, completed + 1)]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text("".join((json.dumps(r) + "\n" for r in ordered)))
    os.replace(temporary, path)
    return ordered


def differentiable_decode(runner, latent):
    value = latent.to(torch.bfloat16).permute(3, 0, 1, 2).unsqueeze(0)
    value = value / float(runner.config.vae.scaling_factor)
    value = value + float(runner.config.vae.get("shifting_factor", 0.0))
    output = runner.vae.decode(value).sample
    if hasattr(runner.vae, "postprocess"):
        output = runner.vae.postprocess(output)
    return output[0].permute(1, 0, 2, 3).float()
