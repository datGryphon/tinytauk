from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

type AudioInput = str | Path | tuple[torch.Tensor, int]


@dataclass(slots=True)
class Conditioning:
    """Reusable semantic and optional reference-audio conditioning."""

    values: torch.Tensor
    attention_mask: torch.Tensor | None = None
    instruction: str = ""
    seed: int | None = None
    reference_latents: torch.Tensor | None = None
    reference_lengths: torch.Tensor | None = None
    stage_seconds: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    instruction: str
    reference_audio: AudioInput | None = None
    gen_seconds: float = 10.0
    seed: int | None = None


@dataclass(slots=True)
class GenerationResult:
    audio: torch.Tensor
    sample_rate: int
    generated_seconds: float
    wall_seconds: float
    stage_seconds: dict[str, float] = field(default_factory=dict)
