from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

type AudioInput = str | Path | tuple[torch.Tensor, int]


@dataclass(slots=True)
class Conditioning:
    """Reusable semantic and optional reference-audio conditioning."""

    values: Any
    attention_mask: Any | None = None
    instruction: str = ""
    seed: int | None = None
    reference_latents: Any | None = None
    reference_lengths: Any | None = None
    stage_seconds: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    instruction: str
    reference_audio: AudioInput | None = None
    gen_seconds: float = 10.0
    seed: int | None = None


@dataclass(slots=True)
class GenerationResult:
    audio: Any
    sample_rate: int
    generated_seconds: float
    wall_seconds: float
    stage_seconds: dict[str, float] = field(default_factory=dict)
