from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Conditioning:
    """Semantic conditioning produced by the Qwen/AuK layer-fusion stage."""

    values: Any
    attention_mask: Any | None = None


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    instruction: str
    reference_audio: Path | None = None
    gen_seconds: float = 10.0
    seed: int | None = None


@dataclass(slots=True)
class GenerationResult:
    audio: Any
    sample_rate: int
    generated_seconds: float
    wall_seconds: float
    stage_seconds: dict[str, float] = field(default_factory=dict)
