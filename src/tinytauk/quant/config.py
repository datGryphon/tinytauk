from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class QuantizationPolicy:
    """Declarative mixed-precision policy for Phase 4 experiments."""

    default: str = "none"
    overrides: dict[str, str] = field(default_factory=dict)
