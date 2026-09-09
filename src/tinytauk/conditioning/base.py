from __future__ import annotations

from typing import Protocol

from tinytauk.types import Conditioning, GenerationRequest


class ConditioningBackend(Protocol):
    def encode(self, request: GenerationRequest) -> Conditioning:
        """Return AuK-compatible semantic conditioning."""
        ...
