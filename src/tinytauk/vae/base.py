from __future__ import annotations

from typing import Protocol

import torch


class VAEBackend(Protocol):
    sample_rate: int

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode AuK latents to waveform samples."""
        ...
