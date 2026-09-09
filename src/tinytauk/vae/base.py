from __future__ import annotations

from typing import Any, Protocol


class VAEBackend(Protocol):
    sample_rate: int

    def decode(self, latents: Any) -> Any:
        """Decode AuK latents to waveform samples."""
        ...
