from __future__ import annotations

from typing import Protocol

import torch

from tinytauk.types import Conditioning, GenerationRequest


class GeneratorBackend(Protocol):
    def generate_latents(
        self,
        request: GenerationRequest,
        conditioning: Conditioning,
    ) -> torch.Tensor:
        """Run the AuK-Flash four-step flow sampler and return VAE latents."""
        ...
