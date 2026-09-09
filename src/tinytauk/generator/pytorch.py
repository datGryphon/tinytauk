from __future__ import annotations

from typing import Any

from tinytauk.config import ComponentConfig, ModelConfig
from tinytauk.types import Conditioning, GenerationRequest


class PyTorchAuKGenerator:
    """Checkpoint-compatible AuK-Flash generator boundary.

    The implementation will own Flux2Edit + CFM sampling without depending on
    the upstream `auk` package at production runtime.
    """

    def __init__(self, model: ModelConfig, config: ComponentConfig) -> None:
        self.model_config = model
        self.config = config

    def generate_latents(self, request: GenerationRequest, conditioning: Conditioning) -> Any:
        raise NotImplementedError("AuK-Flash generation is scheduled for Phase 2.")
