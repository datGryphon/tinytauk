from __future__ import annotations

from typing import Any

from tinytauk.config import ComponentConfig, ModelConfig


class PyTorchVAE:
    """BigVGANFlowVAE runtime boundary; intentionally unquantized in early phases."""

    sample_rate = 24_000

    def __init__(self, model: ModelConfig, config: ComponentConfig) -> None:
        self.model_config = model
        self.config = config

    def decode(self, latents: Any) -> Any:
        raise NotImplementedError("VAE decoding is scheduled for Phase 2.")
