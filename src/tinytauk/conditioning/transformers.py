from __future__ import annotations

from tinytauk.config import ComponentConfig, ModelConfig
from tinytauk.types import Conditioning, GenerationRequest


class TransformersConditioner:
    """Qwen2.5-Omni conditioner implemented with Hugging Face Transformers.

    Phase 2 will load only the Thinker components required by AuK and explicitly
    discard the unused vision tower. Phase 3 replaces all-hidden-state stacking
    with mathematically equivalent streaming AuK layer fusion.
    """

    def __init__(self, model: ModelConfig, config: ComponentConfig) -> None:
        self.model_config = model
        self.config = config

    def encode(self, request: GenerationRequest) -> Conditioning:
        raise NotImplementedError(
            "Transformers conditioning is scheduled for Phase 2/3; v0 only defines the runtime boundary."
        )
