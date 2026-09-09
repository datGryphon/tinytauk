from __future__ import annotations

from pathlib import Path

from .config import RuntimeConfig
from .types import GenerationRequest, GenerationResult


class TinyTAuK:
    """High-level TinyTAuK inference engine.

    v0 deliberately stops at the stable API/config boundary. Model execution is
    added only after upstream reference fixtures are captured.
    """

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config

    @classmethod
    def from_config(cls, config: RuntimeConfig | str | Path) -> TinyTAuK:
        if isinstance(config, (str, Path)):
            config = RuntimeConfig.from_toml(config)
        return cls(config)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "tencent/AuK-Flash",
        *,
        qwen_model_id: str = "Qwen/Qwen2.5-Omni-3B",
        device: str = "cpu",
    ) -> TinyTAuK:
        raw = {
            "model": {"model_id": model_id, "qwen_model_id": qwen_model_id},
            "conditioner": {"device": device},
            "generator": {"device": device},
            "vae": {"device": device},
        }
        return cls(RuntimeConfig.from_dict(raw))

    def generate(
        self,
        instruction: str,
        *,
        reference_audio: str | Path | None = None,
        gen_seconds: float = 10.0,
        seed: int | None = None,
    ) -> GenerationResult:
        _ = GenerationRequest(
            instruction=instruction,
            reference_audio=Path(reference_audio) if reference_audio else None,
            gen_seconds=gen_seconds,
            seed=seed,
        )
        raise NotImplementedError(
            "TinyTAuK v0 is a scaffold. Implement reference parity before enabling generation."
        )
