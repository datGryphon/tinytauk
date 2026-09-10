from __future__ import annotations

import math
import time
from pathlib import Path
from threading import Lock

import torch
from huggingface_hub import snapshot_download

from .conditioning.transformers import TransformersConditioner
from .config import RuntimeConfig
from .generator.pytorch import PyTorchAuKGenerator
from .types import GenerationRequest, GenerationResult
from .vae.pytorch import PyTorchVAE


class TinyTAuK:
    """High-level standalone TinyTAuK inference engine."""

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self._validate_backends()
        self._generate_lock = Lock()

        if config.runtime.num_threads > 0:
            torch.set_num_threads(config.runtime.num_threads)

        self.load_stage_seconds: dict[str, float] = {}

        started = time.perf_counter()
        snapshot_started = time.perf_counter()
        self.snapshot = Path(snapshot_download(repo_id=config.model.model_id))
        checkpoint_path = self._find_transformer_checkpoint(self.snapshot)
        self.load_stage_seconds["snapshot"] = time.perf_counter() - snapshot_started

        component_started = time.perf_counter()
        self.conditioner = TransformersConditioner(
            config.model,
            config.conditioner,
            auk_checkpoint=checkpoint_path,
        )
        self.load_stage_seconds["conditioner"] = time.perf_counter() - component_started

        component_started = time.perf_counter()
        self.generator = PyTorchAuKGenerator(
            config.model,
            config.generator,
            model_snapshot=self.snapshot,
        )
        self.load_stage_seconds["generator"] = time.perf_counter() - component_started

        component_started = time.perf_counter()
        self.vae = PyTorchVAE(
            config.model,
            config.vae,
            model_snapshot=self.snapshot,
        )
        self.load_stage_seconds["vae"] = time.perf_counter() - component_started
        self.load_seconds = time.perf_counter() - started

    def _validate_backends(self) -> None:
        supported = {
            "conditioner": (self.config.conditioner.backend, "transformers"),
            "generator": (self.config.generator.backend, "pytorch"),
            "vae": (self.config.vae.backend, "pytorch"),
        }
        for name, (actual, expected) in supported.items():
            if actual != expected:
                raise ValueError(f"unsupported {name} backend {actual!r}; expected {expected!r}")

    @staticmethod
    def _find_transformer_checkpoint(snapshot: Path) -> Path:
        preferred = snapshot / "auk_flash.safetensors"
        if preferred.is_file():
            return preferred

        candidates = [
            path for path in snapshot.glob("*.safetensors") if "vae" not in path.name.lower()
        ]
        if not candidates:
            raise FileNotFoundError(f"No AuK transformer checkpoint found in {snapshot}")
        return max(candidates, key=lambda path: path.stat().st_size)

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
        if device != "cpu":
            raise ValueError(
                "from_pretrained currently supports only CPU; use from_config for custom runtimes"
            )
        raw = {
            "model": {"model_id": model_id, "qwen_model_id": qwen_model_id},
            "conditioner": {
                "backend": "transformers",
                "device": "cpu",
                "dtype": "fp32",
                "quantization": "int8-weight-only",
            },
            "generator": {
                "backend": "pytorch",
                "device": "cpu",
                "dtype": "fp32",
                "quantization": "int8",
            },
            "vae": {
                "backend": "pytorch",
                "device": "cpu",
                "dtype": "fp32",
                "quantization": "none",
                "compile": True,
                "compile_mode": "default",
                "compile_dynamic": True,
            },
            "runtime": {"seed": 1234, "num_threads": 4},
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
        if not instruction.strip():
            raise ValueError("instruction must not be empty")
        if not math.isfinite(gen_seconds) or gen_seconds <= 0:
            raise ValueError("gen_seconds must be a positive finite number")
        if reference_audio is not None:
            raise NotImplementedError("reference-audio generation is not implemented")

        request = GenerationRequest(
            instruction=instruction,
            gen_seconds=gen_seconds,
            seed=self.config.runtime.seed if seed is None else seed,
        )

        with self._generate_lock:
            started = time.perf_counter()
            stage_seconds: dict[str, float] = {}

            stage_started = time.perf_counter()
            conditioning = self.conditioner.encode(request)
            stage_seconds["conditioning"] = time.perf_counter() - stage_started

            stage_started = time.perf_counter()
            latents = self.generator.generate_latents(request, conditioning)
            stage_seconds["generator"] = time.perf_counter() - stage_started

            stage_started = time.perf_counter()
            decoded = self.vae.decode(latents)
            stage_seconds["vae"] = time.perf_counter() - stage_started

            audio = decoded.squeeze(0).detach().cpu().to(torch.float32)
            generated_seconds = float(audio.shape[-1]) / float(self.vae.sample_rate)
            return GenerationResult(
                audio=audio,
                sample_rate=self.vae.sample_rate,
                generated_seconds=generated_seconds,
                wall_seconds=time.perf_counter() - started,
                stage_seconds=stage_seconds,
            )
