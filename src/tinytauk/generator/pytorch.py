from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import torch
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
from safetensors.torch import load_file

from tinytauk.config import ComponentConfig, ModelConfig
from tinytauk.quant.dynamic import (
    apply_dynamic_int8_linears,
    configure_x86_quantized_engine,
    select_dynamic_int8_linears,
)
from tinytauk.types import Conditioning, GenerationRequest

from .flux2 import Flux2Edit

_FLASH_T_GRID = [0.0, 0.07612049579620361, 0.2928932309150696, 0.6173166036605835, 1.0]
_DTYPE_MAP: dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


class PyTorchAuKGenerator:
    """AuK-Flash Flux2Edit transformer and four-step sampler."""

    def __init__(
        self,
        model: ModelConfig,
        config: ComponentConfig,
        *,
        model_snapshot: str | Path | None = None,
    ) -> None:
        if config.quantization not in {"none", "int8"}:
            raise ValueError("PyTorch AuK generator supports only none or int8 quantization")
        if config.compile:
            raise ValueError("PyTorch AuK generator compilation is not supported")

        self.model_config = model
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = _DTYPE_MAP[config.dtype]
        self.snapshot = Path(model_snapshot or snapshot_download(repo_id=model.model_id))
        self.quantized_linear_names: tuple[str, ...] = ()
        self.quantized_engine: str | None = None

        config_path = self.snapshot / "config.yaml"
        checkpoint_path = self.snapshot / "auk_flash.safetensors"
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        if not checkpoint_path.is_file():
            candidates = [
                path for path in self.snapshot.glob("*.safetensors") if "vae" not in path.name.lower()
            ]
            if not candidates:
                raise FileNotFoundError("No AuK transformer checkpoint found")
            checkpoint_path = max(candidates, key=lambda path: path.stat().st_size)

        raw_config = OmegaConf.load(config_path)
        model_config = raw_config.model
        if str(model_config.get("name", "")) != "AuK-Flash":
            raise ValueError("PyTorchAuKGenerator supports only AuK-Flash")

        vae_config = model_config.vae
        self.target_sample_rate = int(vae_config.target_sample_rate)
        self.downsample_rate = int(vae_config.downsample_rate)
        self.latent_dim = int(vae_config.latent_dim)

        arch = cast(dict[str, Any], OmegaConf.to_container(model_config.arch, resolve=True))
        arch["attn_backend"] = "torch"
        arch["checkpoint_activations"] = False
        self.transformer = Flux2Edit(**arch, latent_dim=self.latent_dim)
        self._load_transformer_weights(checkpoint_path)
        self.transformer = self.transformer.to(device=self.device, dtype=self.dtype).eval()
        self.transformer.requires_grad_(False)

        if config.quantization == "int8":
            if self.device.type != "cpu" or self.dtype != torch.float32:
                raise ValueError("dynamic INT8 generator requires CPU FP32 input weights")
            self.quantized_linear_names = select_dynamic_int8_linears(
                self.transformer,
                min_weight_elements=1_000_000,
                include_sensitive=False,
            )
            self.quantized_engine = configure_x86_quantized_engine()
            apply_dynamic_int8_linears(self.transformer, self.quantized_linear_names)

    def _load_transformer_weights(self, checkpoint_path: Path) -> None:
        checkpoint = load_file(str(checkpoint_path), device="cpu")
        prefix = "transformer."
        state = {
            key.removeprefix(prefix): value for key, value in checkpoint.items() if key.startswith(prefix)
        }
        if not state:
            raise RuntimeError("AuK checkpoint contains no transformer.* tensors")
        missing, unexpected = self.transformer.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Flux2Edit checkpoint mismatch: missing={missing[:10]} unexpected={unexpected[:10]}"
            )

    def _reference_inputs(self, conditioning: Conditioning) -> tuple[torch.Tensor, torch.Tensor]:
        if conditioning.reference_latents is None:
            return (
                torch.zeros((1, 0, self.latent_dim), device=self.device, dtype=self.dtype),
                torch.zeros((1, 0), device=self.device, dtype=torch.bool),
            )
        if not isinstance(conditioning.reference_latents, torch.Tensor):
            raise TypeError("conditioning.reference_latents must be a torch.Tensor")
        if conditioning.reference_lengths is None or not isinstance(
            conditioning.reference_lengths, torch.Tensor
        ):
            raise TypeError("reference conditioning requires tensor reference_lengths")

        reference = conditioning.reference_latents.to(device=self.device, dtype=self.dtype)
        if reference.ndim != 3 or reference.shape[0] != 1 or reference.shape[-1] != self.latent_dim:
            raise ValueError(
                "reference_latents must have shape [1, T, "
                f"{self.latent_dim}], got {tuple(reference.shape)}"
            )
        lengths = conditioning.reference_lengths.to(device=self.device, dtype=torch.long)
        if lengths.shape != (1,):
            raise ValueError(f"reference_lengths must have shape [1], got {tuple(lengths.shape)}")
        if int(lengths[0]) < 0 or int(lengths[0]) > reference.shape[1]:
            raise ValueError("reference_lengths is outside the reference latent sequence")
        positions = torch.arange(reference.shape[1], device=self.device).unsqueeze(0)
        return reference, positions < lengths.unsqueeze(1)

    @torch.inference_mode()
    def generate_latents(
        self,
        request: GenerationRequest,
        conditioning: Conditioning,
    ) -> torch.Tensor:
        if not isinstance(conditioning.values, torch.Tensor):
            raise TypeError("conditioning.values must be a torch.Tensor")
        if conditioning.attention_mask is not None and not isinstance(
            conditioning.attention_mask, torch.Tensor
        ):
            raise TypeError("conditioning.attention_mask must be a torch.Tensor")

        target_len = max(
            1,
            int(math.ceil(request.gen_seconds * self.target_sample_rate / self.downsample_rate)),
        )
        seed = request.seed if request.seed is not None else 1234
        rng = torch.Generator(device=self.device)
        rng.manual_seed(seed)
        latent = torch.randn(
            (target_len, self.latent_dim),
            generator=rng,
            device=self.device,
            dtype=self.dtype,
        ).unsqueeze(0)
        text = conditioning.values.to(device=self.device, dtype=self.dtype)
        context_mask = (
            conditioning.attention_mask.to(self.device) if conditioning.attention_mask is not None else None
        )
        reference, reference_mask = self._reference_inputs(conditioning)

        times = torch.tensor(_FLASH_T_GRID, device=self.device, dtype=self.dtype)
        try:
            for index in range(len(_FLASH_T_GRID) - 1):
                velocity = self.transformer(
                    x=latent,
                    text=text,
                    time=times[index],
                    mask=None,
                    c_mask=context_mask,
                    ref=reference,
                    ref_mask=reference_mask,
                    drop_audio_cond=False,
                    drop_text=False,
                    cfg_infer=False,
                    cache=True,
                )
                latent = latent + (times[index + 1] - times[index]) * velocity
            return latent
        finally:
            self.transformer.clear_cache()
