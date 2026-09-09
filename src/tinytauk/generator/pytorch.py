from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import torch
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
from safetensors.torch import load_file

from tinytauk.config import ComponentConfig, ModelConfig
from tinytauk.types import Conditioning, GenerationRequest

from .flux2 import Flux2Edit

_FLASH_T_GRID = [0.0, 0.07612049579620361, 0.2928932309150696, 0.6173166036605835, 1.0]
_DTYPE_MAP: dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


class PyTorchAuKGenerator:
    """Standalone AuK-Flash Flux2Edit + four-step CPU sampler.

    The current baseline supports instruction-only, batch-one generation.
    Reference-audio/editing support is added after the text-only CPU path is
    qualified end to end.
    """

    def __init__(
        self,
        model: ModelConfig,
        config: ComponentConfig,
        *,
        model_snapshot: str | Path | None = None,
    ) -> None:
        if config.quantization != "none":
            raise ValueError("CPU generator baseline does not quantize yet")

        self.model_config = model
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = _DTYPE_MAP[config.dtype]
        self.snapshot = Path(model_snapshot or snapshot_download(repo_id=model.model_id))

        config_path = self.snapshot / "config.yaml"
        checkpoint_path = self.snapshot / "auk_flash.safetensors"
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        if not checkpoint_path.is_file():
            candidates = [
                path
                for path in self.snapshot.glob("*.safetensors")
                if "vae" not in path.name.lower()
            ]
            if not candidates:
                raise FileNotFoundError("No AuK transformer checkpoint found")
            checkpoint_path = max(candidates, key=lambda path: path.stat().st_size)

        raw_config = OmegaConf.load(config_path)
        model_config = raw_config.model
        if str(model_config.get("name", "")) != "AuK-Flash":
            raise ValueError("PyTorchAuKGenerator currently supports only AuK-Flash")

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

    def _load_transformer_weights(self, checkpoint_path: Path) -> None:
        checkpoint = load_file(str(checkpoint_path), device="cpu")
        prefix = "transformer."
        state = {
            key.removeprefix(prefix): value
            for key, value in checkpoint.items()
            if key.startswith(prefix)
        }
        if not state:
            raise RuntimeError("AuK checkpoint contains no transformer.* tensors")
        missing, unexpected = self.transformer.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "Flux2Edit checkpoint mismatch: "
                f"missing={missing[:10]} unexpected={unexpected[:10]}"
            )

    @torch.inference_mode()
    def generate_latents(
        self,
        request: GenerationRequest,
        conditioning: Conditioning,
    ) -> torch.Tensor:
        if request.reference_audio is not None:
            raise NotImplementedError("reference-audio generation is not in the CPU baseline yet")
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
        torch.manual_seed(seed)
        # Mirror CFMEdit.sample exactly: draw one [duration, channels] tensor,
        # then add the batch dimension. This removes RNG-layout ambiguity from
        # the exact FP32 parity gate.
        latent = torch.randn(
            (target_len, self.latent_dim),
            device=self.device,
            dtype=self.dtype,
        ).unsqueeze(0)
        text = conditioning.values.to(device=self.device, dtype=self.dtype)
        context_mask = (
            conditioning.attention_mask.to(self.device)
            if conditioning.attention_mask is not None
            else None
        )
        empty_ref = torch.zeros(
            (1, 0, self.latent_dim),
            device=self.device,
            dtype=self.dtype,
        )
        empty_ref_mask = torch.zeros((1, 0), device=self.device, dtype=torch.bool)

        # FP32 preserves exact oracle parity. Lower-precision CPU baselines keep
        # the time grid in the model dtype so Linear inputs and latent updates do
        # not silently promote back to FP32 outside an autocast region.
        times = torch.tensor(_FLASH_T_GRID, device=self.device, dtype=self.dtype)
        for index in range(len(_FLASH_T_GRID) - 1):
            velocity = self.transformer(
                x=latent,
                text=text,
                time=times[index],
                mask=None,
                c_mask=context_mask,
                ref=empty_ref,
                ref_mask=empty_ref_mask,
                drop_audio_cond=False,
                drop_text=False,
                cfg_infer=False,
                cache=True,
            )
            latent = latent + (times[index + 1] - times[index]) * velocity

        self.transformer.clear_cache()
        return latent
