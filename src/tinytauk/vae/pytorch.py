from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
from safetensors.torch import load_file

from tinytauk.config import ComponentConfig, ModelConfig

from .bigvgan import BigVGANDecoder, BigVGANDecoderConfig


class PyTorchVAE:
    """Standalone decoder-only BigVGAN VAE path for AuK latents."""

    def __init__(
        self,
        model: ModelConfig,
        config: ComponentConfig,
        *,
        model_snapshot: str | Path | None = None,
    ) -> None:
        if config.quantization != "none":
            raise ValueError("VAE parity does not quantize yet")
        if config.dtype != "fp32":
            raise ValueError("VAE parity currently requires fp32")

        self.model_config = model
        self.config = config
        self.device = torch.device(config.device)
        self.snapshot = Path(model_snapshot or snapshot_download(repo_id=model.model_id))

        config_path = self.snapshot / "config.yaml"
        checkpoint_path = self.snapshot / "vae.safetensors"
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)

        raw_config = OmegaConf.load(config_path)
        vae_config = raw_config.model.vae
        self.sample_rate = int(vae_config.target_sample_rate)
        self.downsample_rate = int(vae_config.downsample_rate)
        self.latent_dim = int(vae_config.latent_dim)

        model_init = cast(
            dict[str, Any],
            OmegaConf.to_container(
                vae_config.get("model_init_kwargs", OmegaConf.create({})),
                resolve=True,
            ),
        )
        decoder_config = BigVGANDecoderConfig.from_dict(model_init)
        if decoder_config.latent_dim != self.latent_dim:
            raise RuntimeError(
                "VAE config latent dimension mismatch: "
                f"decoder={decoder_config.latent_dim} model={self.latent_dim}"
            )

        self.decoder = BigVGANDecoder(decoder_config)
        self._load_decoder_weights(checkpoint_path)
        self.decoder = self.decoder.to(device=self.device, dtype=torch.float32).eval()
        self.decoder.requires_grad_(False)

    def _load_decoder_weights(self, checkpoint_path: Path) -> None:
        checkpoint = load_file(str(checkpoint_path), device="cpu")
        expected = self.decoder.state_dict()
        state = {key: value for key, value in checkpoint.items() if key in expected}
        missing, unexpected = self.decoder.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "BigVGAN decoder checkpoint mismatch: "
                f"missing={missing[:10]} unexpected={unexpected[:10]}"
            )

    @torch.inference_mode()
    def decode(self, latents: Any) -> torch.Tensor:
        if not isinstance(latents, torch.Tensor):
            raise TypeError("latents must be a torch.Tensor")
        if latents.ndim != 3:
            raise ValueError(f"latents must have shape [B, T, D], got {tuple(latents.shape)}")
        if latents.shape[-1] != self.latent_dim:
            raise ValueError(
                f"latent feature dimension must be {self.latent_dim}, got {latents.shape[-1]}"
            )

        value = latents.to(device=self.device, dtype=torch.float32)
        value = self.decoder.denormalize(value)
        value = value.permute(0, 2, 1)
        decoded = cast(torch.Tensor, self.decoder(value))
        return decoded.to(torch.float32)
