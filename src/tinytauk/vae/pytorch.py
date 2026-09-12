from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import torch
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn

from tinytauk.audio import load_audio
from tinytauk.config import ComponentConfig, ModelConfig
from tinytauk.types import AudioInput

from .bigvgan import BigVGANDecoder, BigVGANDecoderConfig
from .encoder import BigVGANEncoder, BigVGANEncoderConfig


class _DecodeGraph(nn.Module):
    def __init__(self, decoder: BigVGANDecoder) -> None:
        super().__init__()
        self.decoder = decoder

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        value = latents.float()
        value = self.decoder.denormalize(value)
        value = value.permute(0, 2, 1)
        return cast(torch.Tensor, self.decoder(value))


class PyTorchVAE:
    """BigVGAN decode path plus lazily loaded reference-audio encoder."""

    def __init__(
        self,
        model: ModelConfig,
        config: ComponentConfig,
        *,
        model_snapshot: str | Path | None = None,
    ) -> None:
        if config.quantization != "none":
            raise ValueError("VAE quantization is not supported")
        if config.dtype != "fp32":
            raise ValueError("VAE currently requires fp32")

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
        self._checkpoint_path = checkpoint_path

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
        self._encoder_config = BigVGANEncoderConfig.from_dict(model_init)
        if self._encoder_config.latent_dim != self.latent_dim:
            raise RuntimeError(
                "VAE config latent dimension mismatch: "
                f"encoder={self._encoder_config.latent_dim} model={self.latent_dim}"
            )
        if self.downsample_rate != math.prod(self._encoder_config.downsample_rates):
            raise RuntimeError(
                "VAE downsample rate mismatch: "
                f"config={self.downsample_rate} encoder={self._encoder_config.downsample_rates}"
            )

        self.decoder = BigVGANDecoder(decoder_config)
        self._load_decoder_weights(checkpoint_path)
        self.decoder.remove_weight_norm()
        self.decoder = self.decoder.to(device=self.device, dtype=torch.float32).eval()
        self.decoder.requires_grad_(False)
        self._decode_graph: nn.Module | None = None
        self._encoder: BigVGANEncoder | None = None

        if config.compile:
            graph = _DecodeGraph(self.decoder).eval()
            compile_kwargs: dict[str, Any] = {
                "backend": "inductor",
                "dynamic": config.compile_dynamic,
            }
            if config.compile_mode != "default":
                compile_kwargs["mode"] = config.compile_mode
            self._decode_graph = cast(nn.Module, torch.compile(graph, **compile_kwargs))

    def _load_decoder_weights(self, checkpoint_path: Path) -> None:
        checkpoint = load_file(str(checkpoint_path), device="cpu")
        expected = self.decoder.state_dict()
        state = {key: value for key, value in checkpoint.items() if key in expected}
        missing, unexpected = self.decoder.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"BigVGAN decoder checkpoint mismatch: missing={missing[:10]} unexpected={unexpected[:10]}"
            )

    def _get_encoder(self) -> BigVGANEncoder:
        if self._encoder is not None:
            return self._encoder

        encoder = BigVGANEncoder(self._encoder_config)
        expected = encoder.state_dict()
        with safe_open(str(self._checkpoint_path), framework="pt", device="cpu") as checkpoint:
            available = set(checkpoint.keys())
            state = {key: checkpoint.get_tensor(key) for key in expected if key in available}
        missing, unexpected = encoder.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"BigVGAN encoder checkpoint mismatch: missing={missing[:10]} unexpected={unexpected[:10]}"
            )
        encoder = encoder.to(device=self.device, dtype=torch.float32).eval()
        encoder.requires_grad_(False)
        self._encoder = encoder
        return encoder

    @torch.inference_mode()
    def encode_reference(
        self,
        source: AudioInput,
        *,
        seed: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        audio = load_audio(source, target_sample_rate=self.sample_rate)
        audio = audio.to(device=self.device, dtype=torch.float32).unsqueeze(0)
        sample_lengths = torch.tensor([audio.shape[-1]], device=self.device, dtype=torch.long)
        encoder = self._get_encoder()
        latents, lengths = encoder.encode(audio, sample_lengths=sample_lengths, seed=seed)
        return latents, lengths

    @torch.inference_mode()
    def decode(self, latents: Any) -> torch.Tensor:
        if not isinstance(latents, torch.Tensor):
            raise TypeError("latents must be a torch.Tensor")
        if latents.ndim != 3:
            raise ValueError(f"latents must have shape [B, T, D], got {tuple(latents.shape)}")
        if latents.shape[-1] != self.latent_dim:
            raise ValueError(f"latent feature dimension must be {self.latent_dim}, got {latents.shape[-1]}")

        value = latents.to(device=self.device, dtype=torch.float32)
        if self._decode_graph is not None:
            decoded = cast(torch.Tensor, self._decode_graph(value))
            return decoded.to(torch.float32)

        value = self.decoder.denormalize(value)
        value = value.permute(0, 2, 1)
        decoded = cast(torch.Tensor, self.decoder(value))
        return decoded.to(torch.float32)
