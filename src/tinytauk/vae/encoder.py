"""Reference-audio encoder path from AuK's BigVGANFlowVAE.

Adapted from Tencent-Hunyuan/AuK and its BigVGAN/HiFi-GAN lineage under MIT.
See THIRD_PARTY.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Any

import torch
from torch import nn
from torch.nn.utils import weight_norm


@dataclass
class BigVGANEncoderConfig:
    downsample_rates: list[int] = field(default_factory=lambda: [2, 2, 2, 3, 4, 5])
    downsample_channels: list[int] = field(default_factory=lambda: [12, 24, 48, 96, 192, 384, 768])
    latent_dim: int = 64
    use_vae: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> BigVGANEncoderConfig:
        valid = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in raw.items() if key in valid})


def _tensor_call(module: nn.Module, value: torch.Tensor) -> torch.Tensor:
    result = module(value)
    if not isinstance(result, torch.Tensor):
        raise TypeError(f"{module.__class__.__name__} returned a non-tensor value")
    return result


class _TensorSequential(nn.Sequential):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for module in self:
            value = _tensor_call(module, value)
        return value


class _Conv1dS(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.layer = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        weight_norm(self.layer)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layer(inputs)


class _ResStack(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation_base: int = 2,
        count: int = 6,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _TensorSequential(
                    nn.LeakyReLU(),
                    _weight_norm_conv(
                        channels,
                        kernel_size=kernel_size,
                        dilation=dilation_base**index,
                    ),
                    nn.LeakyReLU(),
                    _weight_norm_conv(channels, kernel_size=kernel_size, dilation=1),
                )
                for index in range(count)
            ]
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            value = value + _tensor_call(layer, value)
        return value


def _weight_norm_conv(channels: int, *, kernel_size: int, dilation: int) -> nn.Conv1d:
    layer = nn.Conv1d(
        channels,
        channels,
        kernel_size=kernel_size,
        dilation=dilation,
        padding=dilation,
    )
    weight_norm(layer)
    return layer


class _Encoder(nn.Module):
    def __init__(self, config: BigVGANEncoderConfig) -> None:
        super().__init__()
        channels = config.downsample_channels
        if len(channels) != len(config.downsample_rates) + 1:
            raise ValueError("downsample_channels must contain one more entry than downsample_rates")
        if not channels:
            raise ValueError("downsample_channels must not be empty")

        out_channels = config.latent_dim * 2 if config.use_vae else config.latent_dim
        layers: list[nn.Module] = [
            _Conv1dS(1, channels[0], kernel_size=3, stride=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for (in_channels, out_channels_stage), factor in zip(
            zip(channels[:-1], channels[1:], strict=True),
            config.downsample_rates,
            strict=True,
        ):
            layers.extend(
                [
                    _Conv1dS(
                        in_channels,
                        out_channels_stage,
                        kernel_size=factor * 2,
                        stride=factor,
                    ),
                    _ResStack(out_channels_stage),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
        layers.append(_Conv1dS(channels[-1], out_channels, kernel_size=3, stride=1))
        self.generator = _TensorSequential(*layers)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.generator(audio)


class BigVGANEncoder(nn.Module):
    """The reference-audio half of AuK's BigVGANFlowVAE."""

    def __init__(self, config: BigVGANEncoderConfig) -> None:
        super().__init__()
        if not config.use_vae:
            raise ValueError("AuK reference encoding requires use_vae=true")
        self.config = config
        self.hop_size = math.prod(config.downsample_rates)
        self.register_buffer("global_mean", torch.zeros(config.latent_dim, dtype=torch.float32))
        self.register_buffer("global_log_std", torch.ones(config.latent_dim, dtype=torch.float32))
        self.audio_encoder = _Encoder(config)

    @torch.inference_mode()
    def encode(
        self,
        audio: torch.Tensor,
        *,
        sample_lengths: torch.Tensor,
        seed: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if audio.ndim != 3 or audio.shape[1] != 1:
            raise ValueError(f"audio must have shape [B, 1, T], got {tuple(audio.shape)}")
        if sample_lengths.shape != (audio.shape[0],):
            raise ValueError(
                f"sample_lengths must have shape [{audio.shape[0]}], got {tuple(sample_lengths.shape)}"
            )

        latent_stats = self.audio_encoder(audio.float())
        mean, log_std = latent_stats.chunk(2, dim=1)
        if seed is None:
            noise = torch.randn_like(mean)
        else:
            generator = torch.Generator(device=mean.device)
            generator.manual_seed(seed)
            noise = torch.randn(
                mean.shape,
                generator=generator,
                device=mean.device,
                dtype=mean.dtype,
            )
        latents = mean + noise * torch.exp(log_std)
        latents = latents.transpose(1, 2).float()
        global_mean = self.get_buffer("global_mean").float()
        global_log_std = self.get_buffer("global_log_std").float()
        latents = (latents - global_mean) / torch.sqrt(global_log_std)
        latent_lengths = sample_lengths // self.hop_size
        latent_lengths = torch.clamp(latent_lengths, max=latents.shape[1])
        return latents, latent_lengths
