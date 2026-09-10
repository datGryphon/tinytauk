"""Decoder-only BigVGAN path used by AuK's VAE.

Adapted from Tencent-Hunyuan/AuK's BigVGANFlowVAE decoder. The alias-free
activation/resampling code is a modified adaptation of alias-free-torch under
Apache-2.0. See THIRD_PARTY.md.
"""
# mypy: ignore-errors

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import remove_weight_norm, weight_norm


@dataclass
class BigVGANDecoderConfig:
    upsample_rates: list[int] = field(default_factory=lambda: [5, 4, 3, 2, 2, 2])
    upsample_kernel_sizes: list[int] = field(default_factory=lambda: [10, 8, 6, 4, 4, 4])
    upsample_initial_channel: int = 1536
    resblock_kernel_sizes: list[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: list[list[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )
    latent_dim: int = 64
    causal: bool = True
    snake_logscale: bool = True
    act_causal: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> BigVGANDecoderConfig:
        valid = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in raw.items() if key in valid})


def _init_weights(module: nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    if "Conv" in module.__class__.__name__ and hasattr(module, "weight"):
        module.weight.data.normal_(mean, std)


def _get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


class Conv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        *,
        padding: int | None = None,
        causal: bool = False,
        bias: bool = True,
    ) -> None:
        self.causal = causal
        self.left_padding = dilation * (kernel_size - 1) if causal else 0
        if padding is None:
            padding = 0 if causal else _get_padding(kernel_size, dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.causal:
            x = F.pad(x, (self.left_padding, 0))
        return super().forward(x)


class ConvTranspose1d(nn.ConvTranspose1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        *,
        causal: bool = False,
        bias: bool = True,
    ) -> None:
        padding = 0 if causal else (kernel_size - stride) // 2
        if causal and kernel_size != 2 * stride:
            raise ValueError("causal ConvTranspose1d requires kernel_size == 2 * stride")
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.causal = causal
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        return x[..., : -self.stride] if self.causal else x


class SnakeBeta(nn.Module):
    def __init__(self, channels: int, *, alpha_logscale: bool = False) -> None:
        super().__init__()
        init = torch.zeros(channels) if alpha_logscale else torch.ones(channels)
        self.alpha = nn.Parameter(init.clone())
        self.beta = nn.Parameter(init.clone())
        self.alpha_logscale = alpha_logscale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha[None, :, None]
        beta = self.beta[None, :, None]
        if self.alpha_logscale:
            alpha = alpha.exp()
            beta = beta.exp()
        return x + torch.sin(x * alpha).pow(2) / (beta + 1e-9)


def _kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int) -> torch.Tensor:
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2
    delta_f = 4 * half_width
    attenuation = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = 0.5842 * (attenuation - 21) ** 0.4 + 0.07886 * (attenuation - 21.0)
    else:
        beta = 0.0

    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    time = (
        torch.arange(-half_size, half_size, dtype=window.dtype) + 0.5
        if even
        else torch.arange(kernel_size, dtype=window.dtype) - half_size
    )
    if cutoff == 0:
        kernel = torch.zeros_like(time)
    else:
        kernel = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
        kernel /= kernel.sum()
    return kernel.view(1, 1, kernel_size)


class LowPassFilter1d(nn.Module):
    def __init__(
        self,
        *,
        cutoff: float,
        half_width: float,
        stride: int,
        kernel_size: int,
        causal: bool,
    ) -> None:
        super().__init__()
        if causal:
            self.pad_left = kernel_size - 1
            self.pad_right = 0
        else:
            even = kernel_size % 2 == 0
            self.pad_left = kernel_size // 2 - int(even)
            self.pad_right = kernel_size // 2
        self.stride = stride
        self.register_buffer("filter", _kaiser_sinc_filter1d(cutoff, half_width, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        x = F.pad(x, (self.pad_left, self.pad_right), mode="replicate")
        return F.conv1d(x, self.filter.expand(channels, -1, -1), stride=self.stride, groups=channels)


class UpSample1d(nn.Module):
    def __init__(self, ratio: int = 2, kernel_size: int = 12) -> None:
        super().__init__()
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.stride = ratio
        self.pad = kernel_size // ratio - 1
        self.pad_left = self.pad * ratio + (kernel_size - ratio) // 2
        self.pad_right = self.pad * ratio + (kernel_size - ratio + 1) // 2
        self.register_buffer(
            "filter",
            _kaiser_sinc_filter1d(
                cutoff=0.5 / ratio,
                half_width=0.6 / ratio,
                kernel_size=kernel_size,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(
            x,
            self.filter.expand(channels, -1, -1),
            stride=self.stride,
            groups=channels,
        )
        return x[..., self.pad_left : -self.pad_right]


class DownSample1d(nn.Module):
    def __init__(self, ratio: int = 2, kernel_size: int = 12, *, causal: bool = False) -> None:
        super().__init__()
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=kernel_size,
            causal=causal,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lowpass(x)


class Activation1d(nn.Module):
    def __init__(self, activation: nn.Module, *, causal: bool = False) -> None:
        super().__init__()
        self.act = activation
        self.upsample = UpSample1d(2, 12)
        self.downsample = DownSample1d(2, 12, causal=causal)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(self.act(self.upsample(x)))


class AMPBlock1(nn.Module):
    def __init__(
        self,
        config: BigVGANDecoderConfig,
        channels: int,
        kernel_size: int,
        dilation: list[int],
    ) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        dilation=value,
                        causal=config.causal,
                    )
                )
                for value in dilation
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                weight_norm(Conv1d(channels, channels, kernel_size, dilation=1, causal=config.causal))
                for _ in dilation
            ]
        )
        self.convs1.apply(_init_weights)
        self.convs2.apply(_init_weights)
        self.activations = nn.ModuleList(
            [
                Activation1d(
                    SnakeBeta(channels, alpha_logscale=config.snake_logscale),
                    causal=config.act_causal,
                )
                for _ in range(len(self.convs1) + len(self.convs2))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv1, conv2, act1, act2 in zip(
            self.convs1,
            self.convs2,
            self.activations[::2],
            self.activations[1::2],
            strict=True,
        ):
            residual = conv2(act2(conv1(act1(x))))
            x = x + residual
        return x

    def remove_weight_norm(self) -> None:
        for layer in self.convs1:
            remove_weight_norm(layer)
        for layer in self.convs2:
            remove_weight_norm(layer)


class BigVGANDecoder(nn.Module):
    def __init__(self, config: BigVGANDecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("global_mean", torch.zeros(config.latent_dim, dtype=torch.float32))
        self.register_buffer("global_log_std", torch.ones(config.latent_dim, dtype=torch.float32))

        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)
        self.conv_pre = weight_norm(
            Conv1d(config.latent_dim, config.upsample_initial_channel, 7, causal=False)
        )

        self.ups = nn.ModuleList()
        for index, (rate, kernel) in enumerate(
            zip(config.upsample_rates, config.upsample_kernel_sizes, strict=True)
        ):
            self.ups.append(
                nn.ModuleList(
                    [
                        weight_norm(
                            ConvTranspose1d(
                                config.upsample_initial_channel // (2**index),
                                config.upsample_initial_channel // (2 ** (index + 1)),
                                kernel,
                                rate,
                                causal=config.causal,
                            )
                        )
                    ]
                )
            )

        self.resblocks = nn.ModuleList()
        final_channels = 0
        for index in range(len(self.ups)):
            final_channels = config.upsample_initial_channel // (2 ** (index + 1))
            for kernel, dilation in zip(
                config.resblock_kernel_sizes,
                config.resblock_dilation_sizes,
                strict=True,
            ):
                self.resblocks.append(AMPBlock1(config, final_channels, kernel, dilation))

        self.activation_post = Activation1d(
            SnakeBeta(final_channels, alpha_logscale=config.snake_logscale),
            causal=config.act_causal,
        )
        self.conv_post = weight_norm(Conv1d(final_channels, 1, 7, causal=config.causal, bias=False))
        for stage in self.ups:
            stage.apply(_init_weights)
        self.conv_post.apply(_init_weights)

    def denormalize(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.float()
        return latents * torch.sqrt(self.global_log_std.float()) + self.global_mean.float()

    def remove_weight_norm(self) -> None:
        """Materialize weight-normalized convolutions for inference."""
        remove_weight_norm(self.conv_pre)
        for stage in self.ups:
            for layer in stage:
                remove_weight_norm(layer)
        for block in self.resblocks:
            block.remove_weight_norm()
        remove_weight_norm(self.conv_post)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        x = self.conv_pre(latents)
        for index, stage in enumerate(self.ups):
            for upsample in stage:
                x = upsample(x)
            combined: torch.Tensor | None = None
            for offset in range(self.num_kernels):
                value = self.resblocks[index * self.num_kernels + offset](x)
                combined = value if combined is None else combined + value
            if combined is None:
                raise RuntimeError("BigVGAN decoder stage has no residual blocks")
            x = combined / self.num_kernels
        x = self.conv_post(self.activation_post(x))
        return torch.clamp(x, min=-1.0, max=1.0)
