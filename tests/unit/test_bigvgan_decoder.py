from __future__ import annotations

import torch

from tinytauk.vae.bigvgan import BigVGANDecoder, BigVGANDecoderConfig, SnakeBeta, _kaiser_sinc_filter1d


def test_decoder_config_ignores_encoder_only_fields() -> None:
    config = BigVGANDecoderConfig.from_dict(
        {
            "latent_dim": 32,
            "causal": False,
            "flow_hidden_channels": 256,
            "downsample_rates": [2, 2],
        }
    )
    assert config.latent_dim == 32
    assert config.causal is False


def test_kaiser_filter_is_normalized() -> None:
    kernel = _kaiser_sinc_filter1d(cutoff=0.25, half_width=0.3, kernel_size=12)
    assert kernel.shape == (1, 1, 12)
    assert torch.allclose(kernel.sum(), torch.tensor(1.0), atol=1e-6, rtol=0)


def test_snake_beta_preserves_shape() -> None:
    activation = SnakeBeta(4, alpha_logscale=True)
    value = torch.zeros(2, 4, 8)
    assert activation(value).shape == value.shape


def test_remove_weight_norm_preserves_decoder_output() -> None:
    config = BigVGANDecoderConfig(
        upsample_rates=[2],
        upsample_kernel_sizes=[4],
        upsample_initial_channel=8,
        resblock_kernel_sizes=[3],
        resblock_dilation_sizes=[[1]],
        latent_dim=2,
        causal=True,
    )
    decoder = BigVGANDecoder(config).eval()
    latents = torch.randn(1, 2, 4)

    before = decoder(latents)
    assert hasattr(decoder.conv_pre, "weight_g")

    decoder.remove_weight_norm()
    after = decoder(latents)

    assert not hasattr(decoder.conv_pre, "weight_g")
    assert torch.allclose(before, after, atol=1e-6, rtol=0)
