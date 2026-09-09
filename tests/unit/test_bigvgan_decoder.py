from __future__ import annotations

import torch

from tinytauk.vae.bigvgan import BigVGANDecoderConfig, SnakeBeta, _kaiser_sinc_filter1d


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
