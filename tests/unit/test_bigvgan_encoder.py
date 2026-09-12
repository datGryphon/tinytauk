from __future__ import annotations

import torch

from tinytauk.vae.encoder import BigVGANEncoder, BigVGANEncoderConfig


def _tiny_config() -> BigVGANEncoderConfig:
    return BigVGANEncoderConfig(
        downsample_rates=[2],
        downsample_channels=[2, 4],
        latent_dim=2,
        use_vae=True,
    )


def test_encoder_produces_normalized_latents_and_lengths() -> None:
    encoder = BigVGANEncoder(_tiny_config()).eval()
    audio = torch.zeros(1, 1, 32)
    latents, lengths = encoder.encode(
        audio,
        sample_lengths=torch.tensor([32]),
        seed=42,
    )

    assert latents.ndim == 3
    assert latents.shape[0] == 1
    assert latents.shape[-1] == 2
    assert torch.equal(lengths, torch.tensor([16]))
    assert torch.isfinite(latents).all()


def test_encoder_seed_is_request_local_and_repeatable() -> None:
    encoder = BigVGANEncoder(_tiny_config()).eval()
    audio = torch.zeros(1, 1, 32)
    lengths = torch.tensor([32])

    first, _ = encoder.encode(audio, sample_lengths=lengths, seed=7)
    second, _ = encoder.encode(audio, sample_lengths=lengths, seed=7)
    third, _ = encoder.encode(audio, sample_lengths=lengths, seed=8)

    assert torch.equal(first, second)
    assert not torch.equal(first, third)
