from __future__ import annotations

import torch
import torch.nn.functional as F

from tinytauk.vae.encoder import BigVGANEncoder, BigVGANEncoderConfig, _Conv1dS


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


def test_conv1ds_runs_weight_norm_hook_after_state_load() -> None:
    conv = _Conv1dS(1, 2, kernel_size=3).eval()
    state = conv.state_dict()
    state["layer.weight_g"] = torch.tensor([[[2.0]], [[3.0]]])
    state["layer.weight_v"] = torch.tensor(
        [
            [[1.0, 2.0, 3.0]],
            [[-2.0, 1.0, 4.0]],
        ]
    )
    state["layer.bias"] = torch.tensor([0.25, -0.5])
    conv.load_state_dict(state)

    inputs = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8)
    weight_g = state["layer.weight_g"]
    weight_v = state["layer.weight_v"]
    norm = torch.linalg.vector_norm(weight_v, dim=(1, 2), keepdim=True)
    expected_weight = weight_v * (weight_g / norm)
    expected = F.conv1d(inputs, expected_weight, state["layer.bias"], padding=1)

    actual = conv(inputs)

    assert torch.allclose(actual, expected)
