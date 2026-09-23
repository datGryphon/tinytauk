from __future__ import annotations

import numpy as np
import pytest
import torch

from tinytauk.audio import load_audio, qwen_audio_value


def test_load_audio_mixes_channels_and_resamples() -> None:
    stereo = torch.stack([torch.ones(8_000), torch.zeros(8_000)])
    audio = load_audio((stereo, 8_000), target_sample_rate=16_000)

    assert audio.shape == (1, 16_000)
    assert torch.isfinite(audio).all()
    assert torch.allclose(audio.mean(), torch.tensor(0.5), atol=1e-3, rtol=0)


def test_qwen_audio_value_converts_tensor_to_16khz_numpy() -> None:
    value = qwen_audio_value((torch.zeros(8_000), 8_000))

    assert isinstance(value, np.ndarray)
    assert value.dtype == np.float32
    assert value.shape == (16_000,)


def test_audio_tensor_validation() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        load_audio((torch.zeros(4), 0), target_sample_rate=24_000)
    with pytest.raises(ValueError, match="empty"):
        load_audio((torch.zeros(0), 24_000), target_sample_rate=24_000)
    with pytest.raises(ValueError, match="NaN or Inf"):
        load_audio((torch.tensor([float("nan")]), 24_000), target_sample_rate=24_000)
