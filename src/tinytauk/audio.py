from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio  # type: ignore[import-untyped]

from .types import AudioInput

_QWEN_SAMPLE_RATE = 16_000


def _coerce_tensor_audio(audio: torch.Tensor, sample_rate: int) -> tuple[torch.Tensor, int]:
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError(f"audio sample rate must be a positive integer, got {sample_rate!r}")

    value = audio.detach().to(device="cpu", dtype=torch.float32)
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2:
        raise ValueError(f"audio tensor must have shape [channels, samples], got {tuple(value.shape)}")
    if value.shape[-1] == 0:
        raise ValueError("audio tensor is empty")
    if not torch.isfinite(value).all():
        raise ValueError("audio tensor contains NaN or Inf")
    if value.shape[0] > 1:
        value = value.mean(dim=0, keepdim=True)
    return value, sample_rate


def load_audio(source: AudioInput, *, target_sample_rate: int) -> torch.Tensor:
    """Load, validate, mono-mix, and resample audio to ``[1, samples]``."""

    if isinstance(source, tuple):
        audio, sample_rate = _coerce_tensor_audio(source[0], source[1])
    else:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(path)
        loaded, sample_rate = torchaudio.load(str(path))
        audio, sample_rate = _coerce_tensor_audio(loaded, int(sample_rate))

    if sample_rate != target_sample_rate:
        audio = torchaudio.functional.resample(audio, sample_rate, target_sample_rate)
    return audio.contiguous()


def qwen_audio_value(source: AudioInput) -> Any:
    """Return a qwen-omni-utils-compatible audio value.

    Paths are left as paths so qwen-omni-utils owns its normal 16 kHz loading
    path. In-memory tensors are converted to a mono 16 kHz NumPy waveform.
    """

    if not isinstance(source, tuple):
        return str(Path(source))

    audio, sample_rate = _coerce_tensor_audio(source[0], source[1])
    if sample_rate != _QWEN_SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sample_rate, _QWEN_SAMPLE_RATE)
    return np.asarray(audio.squeeze(0).contiguous().numpy(), dtype=np.float32)
