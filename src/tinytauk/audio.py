from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torchaudio
from numpy.typing import NDArray

from .types import AudioInput

_QWEN_SAMPLE_RATE = 16_000

type QwenAudioValue = str | NDArray[np.float32]


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


def _resample(audio: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
    result = torchaudio.functional.resample(audio, source_rate, target_rate)
    if not isinstance(result, torch.Tensor):
        raise TypeError("torchaudio resample returned a non-tensor value")
    return result


def load_audio(source: AudioInput, *, target_sample_rate: int) -> torch.Tensor:
    """Load, validate, mono-mix, and resample audio to ``[1, samples]``."""

    if isinstance(source, tuple):
        audio, sample_rate = _coerce_tensor_audio(source[0], source[1])
    else:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(path)
        loaded, sample_rate = torchaudio.load(str(path))
        if not isinstance(loaded, torch.Tensor):
            raise TypeError("torchaudio load returned a non-tensor waveform")
        audio, sample_rate = _coerce_tensor_audio(loaded, int(sample_rate))

    if sample_rate != target_sample_rate:
        audio = _resample(audio, sample_rate, target_sample_rate)
    return audio.contiguous()


def qwen_audio_value(source: AudioInput) -> QwenAudioValue:
    """Return a qwen-omni-utils-compatible path or mono 16 kHz waveform."""

    if not isinstance(source, tuple):
        return str(Path(source))

    audio, sample_rate = _coerce_tensor_audio(source[0], source[1])
    if sample_rate != _QWEN_SAMPLE_RATE:
        audio = _resample(audio, sample_rate, _QWEN_SAMPLE_RATE)
    return np.asarray(audio.squeeze(0).contiguous().numpy(), dtype=np.float32)
