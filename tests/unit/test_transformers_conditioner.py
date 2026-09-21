from pathlib import Path

import numpy as np
import torch

from tinytauk.conditioning.transformers import TransformersConditioner, _find_tensor_key
from tinytauk.types import GenerationRequest


def test_find_tensor_key_prefers_exact_match() -> None:
    assert _find_tensor_key(["foo.layer_weights", "layer_weights"], "layer_weights") == "layer_weights"


def test_find_tensor_key_accepts_unique_suffix() -> None:
    assert _find_tensor_key(["model.layer_scale"], "layer_scale") == "model.layer_scale"


def test_messages_adds_no_prompt_audio_marker() -> None:
    request = GenerationRequest(instruction="hello")
    messages = TransformersConditioner._messages(request)
    assert messages[0]["content"] == [{"type": "text", "text": "hello|<no_prompt_audio>|"}]


def test_messages_appends_reference_audio_path() -> None:
    request = GenerationRequest(
        instruction="hello",
        reference_audio=Path("voice.wav"),
    )
    messages = TransformersConditioner._messages(request)
    assert messages[0]["content"] == [
        {"type": "text", "text": "hello"},
        {"type": "audio", "audio": "voice.wav"},
    ]


def test_messages_accepts_reference_audio_tensor() -> None:
    request = GenerationRequest(
        instruction="hello",
        reference_audio=(torch.zeros(16_000), 16_000),
    )
    messages = TransformersConditioner._messages(request)
    audio = messages[0]["content"][1]["audio"]

    assert isinstance(audio, np.ndarray)
    assert audio.shape == (16_000,)
