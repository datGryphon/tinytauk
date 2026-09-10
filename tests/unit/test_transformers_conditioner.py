from pathlib import Path

import torch

from tinytauk.conditioning.transformers import (
    TransformersConditioner,
    _find_tensor_key,
)
from tinytauk.config import ComponentConfig, ModelConfig
from tinytauk.types import GenerationRequest


def test_find_tensor_key_prefers_exact_match() -> None:
    assert _find_tensor_key(["foo.layer_weights", "layer_weights"], "layer_weights") == "layer_weights"


def test_find_tensor_key_accepts_unique_suffix() -> None:
    assert _find_tensor_key(["model.layer_scale"], "layer_scale") == "model.layer_scale"


def test_messages_adds_no_prompt_audio_marker() -> None:
    request = GenerationRequest(instruction="hello")
    messages = TransformersConditioner._messages(request)
    assert messages[0]["content"] == [{"type": "text", "text": "hello|<no_prompt_audio>|"}]


def test_messages_appends_reference_audio() -> None:
    request = GenerationRequest(
        instruction="hello",
        reference_audio=Path("voice.wav"),
    )
    messages = TransformersConditioner._messages(request)
    assert messages[0]["content"] == [
        {"type": "text", "text": "hello"},
        {"type": "audio", "audio": "voice.wav"},
    ]


def test_dtype_map_keeps_expected_torch_types() -> None:
    config = ComponentConfig(backend="transformers", dtype="bf16")
    model = ModelConfig()
    assert config.dtype == "bf16"
    assert model.qwen_model_id == "Qwen/Qwen2.5-Omni-3B"
    assert torch.bfloat16.is_floating_point
