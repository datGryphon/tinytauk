from __future__ import annotations

from pathlib import Path
from threading import Lock

import torch

from tinytauk import RuntimeConfig, TinyTAuK
from tinytauk.types import Conditioning, GenerationRequest


class _Conditioner:
    def encode(self, request: GenerationRequest) -> Conditioning:
        assert request.instruction == "Test"
        assert request.gen_seconds == 2.0
        assert request.seed == 42
        return Conditioning(
            values=torch.zeros((1, 1, 2048), dtype=torch.float32),
            attention_mask=torch.ones((1, 1), dtype=torch.bool),
        )


class _Generator:
    def generate_latents(
        self,
        request: GenerationRequest,
        conditioning: Conditioning,
    ) -> torch.Tensor:
        assert request.instruction == "Test"
        assert isinstance(conditioning.values, torch.Tensor)
        return torch.zeros((1, 100, 64), dtype=torch.float32)


class _VAE:
    sample_rate = 24_000

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        assert latents.shape == (1, 100, 64)
        return torch.zeros((1, 1, 48_000), dtype=torch.float32)


class _ConfigOnlyTinyTAuK(TinyTAuK):
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config


def _fake_engine() -> TinyTAuK:
    engine = object.__new__(TinyTAuK)
    engine.config = RuntimeConfig.from_dict({"runtime": {"seed": 42}})
    engine._generate_lock = Lock()
    engine.conditioner = _Conditioner()  # type: ignore[assignment]
    engine.generator = _Generator()  # type: ignore[assignment]
    engine.vae = _VAE()  # type: ignore[assignment]
    return engine


def test_generate_wires_standalone_components() -> None:
    result = _fake_engine().generate("Test", gen_seconds=2.0)

    assert result.audio.shape == (1, 48_000)
    assert result.sample_rate == 24_000
    assert result.generated_seconds == 2.0
    assert set(result.stage_seconds) == {"conditioning", "generator", "vae"}
    assert result.wall_seconds >= 0.0


def test_generate_rejects_invalid_requests_before_execution() -> None:
    engine = _fake_engine()

    for instruction, gen_seconds, match in (
        ("   ", 2.0, "instruction must not be empty"),
        ("Test", 0.0, "gen_seconds must be a positive finite number"),
        ("Test", float("nan"), "gen_seconds must be a positive finite number"),
    ):
        try:
            engine.generate(instruction, gen_seconds=gen_seconds)
        except ValueError as exc:
            assert match in str(exc)
        else:
            raise AssertionError(f"expected ValueError matching {match!r}")


def test_generate_rejects_reference_audio_before_execution() -> None:
    engine = _fake_engine()

    try:
        engine.generate("Test", reference_audio="voice.wav", gen_seconds=2.0)
    except NotImplementedError as exc:
        assert "reference-audio generation is not implemented" in str(exc)
    else:
        raise AssertionError("expected reference audio to fail before execution")


def test_from_pretrained_matches_release_cpu_profile() -> None:
    engine = _ConfigOnlyTinyTAuK.from_pretrained()

    assert engine.config == RuntimeConfig.from_toml(Path("profiles/cpu.toml"))
