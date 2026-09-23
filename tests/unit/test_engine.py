from __future__ import annotations

from pathlib import Path
from threading import Lock

import torch

from tinytauk import RuntimeConfig, TinyTAuK
from tinytauk.types import AudioInput, Conditioning, GenerationRequest


class _Conditioner:
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, request: GenerationRequest) -> Conditioning:
        self.calls += 1
        assert request.instruction == "Test"
        assert request.seed == 42
        return Conditioning(
            values=torch.zeros((1, 1, 2048), dtype=torch.float32),
            attention_mask=torch.ones((1, 1), dtype=torch.bool),
            instruction=request.instruction,
        )


class _Generator:
    def __init__(self) -> None:
        self.calls = 0

    def generate_latents(
        self,
        request: GenerationRequest,
        conditioning: Conditioning,
    ) -> torch.Tensor:
        self.calls += 1
        assert request.instruction == "Test"
        assert request.gen_seconds == 2.0
        assert isinstance(conditioning.values, torch.Tensor)
        return torch.zeros((1, 100, 64), dtype=torch.float32)


class _VAE:
    sample_rate = 24_000

    def __init__(self) -> None:
        self.reference_calls = 0

    def encode_reference(
        self,
        source: AudioInput,
        *,
        seed: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.reference_calls += 1
        assert source == "voice.wav"
        assert seed == 42
        return (
            torch.zeros((1, 12, 64), dtype=torch.float32),
            torch.tensor([12], dtype=torch.long),
        )

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


def test_conditioned_generation_reuses_conditioning() -> None:
    engine = _fake_engine()
    conditioning = engine.condition("Test")

    first = engine.generate_conditioned(conditioning, gen_seconds=2.0, seed=42)
    second = engine.generate_conditioned(conditioning, gen_seconds=2.0, seed=42)

    assert engine.conditioner.calls == 1  # type: ignore[attr-defined]
    assert engine.generator.calls == 2  # type: ignore[attr-defined]
    assert first.audio.shape == second.audio.shape == (1, 48_000)
    assert set(first.stage_seconds) == {"generator", "vae"}


def test_reference_audio_is_encoded_once_into_conditioning() -> None:
    engine = _fake_engine()
    conditioning = engine.condition("Test", reference_audio="voice.wav")

    assert engine.vae.reference_calls == 1  # type: ignore[attr-defined]
    assert isinstance(conditioning.reference_latents, torch.Tensor)
    assert conditioning.reference_latents.shape == (1, 12, 64)
    assert torch.equal(conditioning.reference_lengths, torch.tensor([12]))
    assert set(conditioning.stage_seconds) == {"conditioning", "reference_vae"}

    engine.generate_conditioned(conditioning, gen_seconds=2.0)
    engine.generate_conditioned(conditioning, gen_seconds=2.0, seed=43)
    assert engine.vae.reference_calls == 1  # type: ignore[attr-defined]


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


def test_generate_accepts_reference_audio() -> None:
    engine = _fake_engine()
    result = engine.generate("Test", reference_audio="voice.wav", gen_seconds=2.0)

    assert result.audio.shape == (1, 48_000)
    assert engine.vae.reference_calls == 1  # type: ignore[attr-defined]
    assert set(result.stage_seconds) == {"conditioning", "reference_vae", "generator", "vae"}


def test_from_pretrained_matches_release_cpu_profile() -> None:
    engine = _ConfigOnlyTinyTAuK.from_pretrained()

    assert engine.config == RuntimeConfig.from_toml(Path("profiles/cpu.toml"))
