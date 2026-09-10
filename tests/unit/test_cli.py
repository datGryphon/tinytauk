from __future__ import annotations

import wave
from pathlib import Path

import torch

from tinytauk.cli import _write_pcm16, build_parser


def test_generate_parser_uses_built_in_cpu_defaults() -> None:
    args = build_parser().parse_args(["generate", "Test instruction"])

    assert args.profile is None
    assert args.output == Path("output.wav")
    assert args.seconds == 10.0
    assert args.seed is None


def test_write_pcm16(tmp_path: Path) -> None:
    output = tmp_path / "test.wav"
    audio = torch.tensor([[0.0, 0.5, -0.5]], dtype=torch.float32)

    _write_pcm16(output, audio, 24_000)

    with wave.open(str(output), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 24_000
        assert handle.getnframes() == 3
