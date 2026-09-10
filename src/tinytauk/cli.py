from __future__ import annotations

import argparse
import json
import platform
import wave
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch

from . import __version__
from .config import RuntimeConfig
from .engine import TinyTAuK


def _optional_package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _doctor() -> int:
    report: dict[str, object] = {
        "tinytauk": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "torchao": _optional_package_version("torchao"),
        "cuda_available": torch.cuda.is_available(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _show_config(path: Path) -> int:
    print(json.dumps(RuntimeConfig.from_toml(path).to_dict(), indent=2, sort_keys=True))
    return 0


def _write_pcm16(path: Path, audio: torch.Tensor, sample_rate: int) -> None:
    samples = audio.squeeze(0).detach().cpu().to(torch.float32).clamp(-1.0, 1.0)
    pcm = (samples * 32767.0).round().to(torch.int16).contiguous().numpy().astype("<i2", copy=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def _generate(
    profile: Path | None,
    instruction: str,
    output: Path,
    seconds: float,
    seed: int | None,
) -> int:
    engine = TinyTAuK.from_config(profile) if profile is not None else TinyTAuK.from_pretrained()
    result = engine.generate(
        instruction,
        gen_seconds=seconds,
        seed=seed,
    )
    _write_pcm16(output, result.audio, result.sample_rate)
    print(
        json.dumps(
            {
                "output": str(output),
                "sample_rate": result.sample_rate,
                "generated_seconds": result.generated_seconds,
                "wall_seconds": result.wall_seconds,
                "stage_seconds": result.stage_seconds,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tinytauk")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="Report runtime/platform capabilities")

    config = sub.add_parser("config", help="Resolve and print a runtime profile")
    config.add_argument("path", type=Path)

    generate = sub.add_parser("generate", help="Generate speech with AuK-Flash")
    generate.add_argument("instruction")
    generate.add_argument("--profile", type=Path)
    generate.add_argument("--output", type=Path, default=Path("output.wav"))
    generate.add_argument("--seconds", type=float, default=10.0)
    generate.add_argument("--seed", type=int)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "doctor":
        return _doctor()
    if args.command == "config":
        return _show_config(args.path)
    if args.command == "generate":
        return _generate(
            args.profile,
            args.instruction,
            args.output,
            args.seconds,
            args.seed,
        )

    raise AssertionError(f"Unhandled command: {args.command}")
