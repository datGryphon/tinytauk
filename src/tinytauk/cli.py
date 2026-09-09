from __future__ import annotations

import argparse
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from . import __version__
from .benchmark import dry_run_result
from .config import RuntimeConfig


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
        "torchao": _optional_package_version("torchao"),
    }

    try:
        import torch

        report["torch"] = torch.__version__
        report["torch_threads"] = torch.get_num_threads()
        report["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        report["torch"] = None

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _show_config(path: Path) -> int:
    print(json.dumps(RuntimeConfig.from_toml(path).to_dict(), indent=2, sort_keys=True))
    return 0


def _benchmark(profile: Path, target_seconds: float, dry_run: bool) -> int:
    config = RuntimeConfig.from_toml(profile)
    if not dry_run:
        raise SystemExit(
            "Real inference benchmarking is not implemented in v0. Use --dry-run to validate the harness."
        )
    _ = config
    print(dry_run_result(str(profile), target_seconds).to_json())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tinytauk")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="Report runtime/platform capabilities")

    config = sub.add_parser("config", help="Resolve and print a runtime profile")
    config.add_argument("path", type=Path)

    benchmark = sub.add_parser("benchmark", help="Run the benchmark harness")
    benchmark.add_argument("--profile", type=Path, default=Path("profiles/cpu-baseline.toml"))
    benchmark.add_argument("--target-seconds", type=float, default=10.0)
    benchmark.add_argument("--dry-run", action="store_true")

    generate = sub.add_parser("generate", help="Generate speech (Phase 2; not implemented in v0)")
    generate.add_argument("instruction")
    generate.add_argument("--reference-audio", type=Path)
    generate.add_argument("--seconds", type=float, default=10.0)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "doctor":
        return _doctor()
    if args.command == "config":
        return _show_config(args.path)
    if args.command == "benchmark":
        return _benchmark(args.profile, args.target_seconds, args.dry_run)
    if args.command == "generate":
        print("TinyTAuK v0 does not implement model execution yet.", file=sys.stderr)
        return 2

    raise AssertionError(f"Unhandled command: {args.command}")
