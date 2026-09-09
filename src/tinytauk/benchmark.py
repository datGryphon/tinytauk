from __future__ import annotations

import json
import platform
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    host: str
    profile: str
    target_seconds: float
    generated_seconds: float
    wall_seconds: float
    peak_rss_mb: float | None = None
    conditioning_seconds: float | None = None
    generation_seconds: float | None = None
    vae_seconds: float | None = None

    @property
    def rtf(self) -> float:
        if self.generated_seconds <= 0:
            raise ValueError("generated_seconds must be > 0")
        return self.wall_seconds / self.generated_seconds

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["rtf"] = self.rtf
        return value

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def dry_run_result(profile: str, target_seconds: float) -> BenchmarkResult:
    start = time.perf_counter()
    elapsed = max(time.perf_counter() - start, 1e-9)
    return BenchmarkResult(
        host=platform.node() or "unknown",
        profile=profile,
        target_seconds=target_seconds,
        generated_seconds=target_seconds,
        wall_seconds=elapsed,
    )


def benchmark_callable(fn: Callable[[], float], *, profile: str, target_seconds: float) -> BenchmarkResult:
    """Benchmark a callable that returns generated audio duration in seconds."""
    start = time.perf_counter()
    generated_seconds = fn()
    wall_seconds = time.perf_counter() - start
    return BenchmarkResult(
        host=platform.node() or "unknown",
        profile=profile,
        target_seconds=target_seconds,
        generated_seconds=generated_seconds,
        wall_seconds=wall_seconds,
    )
