"""Lightweight Linux process memory sampling for benchmark scripts."""

from __future__ import annotations

import resource
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MemorySample:
    current_rss_mb: float
    peak_rss_mb: float


def memory_sample() -> MemorySample:
    current_kb = 0.0
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                current_kb = float(line.split()[1])
                break
    peak_kb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return MemorySample(current_rss_mb=current_kb / 1024.0, peak_rss_mb=peak_kb / 1024.0)
