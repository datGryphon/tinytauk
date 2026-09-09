from __future__ import annotations

from typing import Any

from .config import QuantizationPolicy


def apply_torchao_quantization(model: Any, policy: QuantizationPolicy) -> Any:
    """Apply a future TorchAO policy without importing torchao at module import time."""
    raise NotImplementedError("TorchAO quantization is scheduled for Phase 4.")
