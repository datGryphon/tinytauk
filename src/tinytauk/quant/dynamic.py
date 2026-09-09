from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import torch
from torch import nn
from torch.ao.quantization import quantize_dynamic

_SENSITIVE_NAME_PARTS = (
    "time_embed",
    "txt_proj",
    "audio_embed",
    "attn_norm",
    "norm_out",
    "proj_out",
)


def is_sensitive_generator_linear(name: str) -> bool:
    """Return whether a Flux2 Linear is kept in FP32 by the conservative policy."""

    return any(part in name for part in _SENSITIVE_NAME_PARTS)


def select_dynamic_int8_linears(
    model: nn.Module,
    *,
    min_weight_elements: int = 1_000_000,
    include_sensitive: bool = False,
) -> tuple[str, ...]:
    """Select large Linear modules for dynamic INT8 CPU quantization."""

    if min_weight_elements < 1:
        raise ValueError("min_weight_elements must be positive")

    selected: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if module.weight.numel() < min_weight_elements:
            continue
        if not include_sensitive and is_sensitive_generator_linear(name):
            continue
        selected.append(name)
    return tuple(selected)


def configure_x86_quantized_engine() -> str:
    """Prefer PyTorch's x86 quantized backend, falling back to FBGEMM."""

    supported = tuple(str(engine) for engine in torch.backends.quantized.supported_engines)
    for candidate in ("x86", "fbgemm"):
        if candidate in supported:
            torch.backends.quantized.engine = candidate
            return candidate

    current = str(torch.backends.quantized.engine)
    if current == "none":
        raise RuntimeError(f"No x86 dynamic-quantization backend is available: {supported}")
    return current


def apply_dynamic_int8_linears(model: nn.Module, module_names: Iterable[str]) -> None:
    """Replace the selected Linear children with dynamic qint8 implementations in-place."""

    names = set(module_names)
    if not names:
        raise ValueError("No Linear modules selected for dynamic INT8 quantization")
    converted = cast(
        nn.Module,
        quantize_dynamic(  # type: ignore[no-untyped-call]
            model,
            qconfig_spec=names,
            dtype=torch.qint8,
            inplace=True,
        ),
    )
    if converted is not model:
        raise RuntimeError("Expected in-place dynamic quantization to preserve the model root")
