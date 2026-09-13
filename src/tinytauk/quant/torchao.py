from __future__ import annotations

from collections.abc import Iterable

from torch import nn
from torchao.quantization import Int4WeightOnlyConfig, Int8WeightOnlyConfig, quantize_


def apply_weight_only_linears(
    model: nn.Module,
    module_names: Iterable[str],
    *,
    bits: int,
) -> None:
    """Apply TorchAO weight-only quantization to selected Linear modules."""

    names = frozenset(module_names)
    if not names:
        raise ValueError("No Linear modules selected for weight-only quantization")

    if bits == 8:
        config = Int8WeightOnlyConfig(version=2)
    elif bits == 4:
        config = Int4WeightOnlyConfig(group_size=128, version=2)
    else:
        raise ValueError(f"unsupported weight-only bit width: {bits}")

    quantize_(
        model,
        config,
        filter_fn=lambda _module, fqn: fqn in names,
    )
