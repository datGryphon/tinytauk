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

    filter_fn = lambda _module, fqn: fqn in names
    if bits == 8:
        quantize_(model, Int8WeightOnlyConfig(version=2), filter_fn=filter_fn)
        return
    if bits == 4:
        quantize_(model, Int4WeightOnlyConfig(group_size=128, version=2), filter_fn=filter_fn)
        return
    raise ValueError(f"unsupported weight-only bit width: {bits}")
