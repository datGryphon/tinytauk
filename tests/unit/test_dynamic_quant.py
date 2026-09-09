from __future__ import annotations

import pytest
from torch import nn

from tinytauk.quant.dynamic import is_sensitive_generator_linear, select_dynamic_int8_linears


class _ToyGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.time_embed = nn.Linear(16, 16)
        self.attn = nn.Linear(16, 16)
        self.ff = nn.Linear(16, 32)
        self.attn_norm = nn.Module()
        self.attn_norm.linear = nn.Linear(16, 32)
        self.proj_out = nn.Linear(16, 16)
        self.small = nn.Linear(4, 4)


def test_sensitive_generator_linear_names() -> None:
    assert is_sensitive_generator_linear("time_embed.linear_1")
    assert is_sensitive_generator_linear("transformer_blocks.0.attn_norm.linear")
    assert is_sensitive_generator_linear("norm_out.linear")
    assert is_sensitive_generator_linear("proj_out")
    assert not is_sensitive_generator_linear("transformer_blocks.0.attn.to_qkv")
    assert not is_sensitive_generator_linear("single_transformer_blocks.0.ff.linear_in")


def test_select_dynamic_int8_linears_conservative() -> None:
    model = _ToyGenerator()
    selected = select_dynamic_int8_linears(model, min_weight_elements=100)
    assert set(selected) == {"attn", "ff"}


def test_select_dynamic_int8_linears_aggressive() -> None:
    model = _ToyGenerator()
    selected = select_dynamic_int8_linears(
        model,
        min_weight_elements=100,
        include_sensitive=True,
    )
    assert set(selected) == {"time_embed", "attn", "ff", "attn_norm.linear", "proj_out"}


def test_select_dynamic_int8_linears_rejects_invalid_threshold() -> None:
    model = _ToyGenerator()
    with pytest.raises(ValueError, match="positive"):
        select_dynamic_int8_linears(model, min_weight_elements=0)
