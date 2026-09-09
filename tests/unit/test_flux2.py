from __future__ import annotations

import torch

from tinytauk.generator.flux2 import Flux2Edit


def test_flux2_instruction_only_shape() -> None:
    model = Flux2Edit(
        dim=64,
        heads=1,
        dim_head=64,
        ff_mult=2,
        latent_dim=8,
        text_hidden_dim=16,
        num_layers=1,
        num_single_layers=1,
        dropout=0.0,
        attn_mask_enabled=False,
    ).eval()

    output = model(
        x=torch.randn(1, 5, 8),
        text=torch.randn(1, 3, 16),
        time=torch.tensor(0.25),
        c_mask=torch.ones(1, 3, dtype=torch.bool),
        ref=torch.zeros(1, 0, 8),
        ref_mask=torch.zeros(1, 0, dtype=torch.bool),
    )

    assert output.shape == (1, 5, 8)
