from __future__ import annotations

import torch

from tinytauk.generator.flux2 import Flux2Edit


def _tiny_flux2() -> Flux2Edit:
    return Flux2Edit(
        dim=8,
        heads=1,
        dim_head=8,
        dropout=0.0,
        ff_mult=1.0,
        latent_dim=4,
        text_hidden_dim=6,
        num_layers=0,
        num_single_layers=0,
    ).eval()


def test_text_projection_cache_reuses_conditional_context() -> None:
    model = _tiny_flux2()
    text = torch.randn(1, 3, 6)

    first = model._project_text_cached(text, drop_text=False, cache=True)
    second = model._project_text_cached(text, drop_text=False, cache=True)

    assert second is first
    assert model.text_cond is first
    assert model.text_uncond is None


def test_text_projection_cache_separates_unconditional_context() -> None:
    model = _tiny_flux2()
    text = torch.randn(1, 3, 6)

    conditional = model._project_text_cached(text, drop_text=False, cache=True)
    unconditional = model._project_text_cached(text, drop_text=True, cache=True)

    assert model.text_cond is conditional
    assert model.text_uncond is unconditional
    assert torch.count_nonzero(unconditional) == 0

    model.clear_cache()
    assert model.text_cond is None
    assert model.text_uncond is None
