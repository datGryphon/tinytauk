from __future__ import annotations

import torch

from tinytauk.generator.flux2 import Flux2Edit
from tinytauk.generator.pytorch import PyTorchAuKGenerator
from tinytauk.types import Conditioning, GenerationRequest


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


class _FailingTransformer:
    def __init__(self) -> None:
        self.cleared = False

    def __call__(self, **_kwargs: object) -> torch.Tensor:
        raise RuntimeError("injected failure")

    def clear_cache(self) -> None:
        self.cleared = True


def test_generator_clears_cache_and_preserves_global_rng_on_failure() -> None:
    generator = object.__new__(PyTorchAuKGenerator)
    generator.device = torch.device("cpu")
    generator.dtype = torch.float32
    generator.target_sample_rate = 24_000
    generator.downsample_rate = 480
    generator.latent_dim = 4
    transformer = _FailingTransformer()
    generator.transformer = transformer  # type: ignore[assignment]
    conditioning = Conditioning(
        values=torch.zeros((1, 2, 6), dtype=torch.float32),
        attention_mask=torch.ones((1, 2), dtype=torch.bool),
    )
    request = GenerationRequest(instruction="Test", gen_seconds=0.02, seed=42)

    before = torch.random.get_rng_state().clone()
    try:
        generator.generate_latents(request, conditioning)
    except RuntimeError as exc:
        assert "injected failure" in str(exc)
    else:
        raise AssertionError("expected injected generator failure")

    assert transformer.cleared is True
    assert torch.equal(torch.random.get_rng_state(), before)
