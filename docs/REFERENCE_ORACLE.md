# Upstream AuK reference oracle

This document will hold the exact procedure used to capture parity fixtures from the official AuK-Flash implementation.

## Rules

- Keep the oracle environment isolated from the production TinyTAuK dependency graph.
- Record upstream commit SHA, AuK-Flash checkpoint revision, Qwen revision, PyTorch version, Transformers version and seed with every fixture set.
- Never compare only final WAV files if an intermediate boundary can be captured deterministically.
- Generated reference artifacts should remain outside git unless they are intentionally tiny fixtures.

## Planned boundaries

1. preprocessed Qwen inputs
2. Qwen hidden-state/layer-fusion result
3. sampler input/noise at fixed seed
4. final AuK latent
5. VAE waveform

The exact capture code is a Phase 1 task and is intentionally absent from v0.
