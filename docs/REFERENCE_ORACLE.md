# Upstream AuK reference oracle

The reference oracle runs a pinned upstream AuK implementation on CPU and records deterministic parity boundaries for TinyTAuK.

## Setup

```bash
bash scripts/setup-reference
```

## Run

```bash
bash scripts/reference-oracle
```

Artifacts are written under `benchmarks/results/reference-oracle/` and are intentionally ignored by Git:

- `reference.json` — environment, timing, realtime factor, memory, and tensor metadata.
- `reference.wav` — generated waveform.
- `conditioning.pt` — fused Qwen hidden representation consumed by Flux2Edit.
- `context_mask.pt` — conditioning attention mask.
- `sampled_latent.pt` — final target latent from the sampler before VAE denormalization/decoding.

Tensor metadata in `reference.json` includes shape, dtype, element count, and a SHA-256 digest over the raw tensor bytes.

## Rules

- Keep the oracle environment isolated from the production TinyTAuK dependency graph.
- Record upstream commit SHA, AuK-Flash checkpoint location, Qwen model, PyTorch version, Transformers version and seed with every fixture set.
- Compare intermediate tensor boundaries before relying on final waveform comparisons.
- Generated reference artifacts remain outside Git unless intentionally promoted to tiny fixtures.

## Current parity boundaries

1. fused Qwen hidden-state representation
2. conditioning context mask
3. final target latent produced by the AuK-Flash sampler
4. decoded waveform

Preprocessed Qwen inputs and fixed-seed sampler/noise capture can be added if the standalone implementation needs a finer debugging boundary.
