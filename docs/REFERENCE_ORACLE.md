# AuK reference-audio parity

The maintained parity path compares TinyTAuK's zero-shot reference-audio
conditioning, Flux sampling, and waveform decoding against a pinned upstream
AuK implementation. It is a development diagnostic, not a library feature.

Use an isolated Python environment with the pinned upstream AuK checkout for
the upstream stage runner; it must not replace TinyTAuK's qualified runtime.
The upstream revision used during qualification is
`d9f30ffe4231dbc90b48cc83a35d310fece0b060`.

```bash
scripts/run-upstream-parity-staged \
  /path/to/upstream/python \
  /path/to/auk_flash.safetensors \
  /path/to/qwen-snapshot \
  /path/to/reference.wav \
  benchmarks/results/upstream-parity \
  4.5 42

uv run python scripts/local_parity_trace.py \
  /path/to/reference.wav \
  benchmarks/results/local-parity \
  --seconds 4.5 --seed 42

uv run python scripts/compare_parity_traces.py \
  benchmarks/results/local-parity/trace.pt \
  benchmarks/results/upstream-parity/trace.pt
```

The upstream runner uses separate processes for VAE encode, Qwen conditioning,
Flux sampling, and decode to limit simultaneous model residency. Both trace
implementations record matched tensor boundaries; the comparator reports the
first material divergence. Keep the reference clip, target text, seed, model
checkpoints, and runtime versions consistent when comparing.

Generated results under `benchmarks/results/` are ignored by Git.
