# TinyTAuK roadmap

## Current state

TinyTAuK implements standalone AuK-Flash text/instruction inference using official checkpoints:

```text
instruction
        -> Qwen2.5-Omni Thinker
        -> AuK learned hidden-state fusion
        -> Flux2Edit four-step sampler
        -> BigVGAN decoder
        -> waveform
```

The FP32 implementation has been checked against a pinned upstream AuK oracle. CPU profiling and quality measurements produced the current low-memory deployment profile in `profiles/bean.toml`.

The supported Python API is deliberately small: `RuntimeConfig`, `TinyTAuK.from_config()`, and `TinyTAuK.generate()`.

## Current deployment profile

`profiles/bean.toml` uses:

- Qwen text-transformer Linear weights: INT8 weight-only;
- Qwen audio tower: FP32 and retained;
- Flux2 large non-sensitive Linear layers: dynamic INT8;
- Flux2 sensitive paths: FP32;
- VAE: FP32 with `torch.compile`/Inductor;
- four CPU intra-op threads.

The first VAE decode for a new Inductor graph can be much slower than subsequent calls. Service processes should warm the complete `TinyTAuK.generate()` path at their normal serving duration rather than trying to precompile the VAE in isolation.

## Near-term work

### TinyTalk integration

Add a TinyTalk backend that keeps one TinyTAuK engine resident and translates TinyTalk speaker/delivery instructions into AuK instruction text. TinyTalk remains responsible for HTTP serving, request queues, transcript validation, retries, chunking, stitching, and output encoding.

### Reference-audio generation

The conditioner already retains the Qwen audio tower and the public request type reserves `reference_audio`. The Flux2 generation path still needs the corresponding reference-latent/audio-conditioning implementation.

Acceptance criteria should include:

- speaker/timbre consistency;
- intelligibility and transcript accuracy;
- prosody/style preservation;
- memory and latency impact relative to text-only inference.

### Quality validation

Continue using the corpus and scorer under `benchmarks/quality/` as a regression suite. Transcript validation must use proper Levenshtein WER/CER rather than heuristics that undercount insertions or repeated phrases.

## Conditional future work

Only pursue these when deployment measurements justify them:

- streaming or lower-memory Qwen hidden-state fusion;
- audio-tower weight-only quantization after reference-audio quality can be measured;
- alternate VAE runtimes;
- Vulkan/ggml acceleration;
- GPU deployment and coordinated model residency with other inference services.

## Non-goals

TinyTAuK does not own:

- HTTP or RPC serving;
- dialogue or scene planning;
- speaker turn orchestration;
- transcript correction/retry policy;
- pause/crossfade/stitching logic;
- application-specific integration with Hermes or other agents.

Those concerns belong to the calling application.
