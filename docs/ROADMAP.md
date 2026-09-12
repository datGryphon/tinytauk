# Roadmap

## Current

TinyTAuK `0.2.0` adds reusable utterance conditioning and audio-conditioned
AuK-Flash inference on top of the qualified CPU runtime.

The public API is:

```python
TinyTAuK.from_pretrained(...)
TinyTAuK.from_config(...)
TinyTAuK.condition(...)
TinyTAuK.generate_conditioned(...)
TinyTAuK.generate(...)
```

Reference audio accepts a path or an in-memory `(waveform, sample_rate)` tuple.
Zero-shot TTS is the first qualified use of the audio-conditioned path.

## Next

1. Qualify a generic source-audio `edit(...)` API for content repair and speech
   enhancement.
2. Expand quality gates beyond WER/CER with speaker-similarity and edit
   preservation metrics.
3. Benchmark reference-latent caching and audio-conditioned retry strategies in
   real callers.

## Later, if needed

- paralinguistic editing and target-speaker extraction;
- lower-memory Qwen hidden-state fusion;
- audio-tower quantization after reference-audio quality is measured;
- alternate VAE runtimes;
- Vulkan/ggml or GPU acceleration.

TinyTAuK does not own HTTP serving, request queues, transcript retry policy,
chunking, repair policy, or audio stitching. Those stay in the caller.
