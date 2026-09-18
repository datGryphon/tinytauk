# Roadmap

## Current — v0.2

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

The v0.2 release CPU profile favors the repaired upstream-quality path over
aggressive quantization: Qwen BF16, Flux FP32, and VAE FP32. It is aligned with
the shared TinyTalk Python 3.13 / Torch 2.11 / Transformers 5.17 stack and has
no TorchAO or Torchtune dependency.

## Next — v0.3

1. Reuse natively serialized lower-precision model artifacts, starting from the
   qualified Qwen W8A16 result, so optimized profiles do not materialize and
   quantize the FP32 source model on every process start.
2. Keep quantization backend choice platform-specific and configuration-driven.
3. Add explicit prepare/prewarm tooling for deployment.
4. Promote optimized platform profiles only after quality, steady-state memory,
   startup peak, and end-to-end latency are measured.

## Later

1. Qualify a generic source-audio `edit(...)` API for content repair and speech
   enhancement.
2. Expand quality gates beyond WER/CER with speaker-similarity and edit
   preservation metrics.
3. Benchmark reference-latent caching and audio-conditioned retry strategies in
   real callers.
4. Explore paralinguistic editing, target-speaker extraction, alternate VAE
   runtimes, and Vulkan/ggml or GPU acceleration as needed.

TinyTAuK does not own HTTP serving, request queues, transcript retry policy,
chunking, repair policy, or audio stitching. Those stay in the caller.
