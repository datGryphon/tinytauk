# Roadmap

## Current

TinyTAuK supports AuK-Flash text/instruction TTS with the CPU profile in
`profiles/bean.toml`.

The public API is:

```python
TinyTAuK.from_config(...)
TinyTAuK.generate(...)
```

## Next

1. Add TinyTAuK as a TinyTalk backend.
2. Add reference-audio generation. The Qwen audio tower is already retained;
   the Flux2 reference path is not implemented yet.
3. Keep the existing WER/CER corpus as the regression gate.

## Later, if needed

- lower-memory Qwen hidden-state fusion;
- audio-tower quantization after reference-audio quality can be measured;
- alternate VAE runtimes;
- Vulkan/ggml or GPU acceleration.

TinyTAuK does not own HTTP serving, request queues, transcript retry policy,
chunking, or audio stitching. Those stay in TinyTalk.
