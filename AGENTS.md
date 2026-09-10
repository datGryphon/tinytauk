# AGENTS.md

## Project purpose

TinyTAuK is a standalone, memory-efficient AuK-Flash inference runtime. The library owns model loading and waveform generation; higher-level systems own serving, request queues, transcript validation, retries, chunking, stitching, and dialogue orchestration.

## Architectural constraints

- Use official AuK/AuK-Flash checkpoints and configuration as the artifact format.
- Treat the upstream AuK repository as a reference oracle, not a production dependency.
- Keep Qwen conditioning, Flux2 generation, and VAE decode independently configurable by backend, device, dtype, and quantization policy.
- Do not recursively cast or move the complete model stack.
- Preserve Qwen's audio-conditioning path even when optimizing text-only inference.
- Keep Python/PyTorch as the default implementation until measurements justify another backend.
- Do not add GGUF, Vulkan, ROCm, or custom C++ paths without a measured bottleneck and a concrete deployment need.
- Do not add TinyTalk, Hermes, HTTP serving, scene orchestration, or audio stitching to this repository.

## Supported runtime

`profiles/bean.toml` is the current CPU deployment profile. It uses Qwen INT8 weight-only text-transformer weights, dynamic INT8 for selected Flux2 Linear layers, and an FP32 Inductor-compiled VAE.

Text/instruction generation is supported. The public API reserves `reference_audio`, but reference-audio Flux2 generation is not implemented yet and must fail explicitly rather than silently falling back to text-only behavior.

## Code quality

- Prefer typed dataclasses and small interfaces over framework-heavy abstractions.
- Keep heavyweight imports lazy where practical so configuration and diagnostics remain cheap.
- Preserve non-obvious comments that explain numerical parity, checkpoint compatibility, or backend constraints.
- Do not keep comments that merely narrate development history or a previous experiment.
- Every optimization must have a benchmark or parity/quality test demonstrating why it exists.
- Unsupported behavior should fail explicitly.

## Checks

Run:

```bash
./scripts/check
```

For Bean qualification:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 bash scripts/bean-bench
```
