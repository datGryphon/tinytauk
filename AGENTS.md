# AGENTS.md

## Project

TinyTAuK is a standalone, memory-efficient AuK-Flash inference runtime. The
library owns model loading and waveform generation. Callers own serving,
request queues, transcript validation, retries, chunking, and stitching.

## Conventions

Keep this project simple and direct.

- Use official AuK/AuK-Flash checkpoints and configuration.
- Treat upstream AuK as a reference oracle, not a runtime dependency.
- Keep Qwen conditioning, Flux2 generation, and VAE decode independently
  configurable by backend, device, dtype, and quantization policy.
- Do not recursively cast or move the complete model stack.
- Preserve the Qwen audio-conditioning path while optimizing text-only use.
- Keep Python/PyTorch as the default implementation until measurements justify
  another backend.
- Do not add serving, transcript retry policy, chunking, or audio stitching to
  this repository.
- Prefer typed dataclasses and small interfaces over speculative abstractions.
- Preserve comments that explain numerical parity, checkpoint compatibility,
  licensing, or backend constraints. Remove comments that only narrate an old
  experiment or development session.
- Every optimization needs a benchmark or parity/quality test.
- Unsupported behavior should fail explicitly.

## Runtime

`profiles/cpu.toml` is the current low-memory CPU profile. It uses Qwen INT8
weight-only text-transformer weights, dynamic INT8 for selected Flux2 Linear
layers, and an FP32 Inductor-compiled VAE.

Text/instruction generation is supported. `reference_audio` is reserved by the
public API, but Flux2 reference-audio generation is not implemented yet.

## Commands

```bash
./scripts/check
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 bash scripts/cpu-bench
```
