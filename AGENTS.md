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
  configurable by backend, device, dtype, and supported quantization policy.
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

TinyTAuK 0.2 targets Python `>=3.13,<3.14`, Torch/Torchaudio 2.11.0, and
Transformers 5.17.0. NumPy 2.5.3 is a compatibility test point, not an exact
project pin. The v0.2 package must not depend on TorchAO or Torchtune.

`TinyTAuK.from_pretrained()` and `profiles/cpu.toml` use the same qualified
CPU configuration: Qwen BF16, Flux2 FP32, VAE FP32, all unquantized. Dynamic
PyTorch INT8 Flux remains available only as an explicit experiment.

Text/instruction generation and reference-audio zero-shot voice cloning are
supported. One engine instance processes one inference operation at a time
because Flux2 caches projections during its four sampling steps.

## Commands

```bash
rm -rf .venv
uv sync
uv run tinytauk doctor
./scripts/check
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 bash scripts/cpu-bench
```
