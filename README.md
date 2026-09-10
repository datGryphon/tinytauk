# TinyTAuK

TinyTAuK is a standalone Python/PyTorch inference runtime for Tencent Hunyuan AuK-Flash, focused on memory-efficient CPU deployment and compatibility with the official model checkpoints.

TinyTAuK implements the AuK-Flash inference path without requiring the upstream `auk` Python package at runtime:

```text
instruction
  -> Qwen2.5-Omni Thinker
  -> AuK learned hidden-state fusion
  -> Flux2Edit four-step sampler
  -> BigVGAN decoder
  -> 24 kHz waveform
```

The project was built against the official AuK implementation as a reference oracle and is intended to be consumed as a library by higher-level TTS systems such as TinyTalk. It is not an HTTP service and does not own dialogue orchestration, chunking, retry policy, stitching, or output encoding.

## Status

Version `0.1.0` supports:

- standalone AuK-Flash text/instruction inference;
- exact FP32 parity checks against a pinned upstream AuK reference;
- independent dtype/quantization configuration for conditioner, generator, and VAE;
- dynamic INT8 quantization for the large Flux2 Linear layers while retaining sensitive modules in FP32;
- INT8 weight-only Qwen text-transformer weights for lower steady-state RAM use;
- FP32 Qwen audio-tower retention so reference-audio conditioning can be added without changing the model-loading architecture;
- optional `torch.compile`/Inductor VAE decode;
- CPU benchmark and speech-quality tooling.

Reference-audio generation is not implemented yet. `TinyTAuK.generate()` reserves a `reference_audio` argument, but the current Flux2 generation path raises `NotImplementedError` when it is used.

## Installation

The repository includes a Nix development shell and an `uv` lockfile. The optimized Bean profile requires the `quant` extra for TorchAO:

```bash
nix develop
uv sync --extra quant
./scripts/check
```

TinyTAuK downloads the configured official model checkpoints through Hugging Face on first use. Model weights and generated audio are not stored in the repository.

## Generate speech

The recommended CPU profile is `profiles/bean.toml`:

```bash
uv run --extra quant tinytauk generate \
  --profile profiles/bean.toml \
  --seconds 9 \
  --output output.wav \
  'Generate speech based on the following description: "A calm, natural technical narration". The content to speak is: "The service restarted successfully and all health checks passed.".'
```

The command prints generated duration and per-stage timings as JSON after writing the WAV.

The compiled VAE is lazy: the first request for a new Inductor graph can be substantially slower than subsequent requests. Long-running services should load one `TinyTAuK` instance and perform a disposable startup generation at the normal serving duration before reporting ready.

## Python API

```python
from tinytauk import TinyTAuK

engine = TinyTAuK.from_config("profiles/bean.toml")
result = engine.generate(
    'Generate speech based on the following description: "A calm, natural technical narration". '
    'The content to speak is: "The service restarted successfully.".',
    gen_seconds=9,
)

print(result.sample_rate)
print(result.generated_seconds)
print(result.stage_seconds)
```

`GenerationResult.audio` is a CPU `torch.Tensor`. TinyTAuK deliberately leaves HTTP serving, request queues, transcript validation, retries, and audio stitching to the caller.

## Bean profile

`profiles/bean.toml` is the current low-memory CPU deployment profile:

- Qwen text-transformer Linear weights: INT8 weight-only;
- Qwen audio tower: FP32;
- Flux2 large non-sensitive Linear layers: dynamic INT8;
- Flux2 sensitive projections/norm paths: FP32;
- VAE: FP32 with Inductor;
- PyTorch intra-op threads: 4.

The optimization choices and qualification results are documented in [`docs/CPU_BASELINE.md`](docs/CPU_BASELINE.md) and [`docs/QUALITY_BENCHMARK.md`](docs/QUALITY_BENCHMARK.md).

Run the current Bean benchmark with:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 bash scripts/bean-bench
```

Results and generated WAVs are written under `benchmarks/results/` and ignored by Git.

## Reference oracle

The standalone implementation can be checked against a pinned upstream AuK commit. The reference checkout is development-only and ignored by Git.

See [`docs/REFERENCE_ORACLE.md`](docs/REFERENCE_ORACLE.md).

## Development

```bash
./scripts/check
```

runs Ruff linting/format checks, strict mypy, and the unit test suite.

Additional profiling and qualification scripts under `scripts/` preserve the measurements used to select the current CPU profile.

## Licensing

TinyTAuK is MIT-licensed. Portions of the standalone Flux2 and BigVGAN implementations are adapted from upstream projects with compatible licenses. See [`THIRD_PARTY.md`](THIRD_PARTY.md) for attribution and license details.
