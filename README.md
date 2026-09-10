# tinytauk

Memory-efficient AuK-Flash inference runtime for CPU environments.

TinyTAuK loads the official Tencent Hunyuan AuK-Flash checkpoints directly. It
is a library/CLI, not a TTS server; TinyTalk owns serving, chunking, validation,
retries, and audio stitching.

## Status

Version `0.1.0` supports text/instruction TTS with:

- Qwen2.5-Omni conditioning and AuK hidden-state fusion;
- AuK-Flash four-step Flux2 generation;
- BigVGAN waveform decode;
- INT8 weight-only Qwen text weights;
- dynamic INT8 Flux2 core weights;
- FP32 Inductor-compiled VAE.

Reference-audio generation is not implemented yet. The Qwen audio tower is
retained for that path.

## Setup

```bash
nix develop
uv sync
./scripts/check
```

## CLI

```bash
uv run tinytauk generate \
  --profile profiles/cpu.toml \
  --seconds 9 \
  --output output.wav \
  'Generate speech based on the following description: "A calm, natural technical narration". The content to speak is: "The service restarted successfully.".'
```

The command writes a 24 kHz WAV and prints generation timing as JSON.

The VAE compiles lazily on first use. Long-running callers should keep one
`TinyTAuK` instance resident and run one disposable generation before serving.

## Python API

```python
from tinytauk import TinyTAuK

engine = TinyTAuK.from_config("profiles/cpu.toml")
result = engine.generate(
    'Generate speech based on the following description: "A calm, natural technical narration". '
    'The content to speak is: "The service restarted successfully.".',
    gen_seconds=9,
)
```

`result.audio` is a CPU `torch.Tensor`; `sample_rate`, `generated_seconds`, and
per-stage timings are also returned.

## CPU profile

`profiles/cpu.toml` is the current low-memory CPU profile:

| Component | Runtime |
| --- | --- |
| Qwen text transformer | INT8 weight-only |
| Qwen audio tower | FP32 |
| Flux2 | dynamic INT8 core policy |
| VAE | FP32 + Inductor |
| PyTorch threads | 4 |

Benchmark it with:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 bash scripts/cpu-bench
```

See `docs/CPU_BASELINE.md` for the measurements behind this profile and
`docs/QUALITY_BENCHMARK.md` for the speech-quality gate.

## Reference oracle

```bash
bash scripts/setup-reference
bash scripts/reference-oracle
```

The reference checkout and generated artifacts are ignored by Git. See
`docs/REFERENCE_ORACLE.md`.

## License

MIT. Adapted upstream code and licenses are listed in `THIRD_PARTY.md`.
