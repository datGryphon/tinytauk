# tinytauk

Memory-efficient AuK-Flash inference runtime for CPU environments.

TinyTAuK loads the official Tencent Hunyuan AuK-Flash checkpoints directly. It
is a library/CLI, not a TTS server; callers own serving, request queues,
validation, retries, chunking, and audio stitching.

## Status

Version `0.2.0` supports:

- text/instruction TTS;
- reusable Qwen/AuK utterance conditioning for cheap resampling/retries;
- reference-audio conditioning and zero-shot voice cloning;
- file-path and in-memory tensor reference audio;
- Qwen2.5-Omni conditioning and AuK hidden-state fusion;
- AuK-Flash four-step Flux2 generation;
- BigVGAN reference encode and waveform decode;
- INT8 weight-only Qwen text weights with the audio tower kept FP32;
- dynamic INT8 Flux2 core weights;
- FP32 Inductor-compiled VAE decode.

The reference-audio encoder is loaded lazily on the first request that needs it,
so text-only startup keeps the `0.1.0` memory shape.

## Setup

```bash
nix develop
uv sync
./scripts/check
```

## CLI

Text-only generation:

```bash
uv run tinytauk generate \
  --seconds 9 \
  --output output.wav \
  'Generate speech based on the following description: "A calm, natural technical narration". The content to speak is: "The service restarted successfully.".'
```

Zero-shot voice cloning:

```bash
uv run tinytauk generate \
  --reference-audio voice.wav \
  --seconds 6 \
  --output cloned.wav \
  'Say the following with the same voice: "The service restarted successfully."'
```

The command uses the built-in CPU configuration, writes a 24 kHz WAV, and
prints generation timing as JSON. Pass `--profile` to use a TOML runtime
profile instead.

The decoder compiles lazily on first use. Long-running callers should keep one
`TinyTAuK` instance resident and run one disposable generation before serving.
A single instance processes one inference operation at a time.

## Python API

The normal one-shot path stays small:

```python
from tinytauk import TinyTAuK

engine = TinyTAuK.from_pretrained()
result = engine.generate(
    'Generate speech based on the following description: "A calm, natural technical narration". '
    'The content to speak is: "The service restarted successfully.".',
    gen_seconds=9,
)
```

Reference audio can be a path or `(waveform, sample_rate)` tuple:

```python
result = engine.generate(
    'Say the following with the same voice: "The service restarted successfully."',
    reference_audio="voice.wav",
    gen_seconds=6,
)
```

For retries of the same utterance, build conditioning once and resample it:

```python
conditioning = engine.condition(
    'Say the following with the same voice: "The service restarted successfully."',
    reference_audio="voice.wav",
)

first = engine.generate_conditioned(conditioning, gen_seconds=6, seed=42)
retry = engine.generate_conditioned(conditioning, gen_seconds=6, seed=43)
```

`generate_conditioned()` reuses both the Qwen conditioning and any encoded
reference latents. It does not rerun Qwen or the reference VAE.

`result.audio` is a CPU `torch.Tensor`; `sample_rate`, `generated_seconds`, and
per-stage timings are also returned.

Use `TinyTAuK.from_config(...)` for explicit component/runtime configuration.
`profiles/cpu.toml` contains the same low-memory CPU configuration used by
`from_pretrained()`.

## Benchmarks

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 bash scripts/cpu-bench
```

See `docs/CPU_BASELINE.md` for the measurements behind the CPU profile and
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
