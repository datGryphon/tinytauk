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
- optional dynamic INT8 Flux experiments through PyTorch's CPU quantization path.

The v0.2 release CPU profile is intentionally conservative: Qwen runs BF16,
while Flux and the VAE run FP32 without quantization or compilation. The
reference-audio encoder is loaded lazily on the first request that needs it.

TinyTAuK 0.2 targets the same core runtime now qualified by TinyTalk's other
backends: Python 3.13, PyTorch/Torchaudio 2.11, and Transformers 5.17. TinyTAuK
does not depend on TorchAO or Torchtune.

## Setup

```bash
nix develop
rm -rf .venv
uv sync
uv run tinytauk doctor
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

Long-running callers should keep one `TinyTAuK` instance resident so model
loading is paid once. A single instance processes one inference operation at a
time.

## Python API

```python
from tinytauk import TinyTAuK

engine = TinyTAuK.from_pretrained()
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

Use `TinyTAuK.from_config(...)` for explicit component/runtime configuration.
`profiles/cpu.toml` contains the same qualified CPU configuration used by
`from_pretrained()`.

## Runtime compatibility

`tinytauk doctor` prints the actual Python, Torch, Torchaudio, Transformers,
and NumPy versions in the active environment.

The v0.2 compatibility target is:

```text
Python        >=3.13,<3.14 (qualified at 3.13.13)
torch         2.11.0
torchaudio    2.11.0
transformers  5.17.0
numpy         2.5.3 compatibility point; no exact project pin
```

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
