# TinyTAuK

TinyTAuK is an experimental, memory-efficient Python/PyTorch inference runtime for Tencent Hunyuan AuK-Flash.

The initial objective is deliberately narrow:

1. reproduce AuK-Flash inference on CPU,
2. establish parity against the upstream implementation,
3. reduce memory through component isolation and quantization,
4. qualify the resulting runtime on Bean,
5. only then expose it as a TinyTalk backend.

TinyTAuK is **not** currently a TTS HTTP service and does not contain TinyTalk or Hermes integration.

## Status

**v0 scaffold.** The repository structure, configuration model, CLI, benchmark result schema, Nix development shell, and test harness are present. AuK model execution is intentionally not implemented yet.

See [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md).

## Development

The expected development environment is Nix + uv:

```bash
nix develop
uv sync --extra quant
uv run pytest
uv run tinytauk doctor
```

If you use direnv:

```bash
direnv allow
```

The first `nix develop` will create `flake.lock`; commit it after verifying the shell on Neptune. The first `uv sync` will create `uv.lock`; commit it once the Python dependency set has been validated.

## Initial CLI

```bash
uv run tinytauk doctor
uv run tinytauk config profiles/cpu-baseline.toml
uv run tinytauk benchmark --target-seconds 10 --dry-run
```

`generate` is reserved for Phase 2 and currently fails explicitly rather than pretending inference is implemented.

## Model artifacts

Do not commit model weights, generated audio, or reference model caches. Keep Hugging Face caches outside the repository.

The production implementation should consume official AuK/AuK-Flash checkpoints without requiring an installed copy of the upstream `auk` package. Upstream AuK remains the reference oracle during parity development.

## Licensing

This scaffold does not vendor Tencent AuK source code. AuK itself is released under MIT; if upstream code is later copied or adapted, preserve the relevant Tencent copyright and license notices in the affected files and update `THIRD_PARTY.md`.
