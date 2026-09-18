# CPU qualification

TinyTAuK v0.2 was checked against a pinned upstream AuK-Flash implementation
and qualified around a conservative CPU runtime.

## Correctness oracle

The repaired upstream-parity path uses:

```text
Qwen2.5-Omni conditioner   BF16
Flux2 generator            FP32
VAE encoder/decoder        FP32
```

The parity trace compares reference VAE encoder intermediates, sampled
reference latents, conditioner values, initial noise, first Flux velocity,
final latents, and decoded audio. The remaining conditioner drift is small
enough that the final waveform closely tracks upstream perceptually and
numerically.

For the fixed 4.5-second zero-shot cloning smoke case used during final
qualification:

| Stage / metric | Baseline |
| --- | ---: |
| Qwen + reference conditioning | ~117.8 s |
| Flux generator | ~83.2 s |
| VAE decode | ~9.6 s |
| End-to-end request | ~210.6 s |
| Peak RSS | ~14.9 GiB |

These values are a qualification reference, not a hardware-independent
performance promise.

## Optimization findings

The branch also tested several lower-precision paths before the v0.2 release
profile was frozen.

Qwen W8A16 produced the useful result: about 1.40x end-to-end speedup in the
final ladder with quality close to the repaired baseline, but on-the-fly
conversion caused a ~26 GiB transient peak. That experiment depended on
TorchAO, which is intentionally not a v0.2 dependency. The result is retained
as motivation for the v0.3 prepared-quantized-model work.

Flux dynamic INT8 improved generator throughput in isolation but caused a large
speech-quality regression, so it remains an explicit experiment rather than a
release default.

BF16 Flux was both slow and acoustically incorrect on the tested CPU path.
A BF16 VAE preserved waveform quality but made decode dramatically slower and
provided little memory benefit.

## v0.2 release profile

`profiles/cpu.toml` and `TinyTAuK.from_pretrained()` use:

```text
Qwen conditioner   BF16, unquantized
Flux2 generator    FP32, unquantized
VAE                 FP32, eager
PyTorch threads     4
```

The release package does not depend on TorchAO or Torchtune.

## Shared runtime target

```text
Python        >=3.13,<3.14; qualified integration point 3.13.13
torch         2.11.0
torchaudio    2.11.0
transformers  5.17.0
numpy         2.5.3 compatibility point, not exactly pinned
```

Run `uv run tinytauk doctor` to print the versions actually installed.

## v0.3 direction

The next CPU optimization is to prepare a lower-precision Qwen model once using
the quantizer/model stack's native serialization and reload that representation
directly on later starts. The v0.3 work should choose the quantization backend
that fits the target platform rather than making it a mandatory v0.2 runtime
dependency.

Run the release benchmark with:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 bash scripts/cpu-bench
```

Results under `benchmarks/results/` are ignored by Git.
