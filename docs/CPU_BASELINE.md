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
| Peak RSS | ~14.5 GiB (~14,900 MiB) |

These values are a qualification reference, not a hardware-independent
performance promise.

## Deferred optimization

A Qwen W8A16 experiment improved end-to-end throughput, but its on-the-fly
conversion produced an unacceptable startup memory peak. Prepared, natively
serialized lower-precision weights are a possible v0.3 direction; v0.2 does
not implement quantization or compilation.

## v0.2 release profile

`profiles/cpu.toml` and `TinyTAuK.from_pretrained()` use:

```text
Qwen conditioner   BF16, unquantized
Flux2 generator    FP32, unquantized
VAE                 FP32, eager
PyTorch threads     4
```

The release package does not depend on TorchAO or Torchtune. The reference-audio
smoke and staged upstream parity tools remain available.

## Shared runtime target

```text
Python        >=3.13,<3.14; qualified integration point 3.13.13
torch         2.11.0
torchaudio    2.11.0
transformers  5.17.0
numpy         2.5.3 compatibility point, not exactly pinned
```

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
