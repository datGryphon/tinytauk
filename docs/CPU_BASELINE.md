# CPU qualification

TinyTAuK was checked against a pinned upstream AuK-Flash implementation, then
profiled on a Ryzen 5 Pro 2400G system with 32 GB RAM.

## FP32 parity

| Component | Result |
| --- | --- |
| Qwen + AuK layer fusion | exact tensor equality |
| Flux2 four-step sampler | exact sampled-latent equality |
| BigVGAN decode | within one PCM16 LSB |

See `REFERENCE_ORACLE.md` and `scripts/cpu-*-parity`.

## FP32 baseline

The original all-FP32 stack was about `9.4x` realtime for a 10-second target.
Four PyTorch threads performed best on the test system.

## Flux2 INT8

The supported `quantization = "int8"` policy quantizes large non-sensitive
Linear layers and keeps timing/text/audio projections, normalization paths, and
the final projection in FP32.

| Policy | WER | Generator speedup | RMS vs FP32 |
| --- | ---: | ---: | ---: |
| FP32 | 48% | 1.00x | 100% |
| attention | 51% | 1.08x | 79% |
| **core** | **41%** | **1.52x** | **50%** |
| all | 57% | 1.61x | 19% |

`int8-all` was rejected because it added audible scratch/static for little
extra speed. WER is only one quality signal; listening and acoustic regression
checks are also required.

## VAE compile

The VAE stays FP32 and uses TorchInductor.

| Target | Eager RTF | Compiled RTF | Speedup |
| --- | ---: | ---: | ---: |
| 3 s | 1.858 | 1.575 | 1.18x |
| 10 s | 2.593 | 1.870 | 1.39x |
| 20 s | 2.974 | 2.202 | 1.35x |

Compiled/eager waveform SNR was about 100 dB. Inductor compiles lazily; warm
the full `TinyTAuK.generate()` path in a long-running service. Isolated VAE
precompile was tested and rejected because the real request still recompiled
and peak RSS increased.

## Qwen INT8 weight-only

Qwen text-transformer weights use INT8 weight-only quantization. The audio
tower remains FP32 for future reference-audio support.

| Conditioner | Warm RTF | Warm RSS | Peak RSS |
| --- | ---: | ---: | ---: |
| FP32 | ~4.57 | ~17.7 GiB | ~22.1 GiB |
| INT8 weight-only | ~5.32 | ~10.7 GiB | ~15.9 GiB |

The memory saving was worth the modest conditioning slowdown on the test
system.

## Release profile

`profiles/cpu.toml`:

```text
Qwen text transformer   INT8 weight-only
Qwen audio tower        FP32
Flux2                    dynamic INT8 core
VAE                      FP32 + Inductor
PyTorch threads          4
```

Run:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 bash scripts/cpu-bench
```

Results under `benchmarks/results/` are ignored by Git.
