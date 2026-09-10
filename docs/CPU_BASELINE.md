# CPU runtime qualification

TinyTAuK was implemented against a pinned upstream AuK-Flash reference and then optimized component-by-component for a shared-memory CPU deployment.

## FP32 parity

The standalone FP32 paths reproduce the upstream reference behavior without importing the upstream `auk` package at runtime.

| Component | Reference result |
| --- | --- |
| Qwen conditioning + AuK layer fusion | exact tensor equality, shape `[1, 43, 2048]` |
| Flux2Edit four-step sampler | exact sampled-latent equality in FP32 |
| BigVGAN decode | reference waveform within one PCM16 LSB |

The reference tools are retained under `scripts/cpu-*-parity` and documented in [`REFERENCE_ORACLE.md`](REFERENCE_ORACLE.md).

## Bean FP32 baseline

The initial all-FP32 Bean run established an end-to-end baseline around `9.4x` realtime for a 10-second target. Component timing showed that all three major stages were material:

- Qwen conditioning: roughly 36–45 seconds;
- Flux2 generation: roughly 34 seconds;
- VAE decode: roughly 19–24 seconds after weight-normalization removal.

Four PyTorch intra-op threads performed best on the target Ryzen 5 Pro 2400G.

## Flux2 dynamic INT8

Dynamic INT8 was evaluated with progressively broader Linear-layer policies. The selected policy quantizes large non-sensitive transformer Linear layers while retaining timing/text/audio projections, normalization paths, and the final projection in FP32.

Representative quality result:

| Policy | WER | Generator speedup | Hot-path speedup | RMS vs FP32 |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 48% | 1.00x | 1.00x | 100% |
| attention | 51% | 1.08x | 1.04x | 79% |
| **core** | **41%** | **1.52x** | **1.29x** | **50%** |
| all Linear layers | 57% | 1.61x | 1.33x | 19% |

The all-Linears policy produced audible scratch/static and excessive high-frequency energy for only a small incremental speed gain. The core policy is therefore the supported `quantization = "int8"` generator behavior.

WER is not treated as a standalone perceptual metric: quantization can change ASR behavior without improving speech quality. Acceptance uses intelligibility, listening, acoustic regression metrics, and performance together.

## VAE Inductor compile

The VAE is kept in FP32. Removing weight normalization and compiling the decoder with TorchInductor materially reduced warm decode time while remaining numerically close to eager output.

Isolated Bean results:

| Target | Eager RTF | Compiled RTF | Speedup |
| --- | ---: | ---: | ---: |
| 3 s | 1.858 | 1.575 | 1.18x |
| 10 s | 2.593 | 1.870 | 1.39x |
| 20 s | 2.974 | 2.202 | 1.35x |

Compiled/eager waveform SNR was approximately 100 dB in these tests.

Inductor compilation is lazy and can make the first request for a graph much slower. Attempts to precompile the VAE in isolation during engine construction were rejected: a nominally matching 9-second dummy decode still triggered another roughly 109-second VAE compile on the first real request, while retained and peak RSS increased. Service-level warmup should exercise the complete `TinyTAuK.generate()` path instead.

## Qwen INT8 weight-only

The largest persistent-memory improvement came from weight-only INT8 on the Qwen text transformer. The audio tower remains FP32 so future reference-audio conditioning is not designed out of the runtime.

A representative Bean comparison before the isolated-precompile experiment was removed:

| Conditioner | Warm RTF | Warm process RSS | Peak process RSS |
| --- | ---: | ---: | ---: |
| FP32 | ~4.57 | ~17.7 GiB | ~22.1 GiB |
| INT8 weight-only | ~5.32 | ~10.7 GiB | ~15.9 GiB |

The memory saving is large enough to justify the modest conditioning-latency increase on a shared host. Speech from the weight-only candidate did not show the scratch/static failure mode seen with over-quantized Flux2. Content-level AuK prompt leakage/repetition remains possible and should be handled by the calling application's transcript validation and retry policy.

## Supported Bean profile

`profiles/bean.toml` contains the selected release configuration:

```text
Qwen text transformer   INT8 weight-only
Qwen audio tower        FP32
Flux2                    dynamic INT8 core policy
VAE                      FP32 + Inductor
PyTorch threads          4
```

Run the release profile with:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 bash scripts/bean-bench
```

The benchmark records first-request and warm stage timings, current RSS after each request, the process high-water mark, waveform statistics, and the exact Git revision.

Generated benchmark results are intentionally ignored by Git so qualification can be repeated on different machines without committing machine-specific artifacts.
