# CPU baseline

The CPU baseline is implemented incrementally against the captured upstream AuK-Flash oracle.

## Gate 1: semantic conditioning — passed

`TransformersConditioner` owns Qwen2.5-Omni Thinker inference and AuK's learned layer fusion without importing the upstream `auk` package.

For exact oracle comparison, `upstream_parity=True` reproduces upstream's BF16-load-then-FP32-promotion behavior. The normal TinyTAuK path keeps Qwen at the configured dtype.

Neptune result on 2026-09-09:

- output shape: `[1, 43, 2048]`
- exact equality: true
- max absolute error: `0.0`
- encode time: `1.136 s`
- peak RSS for the isolated conditioner process: `18.8 GB`

Re-run with:

```bash
bash scripts/cpu-conditioning-parity
```

## Gate 2: Flux2Edit sampled latent — passed

The standalone AuK-Flash Flux2Edit transformer and fixed four-step Euler sampler reproduce the oracle latent exactly in FP32.

Neptune result on 2026-09-09:

- output shape: `[1, 150, 64]`
- exact equality: true
- max absolute error: `0.0`
- generation time: `5.118 s`
- realtime factor: `1.706x`
- peak RSS for the isolated generator process: `11.98 GB`

Re-run with:

```bash
bash scripts/cpu-generator-parity
```

## Gate 3: BigVGAN decode — passed

The standalone decoder-only BigVGAN path consumes the oracle sampled latent and reproduces the serialized reference WAV within one 16-bit PCM LSB.

Neptune result on 2026-09-09:

- output shape: `[1, 72000]`
- sample rate: `24000 Hz`
- max absolute error: `2.246e-06`
- WAV tolerance: `3.052e-05`
- decode time: `2.277 s`
- realtime factor: `0.759x`
- peak RSS for the isolated VAE process: `1.55 GB`

Re-run with:

```bash
bash scripts/cpu-vae-parity
```

## Gate 4: end-to-end CPU baseline — passed

`TinyTAuK.generate()` wires the three standalone components together. The first BF16/BF16 Neptune run produced valid 24 kHz audio with a `12.29 GB` peak RSS and `6.12x` realtime factor.

A one-time Neptune dtype sweep showed that dtype performance is strongly CPU-specific:

| Qwen | Flux2 | RTF | Peak RSS | Oracle SNR |
| --- | --- | ---: | ---: | ---: |
| BF16 | BF16 | 5.988x | 12.31 GB | -1.25 dB |
| BF16 | FP32 | 3.098x | 13.17 GB | 6.49 dB |
| FP32 | BF16 | 8.006x | 22.59 GB | -1.28 dB |
| FP32 | FP32 | 5.398x | 22.61 GB | 121.93 dB |

On Neptune, Flux2 FP32 is substantially faster and numerically closer to the oracle than Flux2 BF16. These results are characterization only; Neptune is not the deployment target.

Re-run with:

```bash
bash scripts/cpu-e2e-baseline
bash scripts/cpu-dtype-sweep
```

## Gate 5: Bean deployment qualification

Bean is the deployment target. The initial qualification profile uses FP32 for Qwen, Flux2, and VAE because the Ryzen 5 Pro 2400G has no native BF16 execution path. The goal is to establish deployment behavior before spending time on quantization or CPU-specific tuning.

The qualification run defaults to a 10-second utterance and records component load/generation times, total RSS/peak RSS, PyTorch thread counts, CPU/RAM information, waveform statistics, and an 8-minute latency gate.

Run on Bean:

```bash
bash scripts/bean-qualify
```

Optional duration override:

```bash
GEN_SECONDS=15 bash scripts/bean-qualify
```

Outputs:

- `benchmarks/results/bean-qualify/bean-qualify.wav`
- `benchmarks/results/bean-qualify/bean-qualify.json`

If a normal 5–15 second utterance stays within memory, sounds acceptable, and completes in under roughly 8 minutes, the CPU runtime is qualified for TinyTalk integration. Only optimize further when Bean measurements show a concrete memory or latency bottleneck.
