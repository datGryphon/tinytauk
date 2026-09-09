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

## Gate 4: end-to-end BF16 CPU baseline

`TinyTAuK.generate()` now wires the three standalone components together. The default `profiles/cpu-baseline.toml` keeps Qwen and Flux2 in BF16 and the VAE in FP32, which is the first practical CPU baseline rather than an exact upstream-emulation mode.

Run:

```bash
bash scripts/cpu-e2e-baseline
```

The benchmark writes:

- `benchmarks/results/cpu-baseline/baseline.wav`
- `benchmarks/results/cpu-baseline/baseline.json`

It reports component load times, per-stage generation time, overall realtime factor, and process RSS/peak RSS. The structural gate requires exactly three seconds of finite 24 kHz mono audio. Audio quality is evaluated separately before quantization experiments begin.
