# CPU baseline

The CPU baseline is implemented incrementally against the captured upstream AuK-Flash oracle.

## Gate 1: semantic conditioning — passed

`TransformersConditioner` owns Qwen2.5-Omni Thinker inference and AuK's learned layer fusion without importing the upstream `auk` package.

For exact oracle comparison, `upstream_parity=True` reproduces upstream's BF16-load-then-FP32-promotion behavior. The normal TinyTAuK path keeps Qwen at the configured dtype and will be evaluated separately after end-to-end FP32 parity is established.

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

The second gate ports the AuK-Flash Flux2Edit transformer and fixed four-step Euler sampler into TinyTAuK. It consumes the captured `conditioning.pt` and `context_mask.pt` directly, isolating the generator from Qwen.

Neptune result on 2026-09-09:

- output shape: `[1, 150, 64]`
- exact equality: true
- max absolute error: `0.0`
- four-step generation time: `5.118 s`
- realtime factor for 3 s of latent audio: `1.706x`
- peak RSS for the isolated generator process: `11.98 GB`

Re-run with:

```bash
bash scripts/cpu-generator-parity
```

## Gate 3: BigVGAN VAE decode

The third gate ports only the decoder half of `BigVGANFlowVAE`. TinyTAuK does not need the VAE encoder or normalizing flow for instruction-only synthesis, so they are intentionally omitted from this baseline.

The parity runner consumes the exact captured `sampled_latent.pt`, denormalizes it with the checkpoint's global statistics, decodes it to waveform samples, and compares against the oracle `reference.wav`.

Because the oracle WAV is serialized through `torchaudio.save`, the pass criterion permits at most one 16-bit PCM least-significant bit of error:

```bash
bash scripts/cpu-vae-parity
```

The runner reports decode time, realtime factor, RSS, waveform shape, sample rate, and max/mean absolute error.

## Next gate

After VAE parity passes, wire the standalone conditioner, generator, and VAE through `TinyTAuK.generate()` and measure the complete resident CPU pipeline before beginning quantization.
