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

## Gate 2: Flux2Edit sampled latent

The second gate ports the AuK-Flash Flux2Edit transformer and fixed four-step Euler sampler into TinyTAuK. It deliberately targets the oracle fixture that exists today: batch-one, instruction-only generation with no reference audio and CFG disabled.

The implementation consumes the captured `conditioning.pt` and `context_mask.pt` directly, so a failure here is isolated from Qwen. It must reproduce `sampled_latent.pt` with shape `[1, 150, 64]`.

Run:

```bash
bash scripts/cpu-generator-parity
```

Gate 2 passes on exact tensor equality. If exact equality fails but the shape is correct, use the reported max/mean absolute error to locate numerical or architectural divergence before adding the VAE.

## Next gate

Only after Gate 2 passes should the standalone VAE decoder and full `TinyTAuK.generate()` path be enabled.
