from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torchaudio

from tinytauk import TinyTAuK
from tinytauk.generator.pytorch import _FLASH_T_GRID
from tinytauk.types import GenerationRequest


def tensor_stats(value: torch.Tensor) -> dict[str, object]:
    data = value.detach().cpu()
    result: dict[str, object] = {
        "shape": list(data.shape),
        "dtype": str(data.dtype),
    }
    if data.is_floating_point():
        result.update(
            {
                "min": float(data.min()) if data.numel() else None,
                "max": float(data.max()) if data.numel() else None,
                "mean": float(data.mean()) if data.numel() else None,
                "std": float(data.float().std(unbiased=False)) if data.numel() else None,
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace TinyTAuK against upstream AuK boundaries")
    parser.add_argument("reference")
    parser.add_argument("output_dir")
    parser.add_argument("--profile", default="profiles/sweep/cpu-upstream-parity.toml")
    parser.add_argument("--seconds", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--text",
        default="The service restarted successfully and returned a clean health check.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = Path(args.reference)
    instruction = f'Say the following with the same voice: "{args.text}"'

    engine = TinyTAuK.from_config(args.profile)
    if not engine.config.conditioner.upstream_parity:
        raise RuntimeError("local parity trace requires conditioner.upstream_parity=true")
    if engine.config.conditioner.quantization != "none":
        raise RuntimeError("local parity trace requires unquantized conditioner")
    if engine.config.generator.quantization != "none":
        raise RuntimeError("local parity trace requires unquantized generator")

    conditioning = engine.condition(
        instruction,
        reference_audio=reference,
        seed=args.seed,
    )
    if conditioning.reference_latents is None or conditioning.reference_lengths is None:
        raise RuntimeError("reference conditioning did not produce VAE latents")

    generator = engine.generator
    target_len = max(
        1,
        int(math.ceil(args.seconds * generator.target_sample_rate / generator.downsample_rate)),
    )
    rng = torch.Generator(device=generator.device)
    rng.manual_seed(args.seed)
    initial_noise = torch.randn(
        (target_len, generator.latent_dim),
        generator=rng,
        device=generator.device,
        dtype=generator.dtype,
    ).unsqueeze(0)

    text = conditioning.values.to(device=generator.device, dtype=generator.dtype)
    context_mask = (
        conditioning.attention_mask.to(generator.device)
        if conditioning.attention_mask is not None
        else None
    )
    reference_latents, reference_mask = generator._reference_inputs(conditioning)
    times = torch.tensor(_FLASH_T_GRID, device=generator.device, dtype=generator.dtype)

    latent = initial_noise.clone()
    first_velocity: torch.Tensor | None = None
    try:
        for index in range(len(_FLASH_T_GRID) - 1):
            velocity = generator.transformer(
                x=latent,
                text=text,
                time=times[index],
                mask=None,
                c_mask=context_mask,
                ref=reference_latents,
                ref_mask=reference_mask,
                drop_audio_cond=False,
                drop_text=False,
                cfg_infer=False,
                cache=True,
            )
            if first_velocity is None:
                first_velocity = velocity.detach().clone()
            latent = latent + (times[index + 1] - times[index]) * velocity
    finally:
        generator.transformer.clear_cache()

    if first_velocity is None:
        raise RuntimeError("no Flux velocity was produced")

    decoded = engine.vae.decode(latent).squeeze(0).detach().cpu().to(torch.float32)
    torchaudio.save(str(output_dir / "trace.wav"), decoded, engine.vae.sample_rate)

    tensors = {
        "reference_latents": conditioning.reference_latents.detach().cpu(),
        "reference_lengths": conditioning.reference_lengths.detach().cpu(),
        "conditioner_values": conditioning.values.detach().cpu(),
        "attention_mask": (
            conditioning.attention_mask.detach().cpu()
            if conditioning.attention_mask is not None
            else torch.empty(0, dtype=torch.bool)
        ),
        "initial_noise": initial_noise.detach().cpu(),
        "first_velocity": first_velocity.detach().cpu(),
        "final_latents": latent.detach().cpu(),
        "decoded_audio": decoded,
    }
    torch.save(tensors, output_dir / "trace.pt")

    payload = {
        "implementation": "tinytauk",
        "profile": args.profile,
        "reference": str(reference),
        "instruction": instruction,
        "seconds": args.seconds,
        "seed": args.seed,
        "torch_version": torch.__version__,
        "conditioner_upstream_parity": engine.config.conditioner.upstream_parity,
        "tensors": {name: tensor_stats(value) for name, value in tensors.items()},
    }
    (output_dir / "trace.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
