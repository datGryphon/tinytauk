from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import torch
import torchaudio
from auk.infer.infer_auk import AukInfer
from safetensors.torch import load_model

_FLASH_T_GRID = [0.0, 0.07612049579620361, 0.2928932309150696, 0.6173166036605835, 1.0]


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


def _install_memory_safe_safetensors_loader() -> None:
    """Avoid upstream's transient full checkpoint tensor dict during parity tracing.

    The released AuK loader uses load_file(...)+load_state_dict(...), which keeps
    the already-built model and the full checkpoint tensor dictionary resident at
    the same time. For this diagnostic we load the same safetensors directly onto
    the same model with safetensors.torch.load_model(). This changes only loading
    peak memory, not the model architecture, tensor values, or inference path.
    """

    original = AukInfer._load_ema_weights
    logger = logging.getLogger("auk.infer.infer_auk")

    def memory_safe_load(self: AukInfer, model: torch.nn.Module, ckpt_path: str) -> None:
        if not ckpt_path.endswith(".safetensors"):
            original(self, model, ckpt_path)
            return

        logger.info("Loading model checkpoint from %s with direct safetensors model loader ...", ckpt_path)
        missing, unexpected = load_model(
            model,
            ckpt_path,
            strict=False,
            device="cpu",
        )
        n_missing_te = sum(1 for key in missing if key.startswith("text_encoder."))
        n_missing_other = len(missing) - n_missing_te
        logger.info(
            "Loaded EMA weights | missing=%d (text_encoder.*=%d, other=%d) | unexpected=%d",
            len(missing),
            n_missing_te,
            n_missing_other,
            len(unexpected),
        )
        if n_missing_other:
            examples = [key for key in missing if not key.startswith("text_encoder.")][:10]
            logger.warning("Missing non-text-encoder keys: %s", examples)
        if unexpected:
            logger.warning("Unexpected keys in checkpoint: %s", unexpected[:10])

    AukInfer._load_ema_weights = memory_safe_load


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace upstream AuK at TinyTAuK parity boundaries")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--qwen", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seconds", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--text",
        default="The service restarted successfully and returned a clean health check.",
    )
    args = parser.parse_args()

    ckpt = Path(args.ckpt)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    instruction = f'Say the following with the same voice: "{args.text}"'
    config_path = ckpt.parent / "config.yaml"

    _install_memory_safe_safetensors_loader()
    engine = AukInfer(
        config_path=str(config_path),
        ckpt_path=str(ckpt),
        device="cpu",
        dtype="bf16",
        qwen_path=args.qwen,
    )
    if not engine.is_flash:
        raise RuntimeError("parity trace expects AuK-Flash")

    ref_audio, _ = engine._load_audio(args.reference)
    ref_audio = ref_audio.to(engine.device).unsqueeze(0)
    reference_stats = engine.vae_model.audio_encoder(ref_audio.float()).detach()
    ref_latent_len = ref_audio.shape[-1] // engine.downsample_rate
    ref_lengths = torch.tensor([ref_latent_len], dtype=torch.long, device=engine.device)
    audio_lengths = ref_lengths * engine.downsample_rate

    # TinyTAuK's reference encoder uses a dedicated generator seeded with the
    # request seed. Seed upstream immediately before its equivalent randn_like
    # so this diagnostic compares the same stochastic VAE sample.
    torch.manual_seed(args.seed)
    reference_latents, encoded_lengths = engine.vae_model.encoding_and_normalization(
        ref_audio,
        sample_lengths=audio_lengths,
    )
    ref_lengths = torch.minimum(ref_lengths, encoded_lengths.to(ref_lengths.device))
    positions = torch.arange(reference_latents.shape[1], device=engine.device).unsqueeze(0)
    reference_mask = positions < ref_lengths.unsqueeze(1)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "audio", "audio": args.reference},
            ],
        }
    ]
    cond_inputs = engine.model.build_cond_inputs([messages], engine.model.text_processor)
    conditioner_values, attention_mask = engine.model.encode_text(cond_inputs, engine.device)

    target_len = max(
        1,
        int(math.ceil(args.seconds * engine.target_sample_rate / engine.downsample_rate)),
    )
    torch.manual_seed(args.seed)
    initial_noise = torch.randn(
        (1, target_len, engine.latent_dim),
        device=engine.device,
        dtype=reference_latents.dtype,
    )

    times = torch.tensor(_FLASH_T_GRID, device=engine.device, dtype=torch.float32)
    latent = initial_noise.clone()
    first_velocity: torch.Tensor | None = None
    try:
        for index in range(len(_FLASH_T_GRID) - 1):
            velocity = engine.model.transformer(
                x=latent,
                text=conditioner_values,
                time=times[index],
                mask=None,
                c_mask=attention_mask,
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
        engine.model.transformer.clear_cache()

    if first_velocity is None:
        raise RuntimeError("no Flux velocity was produced")

    decode_latent = engine.vae_model.denormalize(latent)
    decode_latent = decode_latent.permute(0, 2, 1)
    decoded = engine.vae_model.inference_from_latents(decode_latent).cpu()
    if decoded.ndim == 3:
        decoded = decoded.squeeze(0)
    decoded = decoded.to(torch.float32)
    torchaudio.save(str(output_dir / "trace.wav"), decoded, engine.target_sample_rate)

    tensors = {
        "reference_stats": reference_stats.detach().cpu(),
        "reference_latents": reference_latents.detach().cpu(),
        "reference_lengths": ref_lengths.detach().cpu(),
        "conditioner_values": conditioner_values.detach().cpu(),
        "attention_mask": attention_mask.detach().cpu(),
        "initial_noise": initial_noise.detach().cpu(),
        "first_velocity": first_velocity.detach().cpu(),
        "final_latents": latent.detach().cpu(),
        "decoded_audio": decoded.detach().cpu(),
    }
    torch.save(tensors, output_dir / "trace.pt")

    payload = {
        "implementation": "upstream-auk",
        "checkpoint": str(ckpt),
        "qwen": args.qwen,
        "reference": args.reference,
        "instruction": instruction,
        "seconds": args.seconds,
        "seed": args.seed,
        "torch_version": torch.__version__,
        "diagnostic_reference_seeded": True,
        "memory_safe_checkpoint_loader": True,
        "tensors": {name: tensor_stats(value) for name, value in tensors.items()},
    }
    (output_dir / "trace.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
