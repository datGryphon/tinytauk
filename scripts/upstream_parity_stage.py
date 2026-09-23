from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torchaudio
from omegaconf import OmegaConf
from safetensors import safe_open
from safetensors.torch import load_model

_FLASH_T_GRID = [0.0, 0.07612049579620361, 0.2928932309150696, 0.6173166036605835, 1.0]


def _tensor_stats(value: torch.Tensor) -> dict[str, object]:
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


def _tensor_probe(value: torch.Tensor, max_elements: int = 65536) -> torch.Tensor:
    flat = value.detach().cpu().reshape(-1)
    if flat.numel() <= max_elements:
        return flat.clone()
    step = max(1, math.ceil(flat.numel() / max_elements))
    return flat[::step][:max_elements].clone()


def _trace_encoder(vae: torch.nn.Module, audio: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {"encoder_input_probe": _tensor_probe(audio)}
    first_conv = vae.audio_encoder.generator[0].layer
    tensors["encoder_first_weight_g"] = first_conv.weight_g.detach().cpu().clone()
    tensors["encoder_first_weight_v"] = first_conv.weight_v.detach().cpu().clone()
    tensors["encoder_first_bias"] = first_conv.bias.detach().cpu().clone()

    value = audio.float()
    for index, module in enumerate(vae.audio_encoder.generator):
        value = module(value)
        tensors[f"encoder_layer_{index:02d}_probe"] = _tensor_probe(value)
    return tensors, value.detach()


def _config(ckpt: Path) -> Any:
    path = ckpt.parent / "config.yaml"
    if not path.is_file():
        raise FileNotFoundError(path)
    return OmegaConf.load(path)


def _find_tensor_key(keys: list[str], name: str) -> str:
    if name in keys:
        return name
    matches = [key for key in keys if key.endswith(f".{name}")]
    if len(matches) != 1:
        raise RuntimeError(f"Unable to identify {name!r}; matches={matches}")
    return matches[0]


def _load_fusion_parameters(ckpt: Path) -> tuple[torch.Tensor, torch.Tensor]:
    with safe_open(str(ckpt), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        weights = handle.get_tensor(_find_tensor_key(keys, "layer_weights"))
        scale = handle.get_tensor(_find_tensor_key(keys, "layer_scale"))
    return weights, scale


def _build_vae(config: Any, ckpt: Path) -> torch.nn.Module:
    from auk.model.vae import load_vae_model
    from auk.model.vae.bigvgan_flow_vae import BigVGANFlowVAEConfig

    vae_config = config.model.vae
    bundled_vae = ckpt.parent / "vae.safetensors"
    if bundled_vae.is_file():
        vae_config.vae_model_path = str(bundled_vae)
    model_init_kwargs = OmegaConf.to_container(
        vae_config.get("model_init_kwargs", OmegaConf.create({})),
        resolve=True,
    )
    vae_model_config = BigVGANFlowVAEConfig.from_dict(model_init_kwargs)
    vae = load_vae_model(
        vae_name=vae_config.vae_name,
        vae_cfg=vae_model_config,
        vae_ckpt=vae_config.vae_model_path,
        map_location="cpu",
    )
    vae = vae.to("cpu").eval()
    vae.requires_grad_(False)
    return vae


def _load_reference(path: str, target_sample_rate: int) -> torch.Tensor:
    audio, sample_rate = torchaudio.load(path)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sample_rate != target_sample_rate:
        audio = torchaudio.transforms.Resample(sample_rate, target_sample_rate)(audio)
    return audio


def _stage_vae(args: argparse.Namespace, ckpt: Path, output_dir: Path) -> None:
    config = _config(ckpt)
    vae_config = config.model.vae
    vae = _build_vae(config, ckpt)

    audio = _load_reference(args.reference, int(vae_config.target_sample_rate))
    ref_audio = audio.to("cpu").unsqueeze(0)
    encoder_tensors, reference_stats = _trace_encoder(vae, ref_audio)
    ref_latent_len = ref_audio.shape[-1] // int(vae_config.downsample_rate)
    reference_lengths = torch.tensor([ref_latent_len], dtype=torch.long)
    audio_lengths = reference_lengths * int(vae_config.downsample_rate)

    torch.manual_seed(args.seed)
    reference_latents, encoded_lengths = vae.encoding_and_normalization(
        ref_audio,
        sample_lengths=audio_lengths,
    )
    reference_lengths = torch.minimum(reference_lengths, encoded_lengths.cpu())

    payload = {
        **encoder_tensors,
        "reference_stats": reference_stats.cpu(),
        "reference_latents": reference_latents.detach().cpu(),
        "reference_lengths": reference_lengths.detach().cpu(),
    }
    torch.save(payload, output_dir / "vae.pt")
    print("VAE stage complete:", {key: _tensor_stats(value) for key, value in payload.items()})


def _stage_conditioner(args: argparse.Namespace, ckpt: Path, output_dir: Path) -> None:
    from auk.model import CFMEdit
    from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

    thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        args.qwen,
        torch_dtype=torch.bfloat16,
    )
    if thinker.visual is not None:
        del thinker.visual
        thinker.visual = None
    thinker = thinker.eval()
    thinker.requires_grad_(False)
    processor = Qwen2_5OmniProcessor.from_pretrained(args.qwen)

    instruction = f'Say the following with the same voice: "{args.text}"'
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "audio", "audio": args.reference},
            ],
        }
    ]
    cond_inputs = CFMEdit.build_cond_inputs([messages], processor)
    if hasattr(cond_inputs, "to"):
        cond_inputs = cond_inputs.to("cpu")
    else:
        cond_inputs = {
            key: value.to("cpu") if torch.is_tensor(value) else value for key, value in cond_inputs.items()
        }
    attention_mask = cond_inputs["attention_mask"].bool()

    with torch.inference_mode():
        outputs = thinker(**cond_inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("Qwen did not return hidden states")

    layer_weights, layer_scale = _load_fusion_parameters(ckpt)
    if len(hidden_states) != layer_weights.numel() + 1:
        raise RuntimeError(
            f"hidden-state count mismatch: states={len(hidden_states)} weights={layer_weights.numel()}"
        )
    hidden_dim = hidden_states[0].shape[-1]
    stacked = torch.stack(
        [F.layer_norm(hidden, [hidden_dim]) for hidden in hidden_states[1:]],
        dim=0,
    )
    weights = F.softmax(layer_weights, dim=0)
    conditioner_values = (stacked * weights[:, None, None, None]).sum(dim=0) * layer_scale

    payload = {
        "conditioner_values": conditioner_values.detach().cpu(),
        "attention_mask": attention_mask.detach().cpu(),
    }
    torch.save(payload, output_dir / "conditioner.pt")
    print("Conditioner stage complete:", {key: _tensor_stats(value) for key, value in payload.items()})


class _FluxContainer(torch.nn.Module):
    def __init__(self, transformer: torch.nn.Module) -> None:
        super().__init__()
        self.transformer = transformer


def _stage_flux(args: argparse.Namespace, ckpt: Path, output_dir: Path) -> None:
    from auk.model import Flux2Edit

    config = _config(ckpt)
    model_config = config.model
    arch = OmegaConf.to_container(model_config.arch, resolve=True)
    if not isinstance(arch, dict):
        raise TypeError("model.arch must resolve to a mapping")
    arch["attn_backend"] = "torch"
    transformer = Flux2Edit(**arch, latent_dim=int(model_config.vae.latent_dim))
    transformer = transformer.to(device="cpu", dtype=torch.float32).eval()
    transformer.requires_grad_(False)

    container = _FluxContainer(transformer)
    missing, unexpected = load_model(container, str(ckpt), strict=False, device="cpu")
    if missing:
        raise RuntimeError(f"Flux checkpoint missing keys: {sorted(missing)[:10]}")
    allowed_unexpected = {"layer_weights", "layer_scale"}
    bad_unexpected = sorted(set(unexpected) - allowed_unexpected)
    if bad_unexpected:
        raise RuntimeError(f"Flux checkpoint unexpected keys: {bad_unexpected[:10]}")

    vae_data = torch.load(output_dir / "vae.pt", map_location="cpu", weights_only=True)
    conditioner_data = torch.load(
        output_dir / "conditioner.pt",
        map_location="cpu",
        weights_only=True,
    )
    reference_latents = vae_data["reference_latents"].to(torch.float32)
    reference_lengths = vae_data["reference_lengths"].to(torch.long)
    positions = torch.arange(reference_latents.shape[1]).unsqueeze(0)
    reference_mask = positions < reference_lengths.unsqueeze(1)
    conditioner_values = conditioner_data["conditioner_values"].to(torch.float32)
    attention_mask = conditioner_data["attention_mask"].bool()

    target_sample_rate = int(model_config.vae.target_sample_rate)
    downsample_rate = int(model_config.vae.downsample_rate)
    latent_dim = int(model_config.vae.latent_dim)
    target_len = max(1, int(math.ceil(args.seconds * target_sample_rate / downsample_rate)))

    torch.manual_seed(args.seed)
    initial_noise = torch.randn(
        (1, target_len, latent_dim),
        device="cpu",
        dtype=reference_latents.dtype,
    )
    times = torch.tensor(_FLASH_T_GRID, dtype=torch.float32)
    latent = initial_noise.clone()
    first_velocity: torch.Tensor | None = None
    try:
        with torch.inference_mode():
            for index in range(len(_FLASH_T_GRID) - 1):
                velocity = transformer(
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
        transformer.clear_cache()

    if first_velocity is None:
        raise RuntimeError("Flux stage produced no velocity")
    payload = {
        "initial_noise": initial_noise.detach().cpu(),
        "first_velocity": first_velocity.detach().cpu(),
        "final_latents": latent.detach().cpu(),
    }
    torch.save(payload, output_dir / "flux.pt")
    print("Flux stage complete:", {key: _tensor_stats(value) for key, value in payload.items()})


def _stage_decode(args: argparse.Namespace, ckpt: Path, output_dir: Path) -> None:
    config = _config(ckpt)
    vae = _build_vae(config, ckpt)
    flux_data = torch.load(output_dir / "flux.pt", map_location="cpu", weights_only=True)
    latent = flux_data["final_latents"].to(torch.float32)

    with torch.inference_mode():
        decode_latent = vae.denormalize(latent)
        decode_latent = decode_latent.permute(0, 2, 1)
        decoded = vae.inference_from_latents(decode_latent).cpu()
    if decoded.ndim == 3:
        decoded = decoded.squeeze(0)
    decoded = decoded.to(torch.float32)
    torchaudio.save(
        str(output_dir / "trace.wav"),
        decoded,
        int(config.model.vae.target_sample_rate),
    )

    vae_data = torch.load(output_dir / "vae.pt", map_location="cpu", weights_only=True)
    conditioner_data = torch.load(
        output_dir / "conditioner.pt",
        map_location="cpu",
        weights_only=True,
    )
    tensors = {
        **vae_data,
        **conditioner_data,
        **flux_data,
        "decoded_audio": decoded.detach().cpu(),
    }
    torch.save(tensors, output_dir / "trace.pt")

    instruction = f'Say the following with the same voice: "{args.text}"'
    metadata = {
        "implementation": "upstream-auk-staged",
        "checkpoint": str(ckpt),
        "qwen": args.qwen,
        "reference": args.reference,
        "instruction": instruction,
        "seconds": args.seconds,
        "seed": args.seed,
        "torch_version": torch.__version__,
        "staged": True,
        "tensors": {name: _tensor_stats(value) for name, value in tensors.items()},
    }
    (output_dir / "trace.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("Decode/assembly stage complete; wrote trace.pt, trace.json, and trace.wav")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one memory-bounded upstream AuK parity stage")
    parser.add_argument("stage", choices=("vae", "conditioner", "flux", "decode"))
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

    if args.stage == "vae":
        _stage_vae(args, ckpt, output_dir)
    elif args.stage == "conditioner":
        _stage_conditioner(args, ckpt, output_dir)
    elif args.stage == "flux":
        _stage_flux(args, ckpt, output_dir)
    else:
        _stage_decode(args, ckpt, output_dir)


if __name__ == "__main__":
    main()
