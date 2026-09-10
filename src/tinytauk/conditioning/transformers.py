from __future__ import annotations

from collections.abc import Iterable
from importlib import import_module
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

from tinytauk.config import ComponentConfig, ModelConfig
from tinytauk.types import Conditioning, GenerationRequest

_NO_PROMPT_AUDIO = "|<no_prompt_audio>|"
_DTYPE_MAP: dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def _find_tensor_key(keys: Iterable[str], name: str) -> str:
    key_list = list(keys)
    if name in key_list:
        return name
    matches = [key for key in key_list if key.endswith(f".{name}")]
    if len(matches) != 1:
        raise RuntimeError(f"Unable to identify {name!r} in AuK checkpoint; matches={matches}")
    return matches[0]


def _load_fusion_parameters(checkpoint_path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        weights = handle.get_tensor(_find_tensor_key(keys, "layer_weights")).to(torch.float32)
        scale = handle.get_tensor(_find_tensor_key(keys, "layer_scale")).to(torch.float32)
    return weights, scale


def _torchao_int8_weight_only_config() -> Any:
    try:
        torchao_quantization: Any = import_module("torchao.quantization")
        from transformers import TorchAoConfig
    except ImportError as exc:
        raise RuntimeError("INT8 weight-only conditioning requires torchao; run `uv sync`") from exc

    # Reference-audio conditioning depends on the audio tower, so it remains FP32.
    return TorchAoConfig(
        quant_type=torchao_quantization.Int8WeightOnlyConfig(),
        modules_to_not_convert=["audio_tower", "lm_head"],
    )


class TransformersConditioner:
    """Qwen2.5-Omni conditioner with AuK learned hidden-state fusion.

    ``upstream_parity`` reproduces AuK's BF16-load-then-FP32-promotion behavior
    for exact reference comparisons.
    """

    def __init__(
        self,
        model: ModelConfig,
        config: ComponentConfig,
        *,
        auk_checkpoint: str | Path,
        upstream_parity: bool = False,
    ) -> None:
        if config.quantization not in {"none", "int8-weight-only"}:
            raise ValueError("TransformersConditioner supports only none or int8-weight-only quantization")
        if upstream_parity and config.quantization != "none":
            raise ValueError("upstream parity requires an unquantized conditioner")

        self.model_config = model
        self.config = config
        self.device = torch.device(config.device)
        if config.quantization == "int8-weight-only" and self.device.type != "cpu":
            raise ValueError("INT8 weight-only conditioning requires CPU")

        load_dtype = torch.bfloat16 if upstream_parity else _DTYPE_MAP[config.dtype]
        quantization_config = (
            _torchao_int8_weight_only_config() if config.quantization == "int8-weight-only" else None
        )
        thinker: Any = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model.qwen_model_id,
            dtype=load_dtype,
            quantization_config=quantization_config,
        )
        if thinker.visual is not None:
            del thinker.visual
            thinker.visual = None

        if upstream_parity:
            thinker = thinker.to(torch.float32)
        if config.quantization == "none":
            thinker = thinker.to(self.device)
        self.thinker: Any = thinker.eval()
        self.thinker.requires_grad_(False)

        self.processor: Any = Qwen2_5OmniProcessor.from_pretrained(model.qwen_model_id)
        layer_weights, layer_scale = _load_fusion_parameters(auk_checkpoint)
        self.layer_weights = layer_weights.to(self.device)
        self.layer_scale = layer_scale.to(self.device)

        expected_layers = int(self.thinker.config.text_config.num_hidden_layers)
        if self.layer_weights.numel() != expected_layers:
            raise RuntimeError(
                "AuK layer fusion does not match Qwen layer count: "
                f"weights={self.layer_weights.numel()} qwen_layers={expected_layers}"
            )

    @staticmethod
    def _messages(request: GenerationRequest) -> list[dict[str, Any]]:
        text = request.instruction
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        if request.reference_audio is None:
            content[0]["text"] = f"{text}{_NO_PROMPT_AUDIO}"
        else:
            content.append({"type": "audio", "audio": str(request.reference_audio)})
        return [{"role": "user", "content": content}]

    def _build_inputs(self, request: GenerationRequest) -> Any:
        messages = self._messages(request)
        formatted = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        if request.reference_audio is None:
            return self.processor(
                text=[formatted],
                padding=True,
                return_tensors="pt",
            )

        process_mm_info: Any = import_module("qwen_omni_utils").process_mm_info
        audios, images, videos = process_mm_info([messages], use_audio_in_video=True)
        return self.processor(
            text=[formatted],
            audio=audios,
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
            use_audio_in_video=True,
        )

    def encode(self, request: GenerationRequest) -> Conditioning:
        cond_inputs: Any = self._build_inputs(request)
        if hasattr(cond_inputs, "to"):
            cond_inputs = cond_inputs.to(self.device)
        else:
            cond_inputs = {
                key: value.to(self.device) if torch.is_tensor(value) else value
                for key, value in cond_inputs.items()
            }

        attention_mask: torch.Tensor = cond_inputs["attention_mask"]
        with torch.inference_mode():
            outputs: Any = self.thinker(
                **cond_inputs,
                output_hidden_states=True,
            )

        hidden_states: Any = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Qwen did not return hidden states")
        if len(hidden_states) != self.layer_weights.numel() + 1:
            raise RuntimeError(
                "Unexpected Qwen hidden-state count: "
                f"states={len(hidden_states)} weights={self.layer_weights.numel()}"
            )

        hidden_dim = hidden_states[0].shape[-1]
        stacked = torch.stack(
            [F.layer_norm(hidden, [hidden_dim]) for hidden in hidden_states[1:]],
            dim=0,
        )
        weights = F.softmax(self.layer_weights, dim=0)
        fused = (stacked * weights[:, None, None, None]).sum(dim=0) * self.layer_scale
        return Conditioning(values=fused, attention_mask=attention_mask.bool())
