from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

Device = str
DType = Literal["fp32", "bf16", "fp16"]
Quantization = Literal["none", "int8", "int8-weight-only", "int4", "int8-act-int4-weight"]

_VALID_DTYPES = {"fp32", "bf16", "fp16"}
_VALID_QUANTIZATION = {"none", "int8", "int8-weight-only", "int4", "int8-act-int4-weight"}


@dataclass(frozen=True, slots=True)
class ModelConfig:
    model_id: str = "tencent/AuK-Flash"
    qwen_model_id: str = "Qwen/Qwen2.5-Omni-3B"


@dataclass(frozen=True, slots=True)
class ComponentConfig:
    backend: str
    device: Device = "cpu"
    dtype: DType = "bf16"
    quantization: Quantization = "none"
    compile: bool = False
    compile_mode: str = "default"
    compile_dynamic: bool = True
    compile_warmup_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    seed: int = 1234
    num_threads: int = 0


def _table(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise TypeError(f"[{name}] must be a TOML table")
    return value


def _string(table: dict[str, Any], key: str, default: str) -> str:
    value = table.get(key, default)
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string")
    return value


def _integer(table: dict[str, Any], key: str, default: int) -> int:
    value = table.get(key, default)
    if not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    return value


def _number(table: dict[str, Any], key: str, default: float) -> float:
    value = table.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be a number")
    return float(value)


def _boolean(table: dict[str, Any], key: str, default: bool) -> bool:
    value = table.get(key, default)
    if not isinstance(value, bool):
        raise TypeError(f"{key} must be a boolean")
    return value


def _component(
    table: dict[str, Any],
    *,
    default_backend: str,
    default_dtype: DType = "bf16",
) -> ComponentConfig:
    dtype = _string(table, "dtype", default_dtype)
    if dtype not in _VALID_DTYPES:
        raise ValueError(f"unsupported dtype: {dtype}")

    quantization = _string(table, "quantization", "none")
    if quantization not in _VALID_QUANTIZATION:
        raise ValueError(f"unsupported quantization: {quantization}")

    compile_warmup_seconds = _number(table, "compile_warmup_seconds", 0.0)
    if compile_warmup_seconds < 0:
        raise ValueError("compile_warmup_seconds must be non-negative")

    return ComponentConfig(
        backend=_string(table, "backend", default_backend),
        device=_string(table, "device", "cpu"),
        dtype=cast(DType, dtype),
        quantization=cast(Quantization, quantization),
        compile=_boolean(table, "compile", False),
        compile_mode=_string(table, "compile_mode", "default"),
        compile_dynamic=_boolean(table, "compile_dynamic", True),
        compile_warmup_seconds=compile_warmup_seconds,
    )


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    conditioner: ComponentConfig = field(default_factory=lambda: ComponentConfig(backend="transformers"))
    generator: ComponentConfig = field(default_factory=lambda: ComponentConfig(backend="pytorch"))
    vae: ComponentConfig = field(default_factory=lambda: ComponentConfig(backend="pytorch", dtype="fp32"))
    runtime: ExecutionConfig = field(default_factory=ExecutionConfig)

    @classmethod
    def from_toml(cls, path: str | Path) -> RuntimeConfig:
        with Path(path).open("rb") as handle:
            return cls.from_dict(tomllib.load(handle))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RuntimeConfig:
        model = _table(raw, "model")
        runtime = _table(raw, "runtime")

        return cls(
            model=ModelConfig(
                model_id=_string(model, "model_id", "tencent/AuK-Flash"),
                qwen_model_id=_string(model, "qwen_model_id", "Qwen/Qwen2.5-Omni-3B"),
            ),
            conditioner=_component(
                _table(raw, "conditioner"),
                default_backend="transformers",
            ),
            generator=_component(
                _table(raw, "generator"),
                default_backend="pytorch",
            ),
            vae=_component(
                _table(raw, "vae"),
                default_backend="pytorch",
                default_dtype="fp32",
            ),
            runtime=ExecutionConfig(
                seed=_integer(runtime, "seed", 1234),
                num_threads=_integer(runtime, "num_threads", 0),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
