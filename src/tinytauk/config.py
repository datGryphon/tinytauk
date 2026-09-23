from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

Device = str
DType = Literal["fp32", "bf16", "fp16"]

_VALID_DTYPES = {"fp32", "bf16", "fp16"}
_COMPONENT_KEYS = {
    "device",
    "dtype",
}


@dataclass(frozen=True, slots=True)
class ModelConfig:
    model_id: str = "tencent/AuK-Flash"
    qwen_model_id: str = "Qwen/Qwen2.5-Omni-3B"


@dataclass(frozen=True, slots=True)
class ComponentConfig:
    device: Device = "cpu"
    dtype: DType = "bf16"


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    seed: int = 1234
    num_threads: int = 0


def _reject_unknown(table: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ValueError(f"unknown keys in {name}: {', '.join(unknown)}")


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
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{key} must be an integer")
    return value


def _component(
    table: dict[str, Any],
    *,
    name: str,
    default_dtype: DType = "bf16",
) -> ComponentConfig:
    _reject_unknown(table, _COMPONENT_KEYS, f"[{name}]")
    dtype = _string(table, "dtype", default_dtype)
    if dtype not in _VALID_DTYPES:
        raise ValueError(f"unsupported dtype: {dtype}")

    return ComponentConfig(
        device=_string(table, "device", "cpu"),
        dtype=cast(DType, dtype),
    )


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    conditioner: ComponentConfig = field(default_factory=ComponentConfig)
    generator: ComponentConfig = field(default_factory=lambda: ComponentConfig(dtype="fp32"))
    vae: ComponentConfig = field(default_factory=lambda: ComponentConfig(dtype="fp32"))
    runtime: ExecutionConfig = field(default_factory=ExecutionConfig)

    @classmethod
    def from_toml(cls, path: str | Path) -> RuntimeConfig:
        with Path(path).open("rb") as handle:
            return cls.from_dict(tomllib.load(handle))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RuntimeConfig:
        _reject_unknown(raw, {"model", "conditioner", "generator", "vae", "runtime"}, "root")
        model = _table(raw, "model")
        runtime = _table(raw, "runtime")
        _reject_unknown(model, {"model_id", "qwen_model_id"}, "[model]")
        _reject_unknown(runtime, {"seed", "num_threads"}, "[runtime]")

        num_threads = _integer(runtime, "num_threads", 0)
        if num_threads < 0:
            raise ValueError("num_threads must be non-negative")

        return cls(
            model=ModelConfig(
                model_id=_string(model, "model_id", "tencent/AuK-Flash"),
                qwen_model_id=_string(model, "qwen_model_id", "Qwen/Qwen2.5-Omni-3B"),
            ),
            conditioner=_component(
                _table(raw, "conditioner"),
                name="conditioner",
            ),
            generator=_component(
                _table(raw, "generator"),
                name="generator",
                default_dtype="fp32",
            ),
            vae=_component(
                _table(raw, "vae"),
                name="vae",
                default_dtype="fp32",
            ),
            runtime=ExecutionConfig(
                seed=_integer(runtime, "seed", 1234),
                num_threads=num_threads,
            ),
        )
