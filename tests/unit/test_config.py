from pathlib import Path

import pytest

from tinytauk.config import RuntimeConfig


def test_default_configuration_uses_qualified_release_dtypes() -> None:
    config = RuntimeConfig()
    assert config == RuntimeConfig.from_dict({})
    assert config.model.model_id == "tencent/AuK-Flash"
    assert config.conditioner.device == "cpu"
    assert config.conditioner.dtype == "bf16"
    assert config.generator.dtype == "fp32"
    assert config.vae.dtype == "fp32"
    assert config.runtime.num_threads == 0


def test_cpu_profile_uses_qualified_release_path() -> None:
    config = RuntimeConfig.from_toml(Path("profiles/cpu.toml"))
    assert config.conditioner.dtype == "bf16"
    assert config.generator.dtype == "fp32"
    assert config.vae.dtype == "fp32"
    assert config.runtime.num_threads == 4


def test_unknown_configuration_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="quantizaton"):
        RuntimeConfig.from_dict({"generator": {"quantizaton": "none"}})


def test_negative_thread_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="num_threads must be non-negative"):
        RuntimeConfig.from_dict({"runtime": {"num_threads": -1}})
