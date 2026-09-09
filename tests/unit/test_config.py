from pathlib import Path

from tinytauk.config import RuntimeConfig


def test_cpu_profile_loads() -> None:
    config = RuntimeConfig.from_toml(Path("profiles/cpu-baseline.toml"))
    assert config.model.model_id == "tencent/AuK-Flash"
    assert config.conditioner.device == "cpu"
    assert config.generator.backend == "pytorch"
    assert config.vae.dtype == "fp32"
