from pathlib import Path

from tinytauk.config import RuntimeConfig


def test_cpu_baseline_profile_loads() -> None:
    config = RuntimeConfig.from_toml(Path("profiles/cpu-baseline.toml"))
    assert config.model.model_id == "tencent/AuK-Flash"
    assert config.conditioner.device == "cpu"
    assert config.generator.backend == "pytorch"
    assert config.vae.dtype == "fp32"
    assert config.vae.compile is False
    assert config.vae.compile_mode == "default"
    assert config.vae.compile_dynamic is True


def test_component_compile_settings_load() -> None:
    config = RuntimeConfig.from_dict(
        {
            "vae": {
                "backend": "pytorch",
                "dtype": "fp32",
                "compile": True,
                "compile_mode": "reduce-overhead",
                "compile_dynamic": False,
            }
        }
    )
    assert config.vae.compile is True
    assert config.vae.compile_mode == "reduce-overhead"
    assert config.vae.compile_dynamic is False


def test_cpu_profile_uses_memory_optimized_conditioner() -> None:
    config = RuntimeConfig.from_toml(Path("profiles/cpu.toml"))
    assert config.conditioner.dtype == "fp32"
    assert config.conditioner.quantization == "int8-weight-only"
    assert config.generator.quantization == "int8"
    assert config.vae.compile is True
