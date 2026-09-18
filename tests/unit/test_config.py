from pathlib import Path

from tinytauk.config import RuntimeConfig


def test_cpu_baseline_profile_loads() -> None:
    config = RuntimeConfig.from_toml(Path("profiles/cpu-baseline.toml"))
    assert config.model.model_id == "tencent/AuK-Flash"
    assert config.conditioner.device == "cpu"
    assert config.conditioner.dtype == "bf16"
    assert config.conditioner.quantization == "none"
    assert config.conditioner.upstream_parity is False
    assert config.generator.backend == "pytorch"
    assert config.generator.dtype == "fp32"
    assert config.generator.quantization == "none"
    assert config.vae.dtype == "fp32"
    assert config.vae.quantization == "none"
    assert config.vae.compile is False
    assert config.runtime.num_threads == 4


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


def test_conditioner_upstream_parity_loads() -> None:
    config = RuntimeConfig.from_dict(
        {
            "conditioner": {
                "backend": "transformers",
                "dtype": "fp32",
                "quantization": "none",
                "upstream_parity": True,
            }
        }
    )
    assert config.conditioner.upstream_parity is True


def test_upstream_parity_profile_enables_parity() -> None:
    config = RuntimeConfig.from_toml(Path("profiles/sweep/cpu-upstream-parity.toml"))
    assert config.conditioner.quantization == "none"
    assert config.conditioner.upstream_parity is True
    assert config.generator.quantization == "none"
    assert config.vae.quantization == "none"
    assert config.vae.compile is False


def test_cpu_profile_uses_qualified_release_path() -> None:
    config = RuntimeConfig.from_toml(Path("profiles/cpu.toml"))
    assert config.conditioner.dtype == "bf16"
    assert config.conditioner.quantization == "none"
    assert config.generator.dtype == "fp32"
    assert config.generator.quantization == "none"
    assert config.vae.dtype == "fp32"
    assert config.vae.quantization == "none"
    assert config.vae.compile is False
    assert config.runtime.num_threads == 4


def test_dynamic_flux_profiles_load() -> None:
    for filename in ("cpu-q16.toml", "cpu-q16-fp32.toml", "cpu-opt-flux-int8.toml"):
        config = RuntimeConfig.from_toml(Path("profiles/sweep") / filename)
        assert config.conditioner.quantization == "none"
        assert config.generator.quantization == "int8"
        assert config.vae.quantization == "none"


def test_non_torchao_experiment_profiles_load() -> None:
    for filename in ("cpu-final-flux-bf16.toml", "cpu-opt-vae-bf16.toml"):
        config = RuntimeConfig.from_toml(Path("profiles/sweep") / filename)
        assert config.conditioner.quantization == "none"
        assert config.generator.quantization == "none"


def test_removed_torchao_quantization_is_rejected() -> None:
    for quantization in ("int8-weight-only", "int4-weight-only", "int8-dynamic"):
        try:
            RuntimeConfig.from_dict(
                {"conditioner": {"backend": "transformers", "quantization": quantization}}
            )
        except ValueError as exc:
            assert "unsupported quantization" in str(exc)
        else:
            raise AssertionError(f"expected {quantization!r} to be rejected")


def test_unknown_configuration_keys_are_rejected() -> None:
    try:
        RuntimeConfig.from_dict({"generator": {"quantizaton": "int8"}})
    except ValueError as exc:
        assert "quantizaton" in str(exc)
    else:
        raise AssertionError("expected unknown configuration key to fail")


def test_negative_thread_count_is_rejected() -> None:
    try:
        RuntimeConfig.from_dict({"runtime": {"num_threads": -1}})
    except ValueError as exc:
        assert "num_threads must be non-negative" in str(exc)
    else:
        raise AssertionError("expected negative thread count to fail")
