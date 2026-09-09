from pathlib import Path

import tinytauk.reference as reference


def test_messages_builds_instruction_turn() -> None:
    messages = reference._messages("hello")
    assert messages == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "hello"}],
        }
    ]


def test_memory_sample_is_non_negative() -> None:
    sample = reference.memory_sample()
    assert sample.current_rss_mb >= 0
    assert sample.peak_rss_mb >= sample.current_rss_mb or sample.peak_rss_mb > 0


def test_reference_config_defaults_to_ignored_results_dir() -> None:
    config = reference.ReferenceConfig()
    assert config.output_dir == Path("benchmarks/results/reference-oracle")
    assert config.gen_seconds == 3.0
    assert config.seed == 42
