from pathlib import Path

import torch

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


def test_save_tensor_records_parity_metadata(tmp_path: Path) -> None:
    tensor = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    artifact = reference._save_tensor(tmp_path / "tensor.pt", tensor)

    assert Path(artifact.path).is_file()
    assert artifact.shape == [1, 3, 4]
    assert artifact.dtype == "torch.float32"
    assert artifact.numel == 12
    assert len(artifact.sha256) == 64
    assert torch.equal(torch.load(artifact.path, weights_only=True), tensor)


def test_tensor_digest_is_stable_and_content_sensitive() -> None:
    left = torch.tensor([1, 2, 3], dtype=torch.int64)
    right = torch.tensor([1, 2, 4], dtype=torch.int64)

    assert reference._tensor_digest(left) == reference._tensor_digest(left.clone())
    assert reference._tensor_digest(left) != reference._tensor_digest(right)
