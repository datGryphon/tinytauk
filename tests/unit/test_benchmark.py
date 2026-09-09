from tinytauk.benchmark import BenchmarkResult


def test_rtf() -> None:
    result = BenchmarkResult(
        host="bean",
        profile="cpu",
        target_seconds=10.0,
        generated_seconds=10.0,
        wall_seconds=25.0,
    )
    assert result.rtf == 2.5
    assert result.to_dict()["rtf"] == 2.5
