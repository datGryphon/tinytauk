from __future__ import annotations

import pytest

from tinytauk.quality import ErrorRate, char_error_rate, combine_error_rates, normalize_text, word_error_rate


def test_normalize_text_for_speech_scoring() -> None:
    assert normalize_text("Hello, CUDA! Twenty-four.") == "hello cuda twenty four"


def test_word_error_rate_counts_insertions() -> None:
    score = word_error_rate("one two", "one bright two")
    assert score.errors == 1
    assert score.reference_units == 2
    assert score.rate == pytest.approx(0.5)


def test_word_error_rate_counts_substitution_and_deletion() -> None:
    score = word_error_rate("one two three", "one four")
    assert score.errors == 2
    assert score.reference_units == 3
    assert score.rate == pytest.approx(2 / 3)


def test_char_error_rate_ignores_spaces_and_punctuation() -> None:
    score = char_error_rate("A B!", "ab")
    assert score == ErrorRate(errors=0, reference_units=2, rate=0.0)


def test_combine_error_rates_weights_by_reference_length() -> None:
    combined = combine_error_rates(
        [
            ErrorRate(errors=1, reference_units=2, rate=0.5),
            ErrorRate(errors=1, reference_units=8, rate=0.125),
        ]
    )
    assert combined.errors == 2
    assert combined.reference_units == 10
    assert combined.rate == pytest.approx(0.2)
