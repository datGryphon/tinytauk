from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ErrorRate:
    """Edit-distance score and the reference-unit count used as its denominator."""

    errors: int
    reference_units: int
    rate: float


def normalize_text(text: str) -> str:
    """Normalize transcript text for speech-intelligibility scoring."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    cleaned = "".join(char if char.isalnum() or char.isspace() else " " for char in normalized)
    return " ".join(cleaned.split())


def _edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_unit in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_unit in enumerate(hypothesis, start=1):
            substitution = previous[hyp_index - 1] + int(ref_unit != hyp_unit)
            deletion = previous[hyp_index] + 1
            insertion = current[hyp_index - 1] + 1
            current.append(min(substitution, deletion, insertion))
        previous = current
    return previous[-1]


def _rate(reference: Sequence[str], hypothesis: Sequence[str]) -> ErrorRate:
    errors = _edit_distance(reference, hypothesis)
    reference_units = len(reference)
    return ErrorRate(
        errors=errors,
        reference_units=reference_units,
        rate=0.0 if reference_units == 0 else errors / reference_units,
    )


def word_error_rate(reference: str, hypothesis: str) -> ErrorRate:
    """Compute proper Levenshtein WER after speech-oriented normalization."""

    return _rate(normalize_text(reference).split(), normalize_text(hypothesis).split())


def char_error_rate(reference: str, hypothesis: str) -> ErrorRate:
    """Compute CER after normalization, ignoring spaces between words."""

    reference_chars = list(normalize_text(reference).replace(" ", ""))
    hypothesis_chars = list(normalize_text(hypothesis).replace(" ", ""))
    return _rate(reference_chars, hypothesis_chars)


def combine_error_rates(values: Iterable[ErrorRate]) -> ErrorRate:
    """Combine utterance scores by summing edit errors and reference units."""

    errors = 0
    reference_units = 0
    for value in values:
        errors += value.errors
        reference_units += value.reference_units
    return ErrorRate(
        errors=errors,
        reference_units=reference_units,
        rate=0.0 if reference_units == 0 else errors / reference_units,
    )
