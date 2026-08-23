from __future__ import annotations

from haloguard.scorers.consistency import split_sentences


def test_split_basic() -> None:
    text = "The meeting is on Tuesday. It starts at 3 PM! Will you attend? Yes."
    assert split_sentences(text) == [
        "The meeting is on Tuesday.",
        "It starts at 3 PM!",
        "Will you attend?",
        "Yes.",
    ]


def test_split_ignores_blank() -> None:
    assert split_sentences("   ") == []


def test_single_claim() -> None:
    assert split_sentences("Just one claim") == ["Just one claim"]
