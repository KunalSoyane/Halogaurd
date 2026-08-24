from __future__ import annotations

import pytest

from haloguard import Firewall


@pytest.mark.integration
def test_self_contradiction_is_detected() -> None:
    fw = Firewall(mode="consistency")
    result = fw.check(
        "When is the meeting?",
        "The meeting is scheduled for Tuesday at 3 PM. "
        "The meeting will take place on Friday at 9 AM.",
    )
    assert result.mode_used == "consistency"
    assert result.verdict in ("FLAG", "BLOCK"), result


@pytest.mark.integration
def test_consistent_response_passes() -> None:
    fw = Firewall(mode="consistency")
    result = fw.check(
        "Tell me about the Eiffel Tower.",
        "The Eiffel Tower is located in Paris. It was completed in 1889. "
        "It is one of the most visited monuments in the world.",
    )
    assert result.mode_used == "consistency"
    assert result.verdict == "PASS", result


@pytest.mark.integration
def test_single_claim_passes() -> None:
    fw = Firewall(mode="consistency")
    result = fw.check("p", "A single claim without anything to contradict it")
    assert result.verdict == "PASS"
    assert result.score == 0.0


@pytest.mark.integration
def test_auto_mode_selects_by_context() -> None:
    fw = Firewall(mode="auto")
    with_context = fw.check(
        "p", "The Eiffel Tower is in Paris.", context="The Eiffel Tower is in Paris."
    )
    without_context = fw.check("p", "The Eiffel Tower is in Paris. It was completed in 1889.")
    assert with_context.mode_used == "entailment"
    assert without_context.mode_used == "consistency"
