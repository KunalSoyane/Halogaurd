from __future__ import annotations

import asyncio

import pytest

from haloguard import Firewall, FirewallInput


@pytest.mark.integration
def test_check_batch() -> None:
    fw = Firewall(mode="entailment")
    items = [
        FirewallInput(
            prompt="p",
            response="The Eiffel Tower is in Paris.",
            context="The Eiffel Tower is located in Paris and was completed in 1889.",
        ),
        FirewallInput(
            prompt="p",
            response="The Eiffel Tower was completed in 1923.",
            context="The Eiffel Tower is located in Paris and was completed in 1889.",
        ),
    ]
    results = fw.check_batch(items)
    assert [r.verdict for r in results] == ["PASS", "BLOCK"]


@pytest.mark.integration
def test_acheck() -> None:
    fw = Firewall(mode="entailment")
    result = asyncio.run(
        fw.acheck(
            "p",
            response="The Eiffel Tower is in Paris.",
            context="The Eiffel Tower is located in Paris and was completed in 1889.",
        )
    )
    assert result.verdict == "PASS"
