from __future__ import annotations

import pytest

from haloguard.scorers.aggregator import Aggregator
from haloguard.scorers.base import SubScore


def _sub(name: str, risk: float) -> SubScore:
    return SubScore(name=name, risk=risk, detail=f"{name} detail")


def test_pass() -> None:
    risk, verdict, _ = Aggregator(0.7, 0.9).combine([_sub("a", 0.1)])
    assert risk == 0.1
    assert verdict == "PASS"


def test_flag_band() -> None:
    _, verdict, _ = Aggregator(0.7, 0.9).combine([_sub("a", 0.8)])
    assert verdict == "FLAG"


def test_block() -> None:
    _, verdict, _ = Aggregator(0.7, 0.9).combine([_sub("a", 0.95)])
    assert verdict == "BLOCK"


def test_boundaries() -> None:
    agg = Aggregator(0.7, 0.9)
    assert agg.combine([_sub("a", 0.7)])[1] == "FLAG"
    assert agg.combine([_sub("a", 0.9)])[1] == "BLOCK"


def test_worst_subscore_wins() -> None:
    risk, verdict, reason = Aggregator(0.7, 0.9).combine(
        [_sub("entailment", 0.2), _sub("consistency", 0.95)]
    )
    assert risk == 0.95
    assert verdict == "BLOCK"
    assert reason.startswith("[consistency]")


def test_empty_raises() -> None:
    with pytest.raises(ValueError):
        Aggregator(0.7, 0.9).combine([])
