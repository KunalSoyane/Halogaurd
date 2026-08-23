"""Combines sub-scores into a single risk, verdict, and reason."""

from __future__ import annotations

from collections.abc import Sequence

from haloguard.core.result import VERDICT_BLOCK, VERDICT_FLAG, VERDICT_PASS
from haloguard.scorers.base import SubScore


class Aggregator:
    """Maps combined sub-scores to a verdict.

    Sub-scores are combined conservatively (max risk wins), so a single
    strong signal is enough to flag a response -- defense-in-depth rather
    than averaging a signal away.
    """

    def __init__(self, threshold: float, block_threshold: float) -> None:
        self.threshold = threshold
        self.block_threshold = block_threshold

    def combine(self, subscores: Sequence[SubScore]) -> tuple[float, str, str]:
        """Return (risk, verdict, reason) for one or more sub-scores."""
        if not subscores:
            raise ValueError("at least one sub-score is required")
        worst = max(subscores, key=lambda s: s.risk)
        risk = worst.risk
        if risk >= self.block_threshold:
            verdict = VERDICT_BLOCK
        elif risk >= self.threshold:
            verdict = VERDICT_FLAG
        else:
            verdict = VERDICT_PASS
        reason = f"[{worst.name}] {worst.detail}"
        return risk, verdict, reason
