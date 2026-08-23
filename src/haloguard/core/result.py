"""The single result contract returned by every entry point."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

VERDICT_PASS = "PASS"
VERDICT_FLAG = "FLAG"
VERDICT_BLOCK = "BLOCK"
VERDICT_UNKNOWN = "UNKNOWN"

MODE_ENTAILMENT = "entailment"
MODE_CONSISTENCY = "consistency"


@dataclass(frozen=True)
class FirewallResult:
    """Outcome of a single hallucination check.

    Attributes:
        score: 0.0-1.0 hallucination risk; higher = more likely hallucinated.
        verdict: "PASS" | "FLAG" | "BLOCK" | "UNKNOWN". UNKNOWN is the fail-open
            verdict when scoring itself fails and strict_mode=False.
        reason: Human-readable explanation of the verdict.
        mode_used: "entailment" | "consistency".
        latency_ms: Wall-clock time of the check in milliseconds.
    """

    score: float
    verdict: str
    reason: str
    mode_used: str
    latency_ms: float


@dataclass(frozen=True)
class FirewallInput:
    """One item for check_batch(). metadata is passed through untouched and
    never inspected for scoring."""

    prompt: str
    response: str
    context: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
