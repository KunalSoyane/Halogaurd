"""Scorer interface shared by entailment and consistency modes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class SubScore:
    """One scorer's output, fed to the aggregator.

    Attributes:
        name: Scorer identifier, e.g. "entailment".
        risk: 0.0-1.0 hallucination risk; higher = more likely hallucinated.
        detail: Human-readable detail used to build the verdict reason.
    """

    name: str
    risk: float
    detail: str


class BaseScorer(ABC):
    """A scorer turns text into a hallucination-risk sub-score."""

    name: str = "base"

    def ensure_loaded(self) -> None:  # noqa: B027
        """Load underlying model weights. Called before the timed inference window."""

    @abstractmethod
    def score(self, prompt: str, response: str, context: str | None = None) -> SubScore:
        """Score a single (prompt, response[, context]) item."""
