"""Mode 2: context-free scoring via internal contradiction checks."""

from __future__ import annotations

import re

from haloguard.scorers.base import BaseScorer, SubScore
from haloguard.scorers.entailment import _load_session, predict_nli

MAX_SENTENCES = 6

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()]


class ConsistencyScorer(BaseScorer):
    """Scores whether a response is internally consistent.

    The response is split into claims; every claim pair is checked both ways
    with the NLI model, and the strongest contradiction signal becomes the
    hallucination risk.
    """

    name = "consistency"

    def ensure_loaded(self) -> None:
        _load_session()

    def score(self, prompt: str, response: str, context: str | None = None) -> SubScore:
        sentences = split_sentences(response)[:MAX_SENTENCES]
        if len(sentences) < 2:
            return SubScore(
                name=self.name,
                risk=0.0,
                detail="fewer than two claims; nothing to cross-check",
            )
        index_pairs = [
            (i, j) for i in range(len(sentences)) for j in range(len(sentences)) if i != j
        ]
        probs = predict_nli([(sentences[i], sentences[j]) for i, j in index_pairs])
        worst = max(range(len(index_pairs)), key=lambda k: probs[k]["contradiction"])
        risk = probs[worst]["contradiction"]
        i, j = index_pairs[worst]
        if risk >= 0.5:
            detail = (
                f"claims contradict each other with probability {risk:.2f}: "
                f"'{sentences[i]}' vs '{sentences[j]}'"
            )
        else:
            detail = f"no internal contradictions found across {len(sentences)} claims"
        return SubScore(name=self.name, risk=risk, detail=detail)
