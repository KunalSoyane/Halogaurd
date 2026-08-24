"""Mode 1: context-aware NLI entailment scoring via ONNX Runtime."""

from __future__ import annotations

import json
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from haloguard.core.exceptions import ConfigError, ModelLoadError
from haloguard.models import ensure_model
from haloguard.models.registry import MAX_SEQUENCE_LENGTH
from haloguard.scorers.base import BaseScorer, SubScore

_PAD_TOKEN_ID = 0


@lru_cache(maxsize=1)
def _load_session() -> tuple[Any, Any, dict[str, int]]:
    try:
        import onnxruntime as ort
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise ModelLoadError(f"missing inference dependency: {exc}") from exc
    paths = ensure_model()
    session = ort.InferenceSession(str(paths.model), providers=["CPUExecutionProvider"])
    tokenizer = Tokenizer.from_file(str(paths.tokenizer))
    tokenizer.enable_truncation(max_length=MAX_SEQUENCE_LENGTH)
    label2id = json.loads(paths.config.read_text(encoding="utf-8"))["label2id"]
    return session, tokenizer, label2id


def _softmax(values: Sequence[float]) -> list[float]:
    import math

    largest = max(values)
    exps = [math.exp(v - largest) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def predict_nli(pairs: Sequence[tuple[str, str]]) -> list[dict[str, float]]:
    """Run the NLI model over (premise, hypothesis) pairs in one batched call."""
    if not pairs:
        return []
    session, tokenizer, label2id = _load_session()
    encodings = [tokenizer.encode(premise, hypothesis) for premise, hypothesis in pairs]
    width = max(len(e.ids) for e in encodings)
    input_ids = [e.ids + [_PAD_TOKEN_ID] * (width - len(e.ids)) for e in encodings]
    attention_mask = [e.attention_mask + [0] * (width - len(e.attention_mask)) for e in encodings]
    logits_batch = session.run(
        ["logits"], {"input_ids": input_ids, "attention_mask": attention_mask}
    )[0]
    id2label = {v: k for k, v in label2id.items()}
    return [
        {id2label[i]: p for i, p in enumerate(_softmax([float(v) for v in logits]))}
        for logits in logits_batch
    ]


class EntailmentScorer(BaseScorer):
    """Scores whether a response is entailed by the supplied source context.

    Hallucination risk is 1 - P(entailment): both contradiction and
    "not mentioned" (neutral) count against the response.
    """

    name = "entailment"

    def ensure_loaded(self) -> None:
        _load_session()

    def score(self, prompt: str, response: str, context: str | None = None) -> SubScore:
        if context is None or not context.strip():
            raise ConfigError("entailment mode requires non-empty context")
        probs = predict_nli([(context, response)])[0]
        entailment = probs["entailment"]
        contradiction = probs["contradiction"]
        neutral = probs["neutral"]
        risk = min(1.0, max(0.0, 1.0 - entailment))
        detail = (
            f"response entailed by context with probability {entailment:.2f} "
            f"(contradiction {contradiction:.2f}, unsupported {neutral:.2f})"
        )
        return SubScore(name=self.name, risk=risk, detail=detail)
