from __future__ import annotations

import json
from pathlib import Path

import pytest

from haloguard import Firewall

GOLDEN = Path(__file__).parent.parent / "golden_dataset" / "labeled_pairs.jsonl"


def _load_pairs() -> list[dict]:
    return [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line]


@pytest.mark.integration
def test_verdicts_match_hand_labels() -> None:
    fw = Firewall(mode="entailment")
    pairs = _load_pairs()
    assert len(pairs) == 17

    faithful_risks: list[float] = []
    hallucination_risks: list[float] = []
    for i, pair in enumerate(pairs):
        result = fw.check("answer based on the context", pair["response"], context=pair["context"])
        assert result.mode_used == "entailment"
        assert result.verdict != "UNKNOWN", f"pair {i} failed scoring: {result.reason}"
        if pair["label"] == 0:
            assert result.verdict == "PASS", f"pair {i} should PASS: {result}"
            faithful_risks.append(result.score)
        else:
            assert result.verdict in ("FLAG", "BLOCK"), f"pair {i} should be detected: {result}"
            hallucination_risks.append(result.score)

    assert max(faithful_risks) < min(hallucination_risks), (
        f"risk distributions overlap: faithful max {max(faithful_risks):.4f} >= "
        f"hallucination min {min(hallucination_risks):.4f}"
    )
