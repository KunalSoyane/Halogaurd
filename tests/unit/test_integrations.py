from __future__ import annotations

from uuid import uuid4

import pytest

from haloguard import Firewall, HaloGuardError
from haloguard.integrations.llamaindex_handler import HaloGuardQueryHook, guard_query_engine
from haloguard.integrations.raw_wrappers import guarded_call

CONTEXT = "The Eiffel Tower is located in Paris and was completed in 1889."
FAITHFUL = "The Eiffel Tower is in Paris."
HALLUCINATED = "The Eiffel Tower was completed in 1923 and is located in Lyon."


@pytest.mark.integration
def test_langchain_handler_records_and_blocks() -> None:
    lc = pytest.importorskip("langchain_core")
    from haloguard.integrations.langchain_handler import HaloGuardCallbackHandler

    generations = lc.outputs.Generation
    handler = HaloGuardCallbackHandler(firewall=Firewall(), context_provider=lambda _text: CONTEXT)

    ok_run = uuid4()
    handler.on_llm_end(
        lc.outputs.LLMResult(generations=[[generations(text=FAITHFUL)]]), run_id=ok_run
    )
    assert handler.results[ok_run].verdict == "PASS"

    bad_run = uuid4()
    with pytest.raises(HaloGuardError):
        handler.on_llm_end(
            lc.outputs.LLMResult(generations=[[generations(text=HALLUCINATED)]]),
            run_id=bad_run,
        )


class _FakeNode:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeSourceNode:
    def __init__(self, text: str) -> None:
        self.node = _FakeNode(text)


class _FakeQueryResponse:
    def __init__(self, response: str, source_nodes: list) -> None:
        self.response = response
        self.source_nodes = source_nodes


class _FakeQueryEngine:
    def __init__(self, response: _FakeQueryResponse) -> None:
        self._response = response

    def query(self, query: str) -> _FakeQueryResponse:
        return self._response


@pytest.mark.integration
def test_llamaindex_hook_scores_against_source_nodes() -> None:
    hook = HaloGuardQueryHook(Firewall())
    good = _FakeQueryResponse(FAITHFUL, [_FakeSourceNode(CONTEXT)])
    bad = _FakeQueryResponse(HALLUCINATED, [_FakeSourceNode(CONTEXT)])
    assert hook.check_response(good).verdict == "PASS"
    assert hook.check_response(bad).verdict in ("FLAG", "BLOCK")


@pytest.mark.integration
def test_guard_query_engine() -> None:
    response = _FakeQueryResponse(FAITHFUL, [_FakeSourceNode(CONTEXT)])
    guarded = guard_query_engine(_FakeQueryEngine(response), Firewall())
    raw, result = guarded.query("Where is the Eiffel Tower?")
    assert raw is response
    assert result.verdict == "PASS"
    assert result.mode_used == "entailment"


@pytest.mark.integration
def test_guarded_call_blocks() -> None:
    with pytest.raises(HaloGuardError):
        guarded_call(lambda _p: HALLUCINATED, "where?", context=CONTEXT)
    response, result = guarded_call(lambda _p: FAITHFUL, "where?", context=CONTEXT)
    assert response == FAITHFUL
    assert result.verdict == "PASS"
