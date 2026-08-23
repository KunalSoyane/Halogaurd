"""LlamaIndex hook: score query-engine responses against retrieved source nodes.

Entailment mode is the default, since RAG context is already on hand. Works with
any object exposing .response/.get_response() and .source_nodes (e.g.
llama_index.core.base.base_query_engine query results), so no hard dependency on
a specific llama-index version is required at import time.

Requires the `llamaindex` extra only if you use guard_query_engine().
"""

from __future__ import annotations

from typing import Any

from haloguard.core.firewall import Firewall
from haloguard.core.result import FirewallResult


def context_from_source_nodes(source_nodes: Any) -> str:
    """Join retrieved node texts into a single context string."""
    parts: list[str] = []
    for node in source_nodes or []:
        candidate = getattr(node, "node", node)
        text = getattr(candidate, "text", None) or getattr(candidate, "get_text", lambda: "")()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


class HaloGuardQueryHook:
    """Scores a query engine's response against its retrieved source nodes."""

    def __init__(self, firewall: Firewall | None = None) -> None:
        self.firewall = firewall or Firewall(mode="auto")

    def check_response(self, response: Any, prompt: str = "") -> FirewallResult:
        text = getattr(response, "response", None) or getattr(response, "get_response", None)
        if callable(text):
            text = text()
        text = str(text or "")
        context = context_from_source_nodes(getattr(response, "source_nodes", None))
        return self.firewall.check(prompt, text, context=context or None)


def guard_query_engine(query_engine: Any, firewall: Firewall | None = None) -> Any:
    """Wrap a LlamaIndex query engine so .query() returns (result, FirewallResult)."""
    hook = HaloGuardQueryHook(firewall)

    class _Guarded:
        def __init__(self, engine: Any) -> None:
            self._engine = engine

        def query(self, query: str) -> tuple[Any, FirewallResult]:
            result = self._engine.query(query)
            return result, hook.check_response(result, prompt=query)

    return _Guarded(query_engine)


__all__ = ["HaloGuardQueryHook", "context_from_source_nodes", "guard_query_engine"]
