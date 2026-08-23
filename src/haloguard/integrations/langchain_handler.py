"""LangChain callback hook: auto-score every LLM call inside a chain.

Requires the `langchain` extra: pip install haloguard[langchain]
"""

from __future__ import annotations

from typing import Any, Callable
from uuid import UUID

from haloguard.core.exceptions import HaloGuardError
from haloguard.core.firewall import Firewall
from haloguard.core.result import VERDICT_BLOCK, FirewallResult

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.outputs import LLMResult
except ImportError as exc:
    raise ImportError(
        "LangChain is required for this integration: pip install haloguard[langchain]"
    ) from exc


class HaloGuardCallbackHandler(BaseCallbackHandler):
    """Scores each LLM generation when a chain finishes an LLM call.

    The verdict is stored per run; a BLOCK verdict raises by default so a chain
    can short-circuit before the output reaches the caller. Supply
    context_provider to score in entailment mode against retrieved context.
    """

    def __init__(
        self,
        firewall: Firewall | None = None,
        context_provider: Callable[[str], str] | None = None,
        raise_on_block: bool = True,
    ) -> None:
        super().__init__()
        self.firewall = firewall or Firewall()
        self.context_provider = context_provider
        self.raise_on_block = raise_on_block
        self.results: dict[UUID, FirewallResult] = {}

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        text = _extract_text(response)
        if not text:
            return
        context = self.context_provider(text) if self.context_provider else None
        result = self.firewall.check(prompt="", response=text, context=context)
        self.results[run_id] = result
        if result.verdict == VERDICT_BLOCK and self.raise_on_block:
            raise HaloGuardError(f"HaloGuard blocked LLM output: {result.reason}")


def _extract_text(response: LLMResult) -> str:
    try:
        generations = response.generations[0]
        if not generations:
            return ""
        return generations[0].text
    except (IndexError, AttributeError):
        return ""


__all__ = ["HaloGuardCallbackHandler"]
