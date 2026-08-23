"""Optional thin helpers for OpenAI / Anthropic / Ollama style clients.

Each helper adapts a client SDK into a plain generate(prompt) -> str callable,
which guarded_call() then scores. No client SDK is imported here.
"""

from __future__ import annotations

from typing import Any, Callable

from haloguard.core.exceptions import HaloGuardError
from haloguard.core.firewall import Firewall
from haloguard.core.result import VERDICT_BLOCK, FirewallResult


def guarded_call(
    generate: Callable[[str], str],
    prompt: str,
    *,
    context: str | None = None,
    firewall: Firewall | None = None,
    raise_on_block: bool = True,
) -> tuple[str, FirewallResult]:
    """Call generate(prompt), score the output, and return (response, result).

    Raises HaloGuardError on a BLOCK verdict unless raise_on_block=False.
    """
    fw = firewall or Firewall()
    response = generate(prompt)
    result = fw.check(prompt, response, context=context)
    if result.verdict == VERDICT_BLOCK and raise_on_block:
        raise HaloGuardError(f"HaloGuard blocked LLM output: {result.reason}")
    return response, result


def openai_generate(client: Any, model: str = "gpt-4o-mini", **kwargs: Any) -> Callable[[str], str]:
    """Adapt an OpenAI client into generate(prompt) -> str."""

    def call(prompt: str) -> str:
        completion = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], **kwargs
        )
        return completion.choices[0].message.content or ""

    return call


def anthropic_generate(
    client: Any, model: str = "claude-3-5-haiku-latest", **kwargs: Any
) -> Callable[[str], str]:
    """Adapt an Anthropic client into generate(prompt) -> str."""

    def call(prompt: str) -> str:
        message = client.messages.create(
            model=model,
            max_tokens=kwargs.pop("max_tokens", 1024),
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return "".join(block.text for block in message.content if hasattr(block, "text"))

    return call


def ollama_generate(client: Any, model: str = "llama3.2", **kwargs: Any) -> Callable[[str], str]:
    """Adapt an Ollama client into generate(prompt) -> str."""

    def call(prompt: str) -> str:
        response = client.chat(
            model=model, messages=[{"role": "user", "content": prompt}], **kwargs
        )
        return response["message"]["content"]

    return call


__all__ = ["anthropic_generate", "guarded_call", "ollama_generate", "openai_generate"]
