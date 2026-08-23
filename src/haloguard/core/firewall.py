"""Firewall: the single entry point for hallucination checks."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import cast

from pydantic import ValidationError

from haloguard.core.config import Config, Mode
from haloguard.core.exceptions import (
    ConfigError,
    HaloGuardError,
    InferenceTimeoutError,
    InputTooLargeError,
)
from haloguard.core.result import (
    MODE_CONSISTENCY,
    MODE_ENTAILMENT,
    VERDICT_PASS,
    VERDICT_UNKNOWN,
    FirewallInput,
    FirewallResult,
)
from haloguard.scorers.aggregator import Aggregator
from haloguard.scorers.base import BaseScorer, SubScore
from haloguard.scorers.consistency import ConsistencyScorer
from haloguard.scorers.entailment import EntailmentScorer

_INFERENCE_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="haloguard")


class Firewall:
    """Scores LLM responses for hallucination risk, fully locally.

    Fail-open by default: if scoring fails or times out, check() returns a
    result with verdict="UNKNOWN" and the error in reason. Set
    strict_mode=True to fail closed (raise instead).
    """

    def __init__(
        self,
        threshold: float = 0.7,
        block_threshold: float = 0.9,
        mode: str = "auto",
        timeout_s: float = 5.0,
        strict_mode: bool = False,
        max_input_chars: int = 100_000,
    ) -> None:
        try:
            self.config = Config(
                threshold=threshold,
                block_threshold=block_threshold,
                mode=cast(Mode, mode),
                timeout_s=timeout_s,
                strict_mode=strict_mode,
                max_input_chars=max_input_chars,
            )
        except ValidationError as exc:
            raise ConfigError(str(exc)) from exc
        self._aggregator = Aggregator(self.config.threshold, self.config.block_threshold)
        self._scorers = {
            MODE_ENTAILMENT: EntailmentScorer(),
            MODE_CONSISTENCY: ConsistencyScorer(),
        }

    def check(self, prompt: str, response: str, context: str | None = None) -> FirewallResult:
        """Score one LLM response for hallucination risk."""
        start = time.perf_counter()
        self._validate_inputs(prompt, response, context)
        mode = self._resolve_mode(context)
        if not response.strip():
            return FirewallResult(
                score=0.0,
                verdict=VERDICT_PASS,
                reason="empty response contains no claims to check",
                mode_used=mode,
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )
        scorer = self._scorers.get(mode)
        if scorer is None:
            raise HaloGuardError(f"mode '{mode}' is not implemented yet")
        try:
            subscore = self._run_with_timeout(scorer, prompt, response, context)
        except HaloGuardError as exc:
            if self.config.strict_mode:
                raise
            return self._unknown(mode, start, str(exc))
        except Exception as exc:
            if self.config.strict_mode:
                raise HaloGuardError(str(exc)) from exc
            return self._unknown(mode, start, str(exc))
        risk, verdict, reason = self._aggregator.combine([subscore])
        return FirewallResult(
            score=risk,
            verdict=verdict,
            reason=reason,
            mode_used=mode,
            latency_ms=(time.perf_counter() - start) * 1000.0,
        )

    async def acheck(
        self, prompt: str, response: str, context: str | None = None
    ) -> FirewallResult:
        """Async check(): offloads the blocking inference via asyncio.to_thread()."""
        return await asyncio.to_thread(self.check, prompt, response, context)

    def check_batch(self, items: Sequence[FirewallInput]) -> list[FirewallResult]:
        """Score many items, reusing the already-loaded model session."""
        return [self.check(item.prompt, item.response, item.context) for item in items]

    def _validate_inputs(self, prompt: str, response: str, context: str | None) -> None:
        for name, value in (("prompt", prompt), ("response", response), ("context", context)):
            if value is None:
                continue
            if not isinstance(value, str):
                raise ConfigError(f"{name} must be a string")
            if len(value) > self.config.max_input_chars:
                raise InputTooLargeError(
                    f"{name} is {len(value)} chars, exceeding the "
                    f"{self.config.max_input_chars} char cap"
                )

    def _resolve_mode(self, context: str | None) -> str:
        mode: str = self.config.mode
        has_context = bool(context and context.strip())
        if mode == "auto":
            mode = MODE_ENTAILMENT if has_context else MODE_CONSISTENCY
        if mode == MODE_ENTAILMENT and not has_context:
            raise ConfigError("entailment mode requires non-empty context")
        return mode

    def _run_with_timeout(
        self, scorer: BaseScorer, prompt: str, response: str, context: str | None
    ) -> SubScore:
        scorer.ensure_loaded()
        future = _INFERENCE_EXECUTOR.submit(scorer.score, prompt, response, context)
        try:
            return future.result(timeout=self.config.timeout_s)
        except FutureTimeoutError as exc:
            future.cancel()
            raise InferenceTimeoutError(
                f"inference exceeded timeout_s={self.config.timeout_s}"
            ) from exc

    def _unknown(self, mode: str, start: float, detail: str) -> FirewallResult:
        return FirewallResult(
            score=0.0,
            verdict=VERDICT_UNKNOWN,
            reason=f"scoring failed (fail-open): {detail}",
            mode_used=mode,
            latency_ms=(time.perf_counter() - start) * 1000.0,
        )
