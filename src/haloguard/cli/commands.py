"""CLI entry points. Exit codes: 0 PASS / 1 FLAG / 2 BLOCK / 3 internal error."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import typer

from haloguard import __version__
from haloguard.core.exceptions import HaloGuardError
from haloguard.core.firewall import Firewall
from haloguard.core.result import (
    VERDICT_BLOCK,
    VERDICT_FLAG,
    VERDICT_PASS,
    VERDICT_UNKNOWN,
)
from haloguard.models import registry

app = typer.Typer(
    add_completion=False,
    help="HaloGuard: local-first hallucination firewall for LLM applications.",
)

_EXIT_CODES = {VERDICT_PASS: 0, VERDICT_FLAG: 1, VERDICT_BLOCK: 2, VERDICT_UNKNOWN: 3}


@app.command()
def check(
    prompt: str = typer.Option(..., "--prompt", "-p", help="The prompt sent to the LLM"),
    response: str = typer.Option(..., "--response", "-r", help="The LLM output to check"),
    context: Path | None = typer.Option(
        None,
        "--context",
        "-c",
        help="Path to source context; presence selects entailment mode",
        exists=True,
        readable=True,
    ),
    threshold: float = typer.Option(0.7, help="PASS/FLAG boundary"),
    block_threshold: float = typer.Option(0.9, help="FLAG/BLOCK boundary"),
    strict: bool = typer.Option(False, "--strict", help="Fail closed on scoring errors"),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable JSON on stdout"),
) -> None:
    """Score one LLM response for hallucination risk."""
    context_text = context.read_text(encoding="utf-8") if context is not None else None
    try:
        firewall = Firewall(
            threshold=threshold, block_threshold=block_threshold, strict_mode=strict
        )
        result = firewall.check(prompt, response, context_text)
    except HaloGuardError as exc:
        if json_output:
            typer.echo(json.dumps({"error": str(exc)}))
        else:
            typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(3) from exc
    if json_output:
        typer.echo(json.dumps(asdict(result)))
    else:
        typer.echo(f"{result.verdict} score={result.score:.4f} mode={result.mode_used}")
        typer.echo(result.reason, err=True)
    raise typer.Exit(_EXIT_CODES[result.verdict])


@app.command()
def version() -> None:
    """Print package version and pinned model identity."""
    typer.echo(f"haloguard {__version__}")
    typer.echo(f"source model: {registry.SOURCE_MODEL}")
    typer.echo(f"inference artifact: {registry.INFERENCE_ARTIFACT}")
    if registry.HF_REPO and registry.HF_REVISION:
        typer.echo(f"model revision: {registry.HF_REPO}@{registry.HF_REVISION}")
    else:
        typer.echo("model revision: local export (no pinned HF release yet)")


if __name__ == "__main__":
    app()
