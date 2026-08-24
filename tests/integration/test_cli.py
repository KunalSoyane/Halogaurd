from __future__ import annotations

import json
import subprocess
import sys

import pytest


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "haloguard", *args],
        capture_output=True,
        text=True,
        timeout=300,
    )


@pytest.mark.integration
def test_check_pass_exit_0(tmp_path) -> None:
    context_file = tmp_path / "context.txt"
    context_file.write_text(
        "The Eiffel Tower is located in Paris and was completed in 1889.", encoding="utf-8"
    )
    proc = run_cli(
        "check",
        "--prompt",
        "Where is the Eiffel Tower?",
        "--response",
        "The Eiffel Tower is in Paris.",
        "--context",
        str(context_file),
        "--json",
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["verdict"] == "PASS"
    assert payload["mode_used"] == "entailment"
    assert set(payload) == {"score", "verdict", "reason", "mode_used", "latency_ms"}


@pytest.mark.integration
def test_check_block_exit_4(tmp_path) -> None:
    context_file = tmp_path / "context.txt"
    context_file.write_text(
        "The Eiffel Tower is located in Paris and was completed in 1889.", encoding="utf-8"
    )
    proc = run_cli(
        "check",
        "--prompt",
        "Where is the Eiffel Tower?",
        "--response",
        "The Eiffel Tower was completed in 1923 and is located in Lyon.",
        "--context",
        str(context_file),
        "--json",
    )
    assert proc.returncode == 4, proc.stderr
    assert json.loads(proc.stdout)["verdict"] == "BLOCK"


@pytest.mark.integration
def test_check_consistency_mode_no_context() -> None:
    proc = run_cli(
        "check",
        "--prompt",
        "When is the meeting?",
        "--response",
        "The meeting is on Tuesday. The meeting is on Friday.",
        "--json",
    )
    assert proc.returncode in (1, 4), proc.stderr
    assert json.loads(proc.stdout)["mode_used"] == "consistency"


def test_version_exit_0() -> None:
    proc = run_cli("version")
    assert proc.returncode == 0, proc.stderr
    assert "haloguard" in proc.stdout
