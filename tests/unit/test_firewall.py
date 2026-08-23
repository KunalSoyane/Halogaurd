from __future__ import annotations

import pytest

from haloguard import ConfigError, Firewall, InputTooLargeError


def test_input_too_large() -> None:
    fw = Firewall(max_input_chars=10)
    with pytest.raises(InputTooLargeError):
        fw.check("p", "r" * 11, context="c")


def test_non_string_input() -> None:
    fw = Firewall()
    with pytest.raises(ConfigError):
        fw.check("p", 123, context="c")  # type: ignore[arg-type]


def test_entailment_mode_requires_context() -> None:
    fw = Firewall(mode="entailment")
    with pytest.raises(ConfigError):
        fw.check("p", "r")
