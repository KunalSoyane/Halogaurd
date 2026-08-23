from __future__ import annotations

import pytest

from haloguard import ConfigError, Firewall


def test_defaults() -> None:
    fw = Firewall()
    assert fw.config.threshold == 0.7
    assert fw.config.block_threshold == 0.9
    assert fw.config.mode == "auto"
    assert fw.config.timeout_s == 5.0
    assert fw.config.strict_mode is False


def test_invalid_threshold() -> None:
    with pytest.raises(ConfigError):
        Firewall(threshold=1.5)


def test_block_threshold_below_threshold() -> None:
    with pytest.raises(ConfigError):
        Firewall(threshold=0.9, block_threshold=0.7)


def test_invalid_mode() -> None:
    with pytest.raises(ConfigError):
        Firewall(mode="vibes")


def test_invalid_timeout() -> None:
    with pytest.raises(ConfigError):
        Firewall(timeout_s=0)
