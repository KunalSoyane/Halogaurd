"""HaloGuard: a local-first hallucination firewall for LLM applications."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from haloguard.core.config import Config
from haloguard.core.exceptions import (
    ConfigError,
    HaloGuardError,
    InferenceTimeoutError,
    InputTooLargeError,
    ModelLoadError,
)
from haloguard.core.firewall import Firewall
from haloguard.core.result import FirewallInput, FirewallResult

try:
    __version__ = version("haloguard")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "Config",
    "ConfigError",
    "Firewall",
    "FirewallInput",
    "FirewallResult",
    "HaloGuardError",
    "InferenceTimeoutError",
    "InputTooLargeError",
    "ModelLoadError",
    "__version__",
]
