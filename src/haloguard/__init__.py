"""HaloGuard: a local-first hallucination firewall for LLM applications."""

from __future__ import annotations

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

__version__ = "0.1.0"

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
