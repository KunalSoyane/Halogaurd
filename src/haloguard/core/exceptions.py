"""One typed exception hierarchy shared by SDK, CLI, and integrations."""

from __future__ import annotations


class HaloGuardError(Exception):
    """Base class for everything HaloGuard raises intentionally."""


class ModelLoadError(HaloGuardError):
    """Model download failed, or a checksum didn't match the pinned manifest."""


class InputTooLargeError(HaloGuardError):
    """prompt/response/context exceeded the configured size cap."""


class InferenceTimeoutError(HaloGuardError):
    """Inference exceeded timeout_s. Raised only when strict_mode=True."""


class ConfigError(HaloGuardError):
    """Invalid threshold, mode, or timeout value at construction time."""
