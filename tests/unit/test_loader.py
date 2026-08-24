from __future__ import annotations

from pathlib import Path

import pytest

from haloguard.core.exceptions import ModelLoadError
from haloguard.models import ensure_model
from haloguard.models.loader import model_dir
from haloguard.models.registry import ARTIFACT_SHA256


def test_model_dir_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HALOGUARD_MODEL_DIR", str(tmp_path))
    assert model_dir() == tmp_path


def test_missing_artifacts_raise(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HALOGUARD_MODEL_DIR", str(tmp_path))

    # Force the registry to act as if no HF repo is pinned, triggering local export error
    monkeypatch.setattr("haloguard.models.registry.HF_REPO", None)

    with pytest.raises(ModelLoadError, match="export_onnx"):
        ensure_model()


def test_checksum_mismatch_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HALOGUARD_MODEL_DIR", str(tmp_path))
    for name in ARTIFACT_SHA256:
        (tmp_path / name).write_bytes(b"tampered")
    with pytest.raises(ModelLoadError, match="checksum mismatch"):
        ensure_model()
