"""Locate, download, and checksum-verify model artifacts."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from haloguard.core.exceptions import ModelLoadError
from haloguard.models import registry


@dataclass(frozen=True)
class ModelPaths:
    model: Path
    tokenizer: Path
    config: Path


def model_dir() -> Path:
    override = os.environ.get("HALOGUARD_MODEL_DIR")
    if override:
        return Path(override)
    from platformdirs import user_cache_dir

    return Path(user_cache_dir("haloguard")) / registry.CACHE_SUBDIR


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(directory: Path) -> None:
    if registry.HF_REPO is None or registry.HF_REVISION is None:
        raise ModelLoadError(
            "model artifacts not found and no pinned HF release is configured yet; "
            f"run 'python scripts/export_onnx.py' to build them into {directory}"
        )
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ModelLoadError("huggingface_hub is required to download model artifacts") from exc
    for name in registry.ARTIFACT_SHA256:
        try:
            hf_hub_download(
                repo_id=registry.HF_REPO,
                filename=name,
                revision=registry.HF_REVISION,
                local_dir=directory,
            )
        except Exception as exc:
            raise ModelLoadError(
                f"failed to download '{name}' from {registry.HF_REPO}: {exc}"
            ) from exc


def ensure_model() -> ModelPaths:
    """Return verified artifact paths, downloading on first use if needed.

    SHA256 is verified against the pinned manifest on every load.
    """
    directory = model_dir()
    missing = [name for name in registry.ARTIFACT_SHA256 if not (directory / name).is_file()]
    if missing:
        _download(directory)
    for name, expected in registry.ARTIFACT_SHA256.items():
        path = directory / name
        if not path.is_file():
            raise ModelLoadError(f"model artifact missing after download: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise ModelLoadError(
                f"checksum mismatch for {path}: expected {expected}, got {actual}"
            )
    return ModelPaths(
        model=directory / registry.INFERENCE_ARTIFACT,
        tokenizer=directory / "tokenizer.json",
        config=directory / "config.json",
    )
