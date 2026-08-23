from __future__ import annotations

import pytest

from haloguard.models import registry
from haloguard.models.loader import model_dir


def _model_available() -> bool:
    directory = model_dir()
    return all((directory / name).is_file() for name in registry.ARTIFACT_SHA256)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _model_available():
        return
    skip = pytest.mark.skip(
        reason="model artifacts not present; run scripts/export_onnx.py first "
        "(or set HALOGUARD_MODEL_DIR to a prepared directory)"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
