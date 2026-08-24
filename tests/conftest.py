from __future__ import annotations

import pytest


def _model_available() -> bool:
    try:
        from haloguard.models.loader import ensure_model

        ensure_model()
        return True
    except Exception:
        return False


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
