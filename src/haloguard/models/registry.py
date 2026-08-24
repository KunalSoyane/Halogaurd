"""Pinned model identity: revision and SHA256 manifest."""

from __future__ import annotations

SOURCE_MODEL = "cross-encoder/nli-deberta-v3-small"
CACHE_SUBDIR = "models/nli-deberta-v3-small"
INFERENCE_ARTIFACT = "model_quantized.onnx"

HF_REPO = "KunalSoyane/haloguard-deberta-onnx"
HF_REVISION = "a7d7ad44cc2e99ae026ae5c25dc8e7657f6cb60d"

ARTIFACT_SHA256 = {
    "model_quantized.onnx": "7807b34fa3fce9d30f54512784caa8a8d5f774480b9304a86d98ba894b45fbcf",
    "tokenizer.json": "4b4f60231058db4b5794e7b124bb7945bc8ade6719282de4d2e0372ee527b929",
    "config.json": "803e9115f99c8ec5dc798a6c3593f3a167a730c986f2be36bc311c5505b03cb8",
}

MAX_SEQUENCE_LENGTH = 512
