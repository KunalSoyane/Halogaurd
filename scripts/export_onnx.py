"""Export cross-encoder/nli-deberta-v3-small to ONNX, quantize it (static,
uint8, calibrated on the golden dataset), and write the artifacts plus a
SHA256 manifest into the local HaloGuard model cache.

Dynamic INT8 quantization measurably hurts accuracy on this model (a borderline
hallucination flips to a pass); static uint8 with calibration data keeps the
golden-set verdicts identical to the torch reference. See CHANGELOG.

Requires the `export` extra: pip install haloguard[export]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

MODEL_ID = "cross-encoder/nli-deberta-v3-small"
MAX_SEQUENCE_LENGTH = 512
ARTIFACTS = ("model_quantized.onnx", "tokenizer.json", "config.json")


def default_out_dir() -> Path:
    from platformdirs import user_cache_dir

    return Path(user_cache_dir("haloguard")) / "models" / "nli-deberta-v3-small"


def default_calibration_data() -> Path:
    return Path(__file__).parent.parent / "tests" / "golden_dataset" / "labeled_pairs.jsonl"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(default_out_dir()), help="output directory")
    parser.add_argument(
        "--calibration-data",
        default=str(default_calibration_data()),
        help="JSONL with 'context'/'response' fields used as calibration data",
    )
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize_static
    from onnxruntime.quantization.preprocess import quant_pre_process
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    print(f"exporting {MODEL_ID} -> {out}")
    model = ORTModelForSequenceClassification.from_pretrained(MODEL_ID, export=True)
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(MODEL_ID).save_pretrained(out)

    fp32 = out / "model.onnx"
    if not fp32.exists():
        candidates = list(out.glob("*.onnx"))
        if len(candidates) != 1:
            raise SystemExit(f"unexpected export layout in {out}: {candidates}")
        candidates[0].rename(fp32)

    prepped = out / "model_prepped.onnx"
    quant_pre_process(str(fp32), str(prepped), skip_symbolic_shape_infer=True, auto_merge=True)

    tokenizer = Tokenizer.from_file(str(out / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=MAX_SEQUENCE_LENGTH)
    calibration_pairs = [
        (item["context"], item["response"])
        for item in map(json.loads, Path(args.calibration_data).read_text(encoding="utf-8").splitlines())
        if item
    ]

    class Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self._index = 0

        def get_next(self) -> dict | None:
            if self._index >= len(calibration_pairs):
                return None
            context, response = calibration_pairs[self._index]
            self._index += 1
            encoding = tokenizer.encode(context, response)
            return {
                "input_ids": np.array([encoding.ids], dtype=np.int64),
                "attention_mask": np.array([encoding.attention_mask], dtype=np.int64),
            }

    quantize_static(
        str(prepped),
        str(out / "model_quantized.onnx"),
        Reader(),
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QUInt8,
    )
    prepped.unlink()

    manifest = {name: sha256_of(out / name) for name in ARTIFACTS}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"done. artifacts in {out}")


if __name__ == "__main__":
    main()
