# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial scaffold: `src/haloguard/` layout with `core/` and `scorers/` subpackages.
- Exception hierarchy (`HaloGuardError`, `ModelLoadError`, `InputTooLargeError`,
  `InferenceTimeoutError`, `ConfigError`).
- `FirewallResult` frozen dataclass contract.
- Golden dataset of 15 hand-labeled (context, response) pairs used to validate
  entailment scoring (`tests/golden_dataset/labeled_pairs.jsonl`).
- `Firewall.check()` / `acheck()` / `check_batch()` backed by ONNX Runtime (CPU),
  with fail-open/fail-closed behaviour, input size caps, and an inference timeout.
- Consistency mode: context-free scoring by cross-checking the response's own
  claims pairwise with the NLI model (batched inference).
- `FirewallInput` / `FirewallResult` contract shared by SDK, CLI, and hooks.
- Typer CLI (`haloguard check`, `haloguard version`) with exit codes
  0 PASS / 1 FLAG / 2 BLOCK / 3 internal error and `--json` output.
- Integrations: LangChain callback handler, LlamaIndex-style query hook, and
  raw client adapters (OpenAI / Anthropic / Ollama) via `guarded_call`.
- `models/` subpackage: pinned SHA256 manifest verified on every load, plus a
  local cache dir (`platformdirs`) with `HALOGUARD_MODEL_DIR` override.
- `scripts/export_onnx.py` to rebuild the ONNX artifact from the source model.
- GitHub Actions CI (lint, type-check, 3 OS x Python 3.9-3.12 test matrix,
  fresh-venv install smoke test, pip-audit) and PyPI trusted-publishing
  release workflow gated on version tags.

### Changed
- Quantization strategy: naive dynamic INT8 quantization measurably hurt accuracy
  (a borderline hallucination's risk dropped 0.84 -> 0.33, flipping it to PASS).
  Switched to static uint8 quantization with graph preprocessing and calibration
  data, which keeps golden-set verdicts identical to the torch reference.
- Runtime inference uses `onnxruntime` + `tokenizers` only; `sentence-transformers`
  / torch moved to the optional `export` extra.

### Notes
- The quantized artifact is ~161MB (fp32 export ~568MB), larger than the design
  doc's ~35-90MB estimate: DeBERTa-v3's 128k-vocabulary embedding dominates the
  file size. Install-size expectations in the doc should be revised accordingly.
