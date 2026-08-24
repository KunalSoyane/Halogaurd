[![PyPI version](https://img.shields.io/badge/pypi-v0.1.5-orange.svg)](https://pypi.org/project/haloguard/)

[![Python](https://img.shields.io/pypi/pyversions/haloguard.svg)](https://pypi.org/project/haloguard/)

# HaloGuard



A local-first hallucination firewall for LLM applications. HaloGuard sits between an

LLM and the application consuming its output, scoring every response for hallucination

risk before it reaches a user. Everything runs on the caller's machine -- no prompt,

response, or context ever leaves the device.



## Scoring modes



- **Entailment mode** (RAG-style): scores whether the response is supported by supplied

  source context, using an NLI cross-encoder (`DeBERTa-v3-small`, ONNX, quantized).

- **Consistency mode** (no context): scores whether the response is internally

  consistent, cross-checking its own claims with the same NLI model.



Mode is selected automatically by input shape (`auto`), or set explicitly.



## Install



```bash

pip install haloguard

```



The model downloads automatically on first use (~160 MB, cached locally).

No manual setup required.



If you want to rebuild the ONNX artifact from source instead:



pip install "haloguard[export]"

python -c "import runpy; runpy.run_module('haloguard.scripts.export_onnx')"



## SDK quickstart



```python

from haloguard import Firewall



fw = Firewall()  # threshold=0.7, block_threshold=0.9, mode="auto"



# Entailment mode: context supplied

result = fw.check(

    prompt="Where is the Eiffel Tower?",

    response="The Eiffel Tower is in Paris.",

    context="The Eiffel Tower is located in Paris and was completed in 1889.",

)

print(result.verdict, result.score, result.reason)



# Consistency mode: no context

result = fw.check(

    prompt="When is the meeting?",

    response="The meeting is on Tuesday. The meeting is on Friday.",

)

```



Every check returns a `FirewallResult`. Also available: `acheck()` (async),

`check_batch()` (many items over one loaded session).



## CLI



```bash

haloguard check --prompt "..." --response "..."                 # consistency mode

haloguard check --prompt "..." --response "..." --context FILE  # entailment mode

haloguard check ... --json                                      # machine-readable

haloguard version

```



Exit codes: `0` PASS / `1` FLAG / `4` BLOCK / `3` internal error (incl. UNKNOWN).



## Framework hooks



```python

# LangChain:  pip install "haloguard[langchain]"

from haloguard.integrations.langchain_handler import HaloGuardCallbackHandler



handler = HaloGuardCallbackHandler(context_provider=lambda _text: retrieved_context)



# LlamaIndex-style query responses (duck-typed, no hard dependency)

from haloguard.integrations.llamaindex_handler import HaloGuardQueryHook



result = HaloGuardQueryHook().check_response(query_response)



# Any client SDK: adapt to generate(prompt) -> str, then score

from haloguard.integrations.raw_wrappers import guarded_call, openai_generate



response, result = guarded_call(openai_generate(client), prompt, context=context)

```



## Verdicts



- `score` -- 0.0-1.0 hallucination risk (higher = more likely hallucinated)

- `verdict` -- `PASS` (risk < threshold), `FLAG` (threshold <= risk < block_threshold),

  `BLOCK` (risk >= block_threshold), `UNKNOWN` (scoring failed, fail-open)

- `reason`, `mode_used`, `latency_ms`



`UNKNOWN` is the fail-open verdict returned when scoring itself fails and

`strict_mode=False` (the default). Set `strict_mode=True` to fail closed instead.



## Honest limitations



HaloGuard is **defense-in-depth, not a guarantee**. An adversarially crafted response

can read as entailed/consistent to any NLI model while still being false. The measured

false-negative rate on the golden benchmark is the real accuracy statement; treat

HaloGuard as one layer in a safety stack, not the only one.



## Development



```bash

pip install -e ".[dev]"

pytest tests -v        # unit tests always; integration tests need model artifacts

ruff check src tests scripts

mypy

```



Integration tests run real inference against `tests/golden_dataset/labeled_pairs.jsonl`

and are skipped automatically when model artifacts are absent.



## License



MIT 

