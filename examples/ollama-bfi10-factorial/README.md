# Ollama BFI-10 Factorial

This marimo notebook previews and optionally runs a paired factorial BFI-10
experiment against a local Ollama OpenAI-compatible endpoint.

The validated protocol creates five UK base personas and crosses:

- age: `25`, `65`;
- affluence: `low`, `middle`, `high`;
- urbanicity: `urban`, `rural`;
- five repeated iterations per base-persona condition.

That produces 12 conditions, 300 expanded personas, and 3,000 questionnaire
calls. Previewing the notebook does not contact Ollama or create experiment
artifacts. The live workflow starts only after pressing **Run experiment**.

## Prerequisites

Install and start Ollama, then make the model available:

```bash
ollama pull gemma4:12b
ollama serve
```

From the repository root, install the example dependency group and launch the
notebook:

```bash
uv sync --group examples
uv run --group examples marimo edit examples/ollama-bfi10-factorial/notebook.py
```

The notebook uses:

- base URL `http://localhost:11434/v1`;
- API key `ollama`;
- model `gemma4:12b`;
- temperature `0`;
- seed `20260609`;
- structured outputs disabled;
- logprobs disabled.

Use a new valid three-part experiment ID for another run. Generated artifacts
are written under `experiments/{experiment_id}/` and must not be committed.
