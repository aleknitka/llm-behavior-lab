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
- maximum attempts `3` with exponential backoff capped at `30` seconds;
- maximum concurrency `1`;
- structured outputs disabled;
- logprobs disabled.

Concurrency is across personas only; BFI-10 items remain sequential within each
persona. Keep local-provider concurrency conservative and increase it only after
measuring model-server throughput and memory use.

Use a new valid three-part experiment ID for another run. Generated artifacts
are written under `experiments/{experiment_id}/` and must not be committed.
If execution is interrupted, rerun `scale-run` with the existing run directory
name:

```bash
uv run llm-behavior-lab scale-run \
  --experiment-id your-study-id \
  --run-id run-bfi10-gemma4-12b-20260611090000 \
  --api-key ollama
```

Add `--retry-failed` to retry latest failed or invalid item attempts.

The staged factorial workflow writes complete snapshots as JSON:

```text
experiments/{experiment_id}/
  design.json
  protocol.json
  base_personas.json
  personas.json
  protocol_assignments.json
  metadata.json
  run-bfi10-{model}-{timestamp}/
    run.json
    scale.json
    responses/{subject_id}.jsonl
    scoring/default-1.0/
      scores.jsonl
      reliability.jsonl
    results/default-1.0/
      responses.csv
      scores.csv
      reliability.csv
      summary.json
```

The persona and assignment files are validated collection documents rather
than JSONL streams. Item responses and derived scoring units remain append-only
JSONL. Response rows reference personas by `subject_id`; `responses.csv` joins
the persisted persona snapshot and includes `persona_*` columns.
