# Ollama BFI-10 Factorial

This marimo notebook previews and optionally runs a canonical BFI-10 experiment
against a local Ollama OpenAI-compatible endpoint.

The immutable unified protocol creates one UK base persona and crosses:

- age: `25`, `65`;
- affluence: `low`, `middle`, `high`;
- urbanicity: `urban`, `rural`;
- one iteration per base-persona condition.

That produces 12 conditions, 12 expanded personas, and 120 questionnaire calls.
Previewing the notebook validates and expands the protocol without contacting
Ollama or creating experiment artifacts. Live execution starts only after
pressing **Run experiment**.

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

The canonical `protocol.json` contains:

- experiment ID `ollama-bfi-ten`;
- persona and run seed `20260609`;
- base URL `http://localhost:11434/v1`;
- model `gemma4:12b`;
- temperature `0`;
- maximum attempts `3` with exponential backoff capped at `30` seconds;
- maximum concurrency `1`;
- structured outputs disabled;
- logprobs disabled;
- one questionnaire step with ID `personality` and `history: reset`.

Concurrency is across personas only; BFI-10 items remain sequential within each
persona. Keep local-provider concurrency conservative and increase it only after
measuring model-server throughput and memory use.

## Canonical Workflow

The notebook performs the same explicit CLI workflow shown below. The first
command stores the immutable protocol and creates its initial cohort without
calling Ollama:

```bash
uv run llm-behavior-lab protocol-create \
  --file examples/ollama-bfi10-factorial/protocol.json
```

The second command reuses that cohort and starts a distinct protocol run:

```bash
uv run llm-behavior-lab protocol-create \
  --file examples/ollama-bfi10-factorial/protocol.json \
  --new-run \
  --api-key ollama
```

To use a different immutable experiment configuration, edit `experiment_id` in
`protocol.json` to another valid three-part ID before creating it. Generated
artifacts are written under `experiments/{experiment_id}/` and must not be
committed.

If execution is interrupted, resume the exact run:

```bash
uv run llm-behavior-lab protocol-create \
  --file examples/ollama-bfi10-factorial/protocol.json \
  --run-id run-protocol-gemma4-12b-20260612090000 \
  --api-key ollama
```

Add `--retry-failed` only when latest failed or invalid item attempts should be
retried. Completed items are skipped.

Score and export the questionnaire step independently:

```bash
uv run llm-behavior-lab scale-score \
  --experiment-id ollama-bfi-ten \
  --run-id run-protocol-gemma4-12b-20260612090000 \
  --step-id personality

uv run llm-behavior-lab scale-results \
  --experiment-id ollama-bfi-ten \
  --run-id run-protocol-gemma4-12b-20260612090000 \
  --step-id personality
```

## Artifacts

The canonical workflow writes immutable protocol and cohort snapshots, then
keeps procedure artifacts inside the protocol step:

```text
experiments/{experiment_id}/
  protocol.json
  metadata.json
  cohorts/cohort-{uuid}/
    personas.json
    metadata.json
    protocol-assignments.json
  run-protocol-{model}-{timestamp}/
    run.json
    conversations/{subject_id}.jsonl
    steps/personality/
      run.json
      scale.json
      conversations/{subject_id}.jsonl
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

Response and conversation ledgers are append-only JSONL. The immutable cohort
stores the exact 12-person persona snapshot and factorial assignments. Response
rows reference personas by `subject_id`; `responses.csv` joins the cohort
snapshot and includes `persona_*` columns.
