# Experiment Workflow

This guide describes the complete lifecycle from a study design to scored, tabular
results. Each stage is explicit so an experiment can be inspected, reproduced, or
restarted without silently regenerating earlier inputs.

## 1. Design

Create a validated manifest:

```bash
uv run llm-behavior-lab scale-design \
  --experiment-id pilot-study-one \
  --questionnaire bfi_10 \
  --persona-count 100 \
  --model openai/gpt-oss-20b \
  --base-url http://localhost:1234/v1 \
  --temperature 0 \
  --seed 123 \
  --scoring-model-id default
```

`design.json` records questionnaire identity, questionnaire parameters, persona or
protocol settings, model settings, seed, and scoring-model choice. It intentionally
does not contain an API key.

For target-dependent questionnaires, repeat `--questionnaire-param KEY=VALUE`:

```bash
--questionnaire consumer_involvement \
--questionnaire-param "target=meal delivery services"
```

Use `--protocol protocol.json` instead of `--persona-count` for a paired-factorial
design. The protocol remains validated by `ExperimentProtocol`.

## 2. Personas

```bash
uv run llm-behavior-lab personas --experiment-id pilot-study-one
```

For a simple design, this writes one deterministic `personas.jsonl` batch. For a
protocol design, it also writes:

- `base_personas.jsonl`
- `protocol.json`
- `protocol_assignments.jsonl`

The command refuses to overwrite an existing persona batch. Delete or archive the
whole experiment directory before intentionally creating a different design with the
same experiment ID.

## 3. Execution

Start the configured OpenAI-compatible provider, then run:

```bash
OPENAI_API_KEY=lm-studio \
uv run llm-behavior-lab scale-run --experiment-id pilot-study-one
```

`scale-run` loads `design.json` and `personas.jsonl`; it never samples personas. Each run
gets a new timestamped directory and snapshots the questionnaire as `scale.json`.
Every item response stores the prompt, typed answer, raw response, provider metadata,
seed, status, and protocol assignment metadata when applicable.

If execution fails before a complete run is written, retain the partial files for
diagnosis. Current staged execution does not resume partial runs; run the command
again to create a separate run directory.

## 4. Scoring

```bash
uv run llm-behavior-lab scale-score --experiment-id pilot-study-one
```

If multiple run directories exist:

```bash
uv run llm-behavior-lab scale-score \
  --experiment-id pilot-study-one \
  --run-id run-bfi10-openai-gpt-oss-20b-20260608120000
```

Scoring uses the versioned model in the run's `scale.json`. For legacy snapshots that
predate scoring models, it may use the current coded definition only when questionnaire
ID, version, item IDs, order, wording, and response formats match exactly. The scoring
manifest stores both definition digests and whether fallback occurred.

Scoring is strict: a scale with any missing, failed, invalid, or unsupported mapped
answer is written as `unscorable`; other scales for that subject are still evaluated.

## 5. Results

```bash
uv run llm-behavior-lab scale-results --experiment-id pilot-study-one
```

The command writes:

- `responses.csv`: flat item-level records.
- `scores.csv`: subject-scale results and metadata.
- `reliability.csv`: Cronbach alpha results.
- `summary.json`: count, mean, sample standard deviation, minimum, and maximum.

When more than one scoring directory exists, select one with
`--scoring-directory default-1.0`.

## Protocol Interpretation

Protocol scores retain `base_subject_id`, `condition_id`, `iteration_index`, and factor
metadata. Reliability is calculated for the complete run and separately for each
condition. A small condition can produce `not_computable`; inspect the reason and
subject/item counts rather than treating a missing alpha as zero.

## Reproducibility Checklist

1. Keep `design.json`, questionnaire snapshot, and protocol artifacts.
2. Record the provider implementation and model revision outside the API model string
   when the provider does not expose immutable revisions.
3. Use explicit seeds, but do not assume every provider guarantees deterministic
   sampling.
4. Preserve raw responses even after scoring.
5. Compare scoring-model IDs, versions, provenance, and definition digests before
   combining runs.
