# Experiment Workflow

This guide describes the complete lifecycle from a study design to scored, tabular
results. Each stage is explicit so an experiment can be inspected, reproduced, or
restarted without silently regenerating earlier inputs.

## Canonical Protocol Workflow

New experiments can define persona creation, provider settings, questionnaires,
and behavioral tasks in one immutable `protocol.json`:

```bash
uv run llm-behavior-lab protocol-create --file study.json
```

Each ordered step declares `history: reset` or `history: inherit`. Provider
credentials are runtime-only and are never stored in the protocol.

When an experiment already exists, an interactive terminal confirms the rerun
and cohort choice. Scripts must be explicit:

```bash
uv run llm-behavior-lab protocol-create \
  --file study.json \
  --new-run \
  --cohort-id cohort-00000000-0000-4000-8000-000000000000 \
  --run-seed 456
```

`--cohort-id` reuses the exact persona snapshot. `--persona-seed` creates a new
cohort and cannot be combined with `--cohort-id`. Seed defaults do not define
protocol identity; changing any other configuration requires a new experiment
ID.

```text
experiments/{experiment_id}/
  protocol.json
  cohorts/cohort-{uuid}/
    personas.json
    metadata.json
    protocol-assignments.json
  run-protocol-{model}-{timestamp}/
    run.json
    steps/{step_id}/
    conversations/
```

Run metadata records the protocol fingerprint, cohort ID, effective seeds,
provider snapshot, and step results. Use `--step-id` with scoring and task
analysis commands. The staged `design.json` workflow below remains supported for
existing single-procedure and factor-only experiments.

## 0. Discover a Questionnaire

List available instruments:

```bash
uv run llm-behavior-lab questionnaire-list
```

Inspect one instrument before designing the study:

```bash
uv run llm-behavior-lab questionnaire-describe consumer_involvement
```

The description reports the stable ID, version, language, citation, licence,
source URL, item count, scale IDs, scoring-model IDs, response formats, and
required parameters. Add `--json` to either command when generating scripts or
other tooling.

Discovery is local and read-only. It does not load provider credentials, contact
an API, or create files. Use the exact stable ID shown by discovery; shorthand
aliases such as `bfi10` and `pdmi` are not accepted.

Inspect persona fields and preview the resolved generation request:

```bash
uv run llm-behavior-lab persona-fields
uv run llm-behavior-lab persona-preview \
  --experiment-id pilot-study-one \
  --persona-count 100 \
  --seed 123 \
  --persona-field age \
  --persona-field country
```

Add `--json` for machine-readable output. Preview validates the same
`PersonaDesign` and `PersonaFactoryRequest` used during materialization without
creating an experiment directory.

## 1. Design

Create a validated manifest:

```bash
uv run llm-behavior-lab scale-design \
  --experiment-id pilot-study-one \
  --questionnaire bfi_10 \
  --persona-count 100 \
  --persona-field age \
  --persona-field country \
  --model openai/gpt-oss-20b \
  --base-url http://localhost:1234/v1 \
  --temperature 0 \
  --seed 123 \
  --scoring-model-id default
```

`design.json` records questionnaire identity, questionnaire parameters, persona or
protocol settings, model settings, seed, and scoring-model choice. It intentionally
does not contain an API key.

`--scoring-model-id` is optional. When supplied, `scale-design` validates that the
questionnaire exposes that model. When omitted, `scale-score` uses the first
executable scoring model in the snapshotted questionnaire. A later
`scale-score --scoring-model-id ...` explicitly overrides the design choice.

For target-dependent questionnaires, repeat `--questionnaire-param KEY=VALUE`:

```bash
--questionnaire consumer_involvement \
--questionnaire-param "target=meal delivery services"
```

`scale-design` rejects missing, blank, unknown, or unexpected questionnaire
parameters before it writes the experiment manifest.

Use `--protocol protocol.json` instead of `--persona-count` for a paired-factorial
design. The protocol remains validated by `ExperimentProtocol`.

## 2. Personas

```bash
uv run llm-behavior-lab personas --experiment-id pilot-study-one
```

For a simple design, this writes one deterministic `personas.json` batch. For a
protocol design, it also writes:

- `base_personas.json`
- `protocol.json`
- `protocol_assignments.json`

The command refuses to overwrite an existing persona batch. Use `--replace` only
when intentionally rematerializing the persisted design; it never changes
`design.json`.

Python callers use the same path:

```python
from pathlib import Path

from llm_behavior_lab import PersonaDesign, create_personas

batch = create_personas(
    Path("."),
    "pilot-study-one",
    PersonaDesign(
        count=100,
        seed=123,
        requested_fields={"age", "country"},
    ),
)
```

## 3. Execution

Start the configured OpenAI-compatible provider, then run:

```bash
OPENAI_API_KEY=lm-studio \
uv run llm-behavior-lab scale-run --experiment-id pilot-study-one
```

`scale-run` loads `design.json` and `personas.json`; it never samples personas. Each run
gets a new timestamped directory and snapshots the questionnaire as `scale.json`.
Every item response stores the prompt, typed answer, raw response, seed, status, and
protocol assignment metadata when applicable. Provider and model settings are
snapshotted once in `run.json`.

Complete snapshots use JSON and are written atomically. Append-only response,
conversation, scoring, and analysis ledgers use JSONL. Result exports resolve
persona features from the experiment or cohort snapshot by `subject_id`; response
records do not duplicate the persona document.

The run snapshot is written with `partial` status before provider calls begin. Resume
an interrupted staged run with `scale-run --run-id RUN_ID`; completed items are not
called again. Pass `--retry-failed` to retry items whose latest attempt failed.

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
If the questionnaire has no executable scoring model, the command exits with a clear
validation error instead of creating partial scoring output.

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
