# llm-behavior-lab

Run reproducible LLM experiments with generated personas, standardized
questionnaires, and stateful behavioral tasks.

The persona remains the system prompt throughout an experiment. Questionnaire
items or task observations are then presented as user messages through an
OpenAI-compatible provider.

## Installation

```bash
uv sync
```

The default provider is an OpenAI-compatible LM Studio endpoint at
`http://localhost:1234/v1`. Ollama, hosted APIs, and other compatible endpoints
use the same client boundary.

## Scale Workflow

Inspect the coded questionnaires before designing a study:

```bash
uv run llm-behavior-lab questionnaire-list
uv run llm-behavior-lab questionnaire-describe bfi_10
```

Add `--json` to either command for machine-readable metadata. Discovery does not
load provider credentials, contact a model, or create experiment artifacts.
Questionnaire selection uses exact stable IDs; the current IDs are `bfi_10`,
`consumer_involvement`, and `purchase_decision_making_inventory`.

Create a canonical multi-step experiment from one protocol file:

```bash
uv run llm-behavior-lab protocol-create --file study.json
uv run llm-behavior-lab protocol-create \
  --file study.json \
  --new-run \
  --cohort-id cohort-00000000-0000-4000-8000-000000000000 \
  --run-seed 456
```

`protocol.json` stores persona generation, shared non-secret provider settings,
and ordered questionnaire or behavioral-task steps. Step history is explicit:
`reset` starts from the persona prompt, while `inherit` carries the prior step's
conversation forward. API keys remain runtime-only.

Reusing an experiment ID requires the same protocol fingerprint. `persona_seed`
and `run_seed` may change between runs; any other change requires a new
experiment ID. A changed persona seed creates a new immutable cohort, while a
changed run seed reuses the selected personas.

Inspect and preview persona configuration locally:

```bash
uv run llm-behavior-lab persona-fields
uv run llm-behavior-lab persona-preview \
  --experiment-id pilot-study-one \
  --persona-count 100 \
  --seed 123 \
  --persona-field age \
  --persona-field country
```

Add `--json` for machine-readable output. These commands do not create files,
load provider credentials, or contact a model.

```bash
uv run llm-behavior-lab scale-design \
  --experiment-id pilot-study-one \
  --questionnaire bfi_10 \
  --persona-count 100 \
  --persona-field age \
  --persona-field country \
  --model openai/gpt-oss-20b \
  --seed 123

uv run llm-behavior-lab personas --experiment-id pilot-study-one

OPENAI_API_KEY=lm-studio \
uv run llm-behavior-lab scale-run --experiment-id pilot-study-one

uv run llm-behavior-lab scale-score --experiment-id pilot-study-one
uv run llm-behavior-lab scale-results --experiment-id pilot-study-one
```

Scale runs snapshot the questionnaire, retain item-level JSONL responses, apply
versioned scoring models, and export response, score, and reliability tables.
See [Questionnaire Scoring](docs/questionnaire-scoring.md).

## Task Workflow

The built-in `four-deck-card-task` implements configurable Iowa Gambling
contingencies without exposing the recognized task name, canonical deck labels,
or future outcomes to the model.

```bash
uv run llm-behavior-lab task-design \
  --experiment-id card-study-one \
  --task four-deck-card-task \
  --task-config task-config.json \
  --persona-count 100 \
  --model openai/gpt-oss-20b \
  --seed 123

uv run llm-behavior-lab personas --experiment-id card-study-one

OPENAI_API_KEY=lm-studio \
uv run llm-behavior-lab task-run \
  --experiment-id card-study-one \
  --concurrency 4

uv run llm-behavior-lab task-analyze --experiment-id card-study-one
uv run llm-behavior-lab task-results --experiment-id card-study-one
```

Omit `--task-config` to use the classic four-deck contingency defaults. A task
configuration can change the trial count, starting balance, payoff templates,
schedule assignment, visible labels, and feedback fields.

Minimal `task-config.json` example:

```json
{
  "trial_count": 100,
  "starting_balance": 2000,
  "schedule_mode": "template",
  "schedule_assignment": "shared",
  "shuffle_template_blocks": false,
  "visible_labels": ["Circle", "Square", "Triangle", "Star"],
  "feedback_fields": ["gain", "penalty", "net", "balance"]
}
```

Task trials are sequential within each persona, while personas may run
concurrently. Successful transitions are appended immediately. Re-running
`task-run` with the same `--run-id` resumes each subject from its validated
ledger. Without `--run-id`, `task-run` creates a new timestamped run.

Failed or twice-invalid subjects remain stopped during normal resume. Add
`--retry-failed` to explicitly retry them without discarding their prior
attempt records.

See [Behavioral Task Authoring](docs/behavioral-tasks.md) and
[Four-Deck Card Task](docs/iowa-gambling-task.md).

## Artifacts

```text
experiments/{experiment_id}/
  design.json
  personas.jsonl
  metadata.jsonl

  run-{questionnaire}-{model}-{timestamp}/       # Scale run
    run.jsonl
    responses/{subject_id}.jsonl
    scale.json
    scoring/{model_id}-{version}/
    results/{model_id}-{version}/

  run-task-{task}-{model}-{timestamp}/            # Behavioral-task run
    run.jsonl
    responses/{subject_id}.jsonl
    task.json
    schedules/{subject_id}.json
    conversations/{subject_id}.jsonl
    analysis/{task_id}-{version}/summaries.jsonl
    results/{task_id}-{version}/
```

Experiment IDs contain exactly three lowercase alphanumeric parts separated by
hyphens, for example `pilot-study-one`.

## Experiment Design

Each `design.json` has one discriminated procedure:

- `scale`: questionnaire identity, parameters, scoring model, and optional
  context.
- `task`: task identity and validated task configuration.

Persona materialization and paired or factorial persona protocols are shared by
both staged workflows.

These staged `design.json` commands remain supported for existing experiments.
New multi-procedure studies should use the canonical `protocol-create` workflow.

Canonical protocol artifacts use this layout:

```text
experiments/{experiment_id}/
  protocol.json
  cohorts/cohort-{uuid}/
    personas.jsonl
    metadata.json
    protocol-assignments.jsonl
  run-protocol-{model}-{timestamp}/
    run.jsonl
    steps/{step_id}/
    conversations/
```

Use `--step-id` with `scale-score`, `scale-results`, `task-analyze`, and
`task-results` to operate on a procedure inside a protocol run.

`personas` refuses to replace an existing snapshot. Use `--replace` only when
intentionally rematerializing the exact persisted design.

## Questionnaire Definitions

Questionnaire definitions live under
`src/llm_behavior_lab/questionnaires/`. Behavioral task definitions live under
`src/llm_behavior_lab/behavioral_tasks/`.

Current questionnaires include BFI-10, Consumer Involvement, and the
[Purchase Decision-Making Inventory](src/llm_behavior_lab/questionnaires/pdmi/README.md).
Use `questionnaire-describe` to inspect each instrument's citation, licence,
response formats, scales, scoring models, and required builder parameters.

## Python API

Create personas without constructing factory requests directly:

```python
from pathlib import Path

from llm_behavior_lab import PersonaDesign, create_personas
from llm_behavior_lab.personas import RandUniformRange
from llm_behavior_lab.personas.factory import PersonaGenerationConfig

design = PersonaDesign(
    count=100,
    seed=123,
    requested_fields={"age", "country"},
    generation_config=PersonaGenerationConfig(
        field_values={
            "age": RandUniformRange(20, 35),
            "country": "PL",
        }
    ),
)
batch = create_personas(Path("."), "pilot-study-one", design)
```

`list_persona_fields()` reports supported fields and configuration capabilities.
`preview_persona_creation()` returns the validated factory request without
creating files.

```python
from pathlib import Path

from llm_behavior_lab.behavioral_tasks.iowa_gambling import (
    IowaGamblingConfig,
    IowaGamblingTask,
)
from llm_behavior_lab.behavioral_tasks.runner import run_behavioral_task

task = IowaGamblingTask(IowaGamblingConfig(trial_count=100))
schedule = task.resolve_schedule(seed=123, subject_id=persona.persona_id)

result = run_behavioral_task(
    persona=persona,
    task=task,
    settings=settings,
    client=client,
    project_root=Path("."),
    experiment_id="card-study-one",
    run_id="run-task-example",
    resolved_schedule=schedule,
)
```

## Development

```bash
uv run ruff check .
uv run ty check
uv run pytest
```

Tests use fake clients and require no live model calls.
