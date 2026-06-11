# llm-behavior-lab

Run reproducible persona-based LLM experiments with standardized
questionnaires, deterministic persona manipulations, and stateful behavioral
tasks.

The package targets OpenAI-compatible APIs, including local LM Studio and
Ollama endpoints. It validates experiment definitions with Pydantic, keeps
provider credentials at runtime, and persists analysis units as JSONL.

## Installation

```bash
uv sync
```

The default endpoint is LM Studio at `http://localhost:1234/v1`. Set
`OPENAI_BASE_URL` and `OPENAI_API_KEY` for another compatible provider.

## Discover Available Inputs

Inspect questionnaires and persona fields without loading credentials, calling
a model, or creating files:

```bash
uv run llm-behavior-lab questionnaire-list
uv run llm-behavior-lab questionnaire-describe bfi_10
uv run llm-behavior-lab persona-fields
```

Current questionnaire IDs are:

- `bfi_10`
- `consumer_involvement`
- `purchase_decision_making_inventory`

Use `--json` for machine-readable output. Questionnaire IDs are exact and
stable; shorthand aliases are not accepted.

Preview a persona request locally:

```bash
uv run llm-behavior-lab persona-preview \
  --experiment-id pilot-study-one \
  --persona-count 100 \
  --seed 123 \
  --persona-field age \
  --persona-field country
```

Persona configuration supports fixed values, weighted enum probabilities, and
deterministic inclusive integer ranges.

## Canonical Protocol Workflow

New multi-step experiments should use one immutable protocol containing persona
generation, shared non-secret provider settings, and ordered questionnaire or
task steps.

```json
{
  "version": "1.0",
  "experiment_id": "pilot-study-one",
  "name": "Pilot study",
  "persona_seed": 123,
  "run_seed": 456,
  "personas": {
    "count": 100,
    "requested_fields": ["age", "country"]
  },
  "provider": {
    "model": "openai/gpt-oss-20b",
    "base_url": "http://localhost:1234/v1",
    "temperature": 0,
    "timeout_seconds": 60
  },
  "steps": [
    {
      "id": "personality",
      "kind": "questionnaire",
      "questionnaire_id": "bfi_10",
      "scoring_model_id": "default",
      "history": "reset"
    },
    {
      "id": "decision-task",
      "kind": "task",
      "task_id": "four-deck-card-task",
      "task_config": {"trial_count": 100},
      "history": "inherit"
    }
  ]
}
```

Create the experiment and initial persona cohort:

```bash
uv run llm-behavior-lab protocol-create --file study.json
```

Run the same protocol again:

```bash
OPENAI_API_KEY=lm-studio \
uv run llm-behavior-lab protocol-create \
  --file study.json \
  --new-run \
  --run-seed 789
```

In a terminal, rerunning without `--new-run` prompts for confirmation, cohort
reuse, and an optional run seed. Scripts must pass `--new-run`.

- `--cohort-id` reuses an exact persona snapshot.
- `--persona-seed` creates or reuses the cohort for that seed.
- `--cohort-id` and `--persona-seed` are mutually exclusive.
- Changing only `persona_seed` or `run_seed` preserves protocol identity.
- Any other configuration change requires a new experiment ID.
- `history: reset` starts from the persona prompt.
- `history: inherit` carries the prior step conversation forward.

The initial `protocol-create` call validates and stores the protocol and cohort.
Subsequent confirmed or `--new-run` calls execute the protocol.

## Scoring and Task Analysis

Questionnaire steps snapshot their validated definition and can be scored and
exported independently:

```bash
uv run llm-behavior-lab scale-score \
  --experiment-id pilot-study-one \
  --run-id run-protocol-openai-gpt-oss-20b-20260611090000 \
  --step-id personality

uv run llm-behavior-lab scale-results \
  --experiment-id pilot-study-one \
  --run-id run-protocol-openai-gpt-oss-20b-20260611090000 \
  --step-id personality
```

Behavioral-task steps can be analyzed and exported with `task-analyze` and
`task-results` using the same `--run-id` and `--step-id` pattern.

The built-in `four-deck-card-task` implements configurable Iowa Gambling Task
contingencies without exposing the recognized task name, canonical deck labels,
advantageous classifications, or future outcomes to the model. Trials remain
sequential within a subject; subjects may run concurrently.

## Staged Compatibility Workflow

Existing single-procedure experiments can continue to use `design.json`:

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

Use `task-design`, `task-run`, `task-analyze`, and `task-results` for the staged
behavioral-task equivalent. Task runs support bounded cross-subject concurrency,
explicit run resumption, and opt-in retries of failed or twice-invalid subjects.
Questionnaire runs do not yet resume partially completed subjects.

## Runtime Layout

Canonical protocol experiments:

```text
experiments/{experiment_id}/
  protocol.json
  metadata.jsonl
  cohorts/
    cohort-{uuid}/
      personas.jsonl
      metadata.json
      protocol-assignments.jsonl
  run-protocol-{model}-{timestamp}/
    run.jsonl
    conversations/{subject_id}.jsonl
    steps/
      {questionnaire_step}/
        run.jsonl
        scale.json
        responses/{subject_id}.jsonl
      {task_step}/
        run.jsonl
        task.json
        schedules/{subject_id}.json
        conversations/{subject_id}.jsonl
        responses/{subject_id}.jsonl
```

Staged scale and task experiments retain their existing
`run-{questionnaire}-...` and `run-task-{task}-...` layouts.

Experiment IDs contain exactly three lowercase alphanumeric parts separated by
hyphens, such as `pilot-study-one`.

## Python API

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

The public API also exports `UnifiedExperimentProtocol`,
`create_protocol_experiment()`, `create_protocol_run()`,
`list_persona_fields()`, `preview_persona_creation()`, questionnaire runners,
task runners, scoring, and result exporters.

## Questionnaire Definitions

Questionnaire definitions live under `src/llm_behavior_lab/questionnaires/`
and use the shared Pydantic models under `questionnaires/base/`. Preserve source
wording, response formats, citations, licences, stable IDs, scale mappings, and
versioned scoring models when adding an instrument.

The Purchase Decision-Making Inventory includes
[source and implementation notes](src/llm_behavior_lab/questionnaires/pdmi/README.md).
Use `questionnaire-describe` to inspect metadata for any available instrument.

## Project Documentation

- [Experiment workflow](docs/experiment-workflow.md)
- [Questionnaire implementation models](src/llm_behavior_lab/questionnaires/base/scale.py)
- [Behavioral task protocol](src/llm_behavior_lab/behavioral_tasks/base.py)
- [Four-deck task implementation](src/llm_behavior_lab/behavioral_tasks/iowa_gambling.py)
- [Examples](examples/README.md)

## Development

```bash
uv run ruff check .
uv run ty check
uv run pytest
```

Tests use fake clients and require no live model calls.
