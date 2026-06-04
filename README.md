# llm-psych-scales

![Developed with Codex](https://img.shields.io/badge/developed%20with-Codex-111827?style=for-the-badge)

Application for running LLM persona-based psychological questionnaire batches and saving
JSONL data for later analysis.

This project was developed with OpenAI Codex.

The current default workflow is:

1. Generate a batch of demographic personas.
2. Ask an OpenAI-compatible chat model to answer BFI-10 as each persona.
3. Save one experiment directory containing the persona batch, run metadata, scale copy,
   and one response file per generated subject.

## Quick Start

Install dependencies and run the default LM Studio batch:

```bash
uv sync

uv run llm-psych-scales \
  --project-root . \
  --experiment-id bfi10-lmstudio-test \
  --persona-count 100 \
  --model openai/gpt-oss-20b \
  --base-url http://localhost:1234/v1 \
  --api-key lm-studio
```

Defaults are tuned for LM Studio:

- `--base-url http://localhost:1234/v1`
- `--api-key lm-studio`
- `--model openai/gpt-oss-20b`
- `--persona-count 100`
- `--logprobs` enabled by default
- structured outputs disabled by default

If `--experiment-id` is omitted, a valid three-part experiment ID is generated.
Experiment IDs must match `item-item-item`, using only lowercase letters and digits
inside each item, for example `pilot-study-one`.

## Provider Configuration

The CLI accepts provider settings directly:

```bash
uv run llm-psych-scales \
  --base-url http://localhost:1234/v1 \
  --api-key lm-studio \
  --model openai/gpt-oss-20b
```

It can also read a project-root `.env` file:

```dotenv
OPENAI_BASE_URL=http://localhost:1234/v1
OPENAI_API_KEY=lm-studio
```

Resolution order is CLI argument, process environment, `.env`, then built-in default.
`.env` is ignored by git.

Use `--log-level DEBUG` to see detailed progress from loguru. If logprobs are requested
but the provider rejects them, the client logs a warning, retries the call without
logprobs, and stores `logprobs` as `null`.

## Reproducibility

Use `--seed` to make persona generation and per-item model call seeds deterministic:

```bash
uv run llm-psych-scales \
  --experiment-id bfi10-lmstudio-test \
  --seed 123
```

The base seed is stored in `run.jsonl` metadata. Each questionnaire item call receives
a deterministic item-specific seed, and that seed is stored in each response record's
metadata.

## Storage Layout

Runtime data is written under `experiments/{experiment_id}/`:

```text
experiments/{experiment_id}/
  personas.jsonl
  metadata.jsonl
  run-{questionnaire}-{model}-{timestamp}/
    run.jsonl
    scale.json
    responses/
      {subject_id}.jsonl
```

`personas.jsonl` stores the generated persona batch once. Response files only include
`subject_id`; they do not duplicate the full persona data.

`run.jsonl` contains one `RunRecord` for the batch. `metadata.jsonl` mirrors run-level
metadata at the experiment root. Each file under `responses/` contains one
`ItemResponseRecord` per questionnaire item for that subject.

## Persona Generation

Generate typed demographic personas directly:

```python
from llm_psych_scales.personas.factory import (
    PersonaFactory,
    PersonaFactoryRequest,
    RequestedDemographicField,
)

request = PersonaFactoryRequest(
    count=10,
    requested_fields=set(RequestedDemographicField),
    seed=123,
    experiment_id="pilot-study-one",
)

batch = PersonaFactory().create_demographics_batch(request)

for persona in batch.personas:
    print(persona.subject_id, persona.features.model_dump(exclude_none=True))
```

Supported requested fields are:

- `age`
- `gender`
- `education_level`
- `employment_status`
- `affluence_level`
- `country`
- `region`
- `urbanicity`
- `family_status`
- `household_size`
- `has_children`
- `number_of_dependants`

Only requested fields are populated. The factory applies basic realism constraints,
such as avoiding under-18 retirees, minors with children, doctorate holders below
age 25, retirees below age 55, and household sizes smaller than dependants.

## Weighted Persona Config

Use `--persona-config` to restrict enum fields and assign sampling probabilities.

Example `persona-config.json`:

```json
{
  "field_probabilities": {
    "country": {
      "PL": 0.8,
      "DE": 0.2
    },
    "affluence_level": {
      "middle": 1.0
    }
  }
}
```

Run with:

```bash
uv run llm-psych-scales \
  --experiment-id bfi10-lmstudio-test \
  --persona-config persona-config.json
```

Weighted config currently supports enum-backed persona fields such as `country`,
`gender`, `education_level`, `employment_status`, `affluence_level`, `urbanicity`,
and `family_status`. Invalid enum values or non-positive probabilities are rejected.

## Questionnaire Definitions

Questionnaires are coded as Python module-level constants under
`src/llm_psych_scales/questionnaires/`. Use
`llm_psych_scales.questionnaires.base.scale.Questionnaire` as the source-of-truth
object for new instruments.

The base structure separates questionnaire definitions from runtime output:

- Questionnaire modules store sections, items, response formats, scales, scoring
  rules, reference, and licence.
- Runtime JSONL files store stable questionnaire and item identifiers, prompt
  messages, typed parsed answers, raw responses, structured responses, logprobs,
  statuses, errors, and metadata.

Minimal example:

```python
from llm_psych_scales.questionnaires.base.response_formats import LikertFormat
from llm_psych_scales.questionnaires.base.scale import (
    Item,
    ItemMapping,
    Questionnaire,
    Scale,
    ScaleScoringRule,
    ScoringModel,
    Section,
    Transformation,
)

EXAMPLE_SCALE = Questionnaire(
    id="example_scale",
    shorthand="example",
    name="Example Scale",
    version="1.0",
    sections=[Section(id="main", item_ids=["example_01", "example_02"])],
    items=[
        Item(
            id="example_01",
            code="1",
            order=1,
            text="I prefer predictable routines.",
            response_format=LikertFormat(min_value=1, max_value=5),
        ),
        Item(
            id="example_02",
            code="2",
            order=2,
            text="I enjoy unfamiliar situations.",
            response_format=LikertFormat(min_value=1, max_value=5),
        ),
    ],
    scales=[
        Scale(
            id="novelty_preference",
            name="Novelty Preference",
            construct="Preference for novelty over routine",
            item_mappings=[
                ItemMapping(item_id="example_01", reverse_scored=True),
                ItemMapping(item_id="example_02"),
            ],
        )
    ],
    scoring_models=[
        ScoringModel(
            id="default",
            name="Default mean score",
            version="1.0",
            scale_rules=[
                ScaleScoringRule(
                    scale_id="novelty_preference",
                    transformation=Transformation.MEAN,
                    output_min=1,
                    output_max=5,
                )
            ],
        )
    ],
)
```

Keep `Item.id`, `Section.id`, `Scale.id`, and `ScoringModel.id` stable because run
records and analysis code may refer to them later.

## Running From Python

```python
from pathlib import Path

from llm_psych_scales.client import OpenAiChatClient
from llm_psych_scales.models import ModelSettings, ProviderCapabilities
from llm_psych_scales.questionnaires.bfi10 import BFI_10
from llm_psych_scales.runner import run_persona_questionnaire_batch

settings = ModelSettings(
    model="openai/gpt-oss-20b",
    provider_base_url="http://localhost:1234/v1",
    temperature=0.0,
    timeout_seconds=60.0,
    seed=123,
    capabilities=ProviderCapabilities(supports_logprobs=True),
)
client = OpenAiChatClient(api_key="lm-studio", base_url=settings.provider_base_url)

result = run_persona_questionnaire_batch(
    questionnaire=BFI_10,
    settings=settings,
    client=client,
    project_root=Path("."),
    experiment_id="bfi10-lmstudio-test",
    persona_count=100,
    seed=123,
)

print(result.experiment_id, result.runs[0].run_id)
```

`run_questionnaire` and `run_questionnaire_async` are also available for single
runtime personas and return `ItemResponseRecord` objects.

## Response Table Extraction

Use `llm_psych_scales.analysis` to flatten response JSONL into tabular rows:

```python
from pathlib import Path

from llm_psych_scales.analysis import load_response_table, write_response_table_csv

rows = load_response_table(
    Path("experiments/bfi10-lmstudio-test/run-bfi10-openai-gpt-oss-20b-20260603120000")
)
write_response_table_csv(rows, Path("analysis/bfi10-responses.csv"))
```

`load_response_table` accepts a single subject JSONL file, a `responses/` directory,
or a run directory containing `responses/`.

## Development

Run the project checks before merging changes:

```bash
uv run pytest -q
uv run ruff check .
uv run ty check
```
