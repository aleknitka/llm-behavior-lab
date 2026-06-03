# llm-psych-scales

Application for running LLM persona-based psychological questionnaire sessions and saving the resulting data as JSONL for analysis.

## Questionnaire Definitions

New questionnaires should be coded as Python data definitions under `src/llm_psych_scales/questionnaires/`. Use the Pydantic models in `llm_psych_scales.questionnaires.base` as the source-of-truth structure for questionnaire data.

The base structure separates questionnaire source data from run output:

- Questionnaire modules store the instrument definition: sections, items, response formats, scales, scoring rules, reference, and licence.
- Runtime JSONL files store model responses: one answered item per line, with stable questionnaire and item identifiers plus prompt messages, typed parsed answer, raw response, structured response, logprobs, status, and errors where available.

### Definition Layout

Import definition models from:

```python
from llm_psych_scales.questionnaires.base.response_formats import (
    LikertFormat,
    Option,
    SingleChoiceFormat,
)
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
```

Define one module-level `Questionnaire` constant per instrument or version:

```python
EXAMPLE_SCALE = Questionnaire(
    id="example_scale",
    name="Example Scale",
    version="1.0",
    language="en",
    sections=[
        Section(
            id="main",
            title="Example items",
            item_ids=["example_01", "example_02"],
        )
    ],
    items=[
        Item(
            id="example_01",
            code="1",
            order=1,
            text="I prefer predictable routines.",
            response_format=LikertFormat(
                min_value=1,
                max_value=5,
                labels={
                    1: "Strongly disagree",
                    5: "Strongly agree",
                },
            ),
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
    reference="Replace with the instrument citation or source URL.",
    licence="Replace with the licence or use restrictions.",
)
```

Use `SingleChoiceFormat` or `MultipleChoiceFormat` when answers are named categories rather than numeric Likert anchors:

```python
Item(
    id="example_choice_01",
    order=1,
    text="Which statement fits best?",
    response_format=SingleChoiceFormat(
        options=[
            Option(id="a", label="First option", value=1),
            Option(id="b", label="Second option", value=2),
        ]
    ),
)
```

Keep identifiers stable. `Item.id`, `Section.id`, `Scale.id`, and `ScoringModel.id` may be referenced by answer records, scoring code, and later analysis. Put source item numbers or published codes in `Item.code`; use `ItemMapping` for reverse scoring, scale membership, scoring weights, and scoring-specific metadata.

## Response Records

Runtime response data uses Pydantic models from `llm_psych_scales.responses.base`. Store run-level metadata in a `SessionRecord` and store each answered questionnaire item as an `ItemResponseRecord`.

```python
from datetime import UTC, datetime

from llm_psych_scales.responses.base import (
    ChatMessage,
    ItemResponseRecord,
    LikertAnswerValue,
    ProviderSnapshot,
    ResponseStatus,
    SessionRecord,
)

provider = ProviderSnapshot(
    provider_base_url="http://localhost:11434/v1",
    model="llama3.1",
    temperature=0.2,
    timeout_seconds=60.0,
    supports_structured_outputs=False,
    supports_logprobs=False,
)

session = SessionRecord(
    session_id="session-00000000-0000-4000-8000-000000000002",
    run_id="run-00000000-0000-4000-8000-000000000001",
    questionnaire_id="bfi_10",
    questionnaire_version="1.0",
    scoring_model_id=None,
    persona_id="persona-1",
    persona_snapshot={"age": 35, "country": "PL"},
    provider=provider,
    started_at=datetime.now(UTC),
    status=ResponseStatus.COMPLETED,
)

item_response = ItemResponseRecord(
    session_id=session.session_id,
    run_id=session.run_id,
    questionnaire_id=session.questionnaire_id,
    questionnaire_version=session.questionnaire_version,
    item_id="bfi10_01_reserved",
    item_order=1,
    item_text="I see myself as someone who is reserved",
    response_format_type="likert",
    messages=[
        ChatMessage(role="system", content="Assume the persona."),
        ChatMessage(role="user", content="Question text"),
    ],
    answer=LikertAnswerValue(value=2, label="Agree"),
    raw_response="2",
    structured_response={"selected_answer_id": "2"},
    logprobs=None,
    status=ResponseStatus.COMPLETED,
)
```

`ItemResponseRecord` is the JSONL analysis unit. It stores stable identifiers plus prompt, raw output, parsed answer, structured output, logprobs, error state, and metadata. It does not duplicate the full questionnaire definition.

## Storage Layout

Questionnaire outputs are stored under a strict experiment/session/run hierarchy:

```text
experiments/{experiment_id}/
  personas.jsonl
  sessions/{session_id}/
    session.jsonl
    runs/{run_id}/
      responses.jsonl
```

Experiment IDs must be exactly three lowercase letter-or-digit items joined with hyphens, such as `pilot-study-one`. Spaces, underscores, dots, path separators, and other symbols are rejected. If an experiment ID is omitted, the app generates a friendly three-item name.

Session IDs use `session-[uuid]`. Run IDs use `run-[uuid]`. A session can hold multiple runs, and each run directory contains only JSONL files.

## Running Questionnaires

`run_questionnaire` and `run_questionnaire_async` consume the new base `Questionnaire` objects and return `ItemResponseRecord` objects. The current BFI-10 definition is exported from `llm_psych_scales.questionnaires.bfi10`.

```python
from pathlib import Path

from llm_psych_scales.client import OpenAiChatClient
from llm_psych_scales.models import ModelSettings, Persona, ProviderCapabilities
from llm_psych_scales.questionnaires.bfi10 import BFI_10
from llm_psych_scales.runner import run_questionnaire

persona = Persona(
    persona_id="persona-1",
    features={"age": "35", "country": "Poland"},
)
settings = ModelSettings(
    model="llama3.1",
    provider_base_url="http://localhost:11434/v1",
    temperature=0.2,
    timeout_seconds=60.0,
    capabilities=ProviderCapabilities(),
)
client = OpenAiChatClient(api_key="ollama", base_url=settings.provider_base_url)

records = run_questionnaire(
    persona=persona,
    questionnaire=BFI_10,
    settings=settings,
    client=client,
    project_root=Path("."),
    experiment_id="pilot-study-one",
)

print(records[0].item_id, records[0].answer, records[0].status)
```

When `run_id` is omitted, the runner generates a `run-[uuid]` value. Each questionnaire call also gets a generated `session-[uuid]` value unless `session_id` is supplied. If `run_id` is supplied, pass either a bare UUID or an already prefixed `run-[uuid]`.

The CLI uses the same hierarchy and writes item-level response records:

```bash
uv run llm-psych-scales \
  --model llama3.1 \
  --base-url http://localhost:11434/v1 \
  --project-root . \
  --experiment-id pilot-study-one \
  --feature age=35 \
  --feature country=Poland
```

## PersonaFactory

`llm_psych_scales.personas.factory` generates batches of demographic personas for later questionnaire runs. The current implementation focuses on demographics, uses random enum/value selection, and applies basic realism constraints so generated personas avoid obvious contradictions.

Each generated persona has:

- `subject_id`: a UUID identifying the generated subject.
- `features`: a typed `Demographics` object from `llm_psych_scales.personas.dimensions`.

### Generate Personas

```python
from llm_psych_scales.personas.factory import (
    PersonaFactory,
    PersonaFactoryRequest,
    RequestedDemographicField,
)

request = PersonaFactoryRequest(
    count=10,
    requested_fields={
        RequestedDemographicField.AGE,
        RequestedDemographicField.GENDER,
        RequestedDemographicField.COUNTRY,
        RequestedDemographicField.REGION,
        RequestedDemographicField.EDUCATION_LEVEL,
        RequestedDemographicField.EMPLOYMENT_STATUS,
    },
    seed=123,
    experiment_id="pilot-study-one",
)

batch = PersonaFactory().create_demographics_batch(request)

print(len(batch))
for persona in batch.personas:
    print(persona.subject_id, persona.features.model_dump(exclude_none=True))
```

Use `seed` when you need reproducible persona generation. Omit `experiment_id` to let the factory create a friendly generated name such as `careful-bright-signal`.

### Available Demographic Fields

Request fields with `RequestedDemographicField`:

- `AGE`
- `GENDER`
- `EDUCATION_LEVEL`
- `EMPLOYMENT_STATUS`
- `COUNTRY`
- `REGION`
- `URBANICITY`
- `FAMILY_STATUS`
- `HOUSEHOLD_SIZE`
- `HAS_CHILDREN`
- `NUMBER_OF_DEPENDANTS`

Only requested fields are populated. Unrequested fields remain `None`.

### Save Personas as JSONL

```python
from pathlib import Path

from llm_psych_scales.storage import write_persona_batch_jsonl

path = write_persona_batch_jsonl(Path.cwd(), batch)
print(path)
```

This writes the experiment dump to:

```text
experiments/{experiment_id}/personas.jsonl
```

The dump contains a top-level `metadata` object and a `personas` list:

```json
{
  "metadata": {
    "experiment_id": "pilot-study-one",
    "persona_count": 10,
    "requested_fields": ["age", "country", "region"],
    "seed": 123
  },
  "personas": [
    {
      "subject_id": "00000000-0000-4000-8000-000000000000",
      "features": {
        "age": 35,
        "country": "PL",
        "region": "Mazowieckie"
      }
    }
  ]
}
```

Experiment IDs must contain exactly three lowercase letter-or-digit items separated by hyphens, for example `pilot-study-one`.

### Current Constraints

PersonaFactory currently uses uniform random selection after applying consistency rules. It avoids cases such as under-18 retirees, minors with children, doctorate holders below age 25, retirees below age 55, and household sizes smaller than the number of dependants. Weighted distributions, fixed values, and custom demographic mixes are planned for a later version.
