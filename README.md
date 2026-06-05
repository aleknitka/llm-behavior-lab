# llm-psych-scales

![Developed with Codex](https://img.shields.io/badge/developed%20with-Codex-111827?style=for-the-badge)

Application for running LLM persona-based psychological questionnaire sessions and
saving JSONL data for later analysis. The project is designed to support multiple
standardized questionnaires, extensible persona feature maps, protocol-based
persona manipulations, and simple pairwise preference experiments.

This project was developed with OpenAI Codex.

The current CLI workflow runs BFI-10 by default. The shared questionnaire models
and runner accept standardized questionnaire definitions, so additional coded
instruments can be used from Python and exposed through the CLI as needed.

1. Generate a batch of demographic personas.
2. Optionally expand those personas through a paired or factorial protocol that
   manipulates selected persona fields across comparison conditions.
3. Ask an OpenAI-compatible chat model to answer each questionnaire item as each
   runtime persona.
4. Save one experiment directory containing the persona batch, run metadata, scale copy,
   and one response file per generated subject.

Current implemented surfaces include:

- Demographic persona generation with weighted enum sampling.
- BFI-10 batch runs through the CLI.
- Questionnaire base models for sections, items, response formats, scales, and
  scoring-model metadata.
- Runtime response models for run records and item-level JSONL records.
- Paired-factorial protocol expansion for manipulated persona fields.
- Pairwise preference-test helpers for blinded `A`/`B` stimulus comparisons.

## Quick Start

Install dependencies and run the default BFI-10 LM Studio batch. This assumes a
local OpenAI-compatible provider is already serving chat completions at
`http://localhost:1234/v1`.

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

The CLI currently exposes BFI-10. Use the Python runner APIs for other coded
questionnaires until additional CLI selection is added.

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

Provider capability flags are stored in run metadata so later analysis can tell
whether logprobs or structured-output support was requested for a given run.

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

Persona features are intentionally selected from explicit typed fields. This keeps
small experiments simple while making it straightforward to add more persona
dimensions later in `src/llm_psych_scales/personas/`.

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

## Protocol Experiments

Use `--protocol` for paired or factorial experiments where the same generated base
persona is cloned across manipulated factor levels and repeated iterations. This
supports simple designs for comparing how selected persona parameters influence
simulated questionnaire results while holding the remaining base persona features
constant.

Example `protocol.json`:

```json
{
  "version": "1.0",
  "name": "gender-affluence-factorial",
  "design": "paired_factorial",
  "base_persona_count": 20,
  "seed": 123,
  "iterations": 3,
  "requested_fields": [
    "age",
    "country",
    "education_level",
    "employment_status",
    "gender",
    "affluence_level"
  ],
  "base_persona_config": {
    "field_probabilities": {
      "country": {
        "PL": 1.0
      }
    }
  },
  "factors": [
    {
      "name": "gender",
      "field": "gender",
      "levels": [
        { "id": "female", "value": "female" },
        { "id": "male", "value": "male" }
      ]
    },
    {
      "name": "affluence",
      "field": "affluence_level",
      "levels": [
        { "id": "low", "value": "low" },
        { "id": "middle", "value": "middle" },
        { "id": "very_high", "value": "very_high" }
      ]
    }
  ]
}
```

Run with:

```bash
uv run llm-psych-scales \
  --experiment-id bfi10-protocol-test \
  --protocol protocol.json \
  --model openai/gpt-oss-20b \
  --base-url http://localhost:1234/v1 \
  --api-key lm-studio
```

The CLI currently runs the protocol with BFI-10. The protocol machinery is
questionnaire-agnostic in the runner, so additional coded questionnaires can use the
same persona expansion path from Python and can be exposed through the CLI later.

For each generated base persona, the runner creates every factor-level cell and every
iteration. In the example above, each base persona expands to `2 x 3 x 3 = 18`
runtime subjects. Persona features stay identical within the same base/cell across
iterations; only deterministic item call seeds vary.

Protocol mode writes additional experiment-level files:

```text
experiments/{experiment_id}/
  protocol.json
  base_personas.jsonl
  personas.jsonl
  protocol_assignments.jsonl
```

`protocol_assignments.jsonl` maps each expanded subject to its base subject,
condition ID, iteration index, factor values, and factor level IDs. Response metadata
contains those assignment identifiers without duplicating the full persona snapshot.

## Questionnaire Definitions

Questionnaires are coded as Python module-level constants under
`src/llm_psych_scales/questionnaires/`. Use
`llm_psych_scales.questionnaires.base.scale.Questionnaire` as the source-of-truth
object for new instruments.

Current coded questionnaires include:

- BFI-10
- Consumer Involvement Scale, built for a supplied product or service target
- [Purchase Decision-Making Inventory (PDMI)](src/llm_psych_scales/questionnaires/pdmi/README.md)

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

Local source papers used while coding new instruments can be kept in `.local_sources/`.
Keep that directory local and out of commits; do not commit source PDFs unless a commit
explicitly needs and names them.

### Consumer Involvement Scale

The consumer involvement questionnaire from Broderick and Mueller (1999) requires a
product or service category before it can be run. Build a target-specific copy and pass
that questionnaire to the runner:

Source DOI: <https://doi.org/10.1080/10696679.1999.11501855>

```python
from llm_psych_scales.questionnaires.consumer_involvement import (
    build_consumer_involvement_questionnaire,
)

questionnaire = build_consumer_involvement_questionnaire("meal delivery services")
```

The builder fills the source item blanks with the supplied target phrase, preserves the
source five-point agreement response format, and stores the target plus source DOI in
questionnaire metadata. The original empirical application was food shopping, so using
the scale for another product or service category should be treated as part of the study
design.

Target-dependent questionnaires should follow this pattern: keep a template constant
for inspection, expose a small builder that validates the target input, and store the
resolved product or service category in questionnaire metadata so downstream analysis
can tell which stimulus domain was used.

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

## Pairwise Preference Tests

Use `llm_psych_scales.preference_tests` for simple paired preference experiments.
Each trial presents two blinded stimulus versions as `A` and `B`, asks the persona
to choose one, and stores one JSONL record per persona-trial.

```python
from pathlib import Path

from llm_psych_scales.client import OpenAiChatClient
from llm_psych_scales.models import ModelSettings, ProviderCapabilities
from llm_psych_scales.preference_tests import (
    PairwisePreferenceExperiment,
    Stimulus,
    generate_pairwise_trials,
    run_pairwise_preference_batch,
)

stimulus_ids = ["direct", "social", "calm"]
experiment = PairwisePreferenceExperiment(
    id="landing-copy-test",
    name="Landing copy test",
    version="1.0",
    instruction="Choose the message you would be more likely to click.",
    stimuli=[
        Stimulus(id="direct", text="Start saving time today."),
        Stimulus(id="social", text="Join 10,000 teams saving time."),
        Stimulus(id="calm", text="A quieter way to manage your work."),
    ],
    trials=generate_pairwise_trials(stimulus_ids),
)

settings = ModelSettings(
    model="openai/gpt-oss-20b",
    provider_base_url="http://localhost:1234/v1",
    temperature=0.0,
    timeout_seconds=60.0,
    seed=123,
    capabilities=ProviderCapabilities(supports_logprobs=True),
)
client = OpenAiChatClient(api_key="lm-studio", base_url=settings.provider_base_url)

result = run_pairwise_preference_batch(
    experiment=experiment,
    settings=settings,
    client=client,
    project_root=Path("."),
    experiment_id="pref-study-one",
    persona_count=20,
    seed=123,
)
print(result.experiment_id, result.runs[0].run_id)
```

Pairwise preference runs write `experiment.json`, `run.jsonl`, and per-subject
JSONL files under `responses/`. The module also includes helpers to flatten
records and summarize stimulus win counts.

The runner randomizes which stimulus is displayed as `A` or `B` for each
persona-trial using a deterministic hash of the run seed, persona ID, and trial ID.
This reduces fixed-position bias while preserving reproducibility, and each record
stores `displayed_stimulus_ids` so later analysis can audit or model order effects.
The current implementation does not enforce exact 50/50 counterbalancing across
personas; larger samples should approach balance through randomization, while small
studies that require strict counterbalancing should add explicit allocation logic.

The methodology follows standard paired-comparison / two-alternative forced-choice
practice:

- Thurstone introduced paired comparative judgment as a way to scale qualitative
  judgments and preferences: <https://brocku.ca/MeadProject/Thurstone/Thurstone_1927f.html>
- Paired-comparison data can be summarized into preference proportions and later
  modeled with Thurstone or Bradley-Terry approaches:
  <https://www.mathematicalpsychology.com/Paired_Comparisons>
- Brown and Peterson review paired-comparison reliability, consistency, and scaling:
  <https://research.fs.usda.gov/treesearch/31863>
- ISO 5495 describes the paired comparison test as a forced-choice test between
  two alternatives, also known as 2-AFC:
  <https://www.iso.org/standard/31621.html>
- Modern Bradley-Terry extensions can handle ties, order effects, subject
  predictors, and hierarchical models, which are useful future analysis directions:
  <https://link.springer.com/article/10.3758/s13428-021-01714-2>

## Development

Application code lives under `src/` and tests live under `tests/`. Dependencies are
managed with `uv`.

Run the standard checks before considering implementation work complete:

```bash
uv run ruff check .
uv run ty check
uv run pytest
```

Questionnaire definitions should be validated at import or load time through the
Pydantic base models in `llm_psych_scales.questionnaires.base`. Runtime output
should use the response models in `llm_psych_scales.responses.base` and remain
stable enough for downstream analysis.
