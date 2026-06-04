# Agent Guidance

## Project Purpose

This repository is an application for running LLM-based psychological questionnaire sessions with generated personas.

The project should grow beyond a single questionnaire or fixed persona set. It is
intended to support multiple standardized questionnaires, extensible persona feature
maps, and simple experimental designs that compare how selected persona parameters
influence simulated questionnaire results.

The app should allow a user to:

1. Create a persona from demographic inputs selected from a map of available features.
2. Ask an OpenAI-compatible LLM to assume that persona.
3. Optionally expand generated personas through paired or factorial protocol designs
   that manipulate selected fields while keeping the remaining base persona stable.
4. Run a chat session with that persona by asking questions from standard psychological questionnaires.
5. Save response data in JSONL format for later analysis.

## Development Approach

- Put application code under `src/`.
- Before adding or changing code, inspect the current codebase and follow the patterns already present.
- Keep code simple, explicit, and easy to maintain.
- Do not create unnecessary abstractions, framework layers, services, registries, factories, or directory structures.
- Prefer small functions and straightforward data models over complex object hierarchies.
- Add structure only when it is clearly needed by existing behavior or tests.
- Make expansions easy to maintain: new persona fields should fit the existing
  persona models and factory flow, and new questionnaires should fit the shared
  questionnaire base models.

## Dependencies and Tooling

- Manage dependencies with `uv`.
- Use `ruff` for linting and code audit.
- Use `ty` for type checking.
- Use `pytest` for unit tests.
- Use `pydantic` for data validation and structured data models.
- Build prompts with `jinja2`.
- The LLM integration should target OpenAI-compatible APIs.
- LangChain or LangGraph may be used if they materially simplify questionnaire execution or state handling, but they should not be introduced by default if direct API calls are clearer.

## Runtime Model

- Design the application with async-capable execution.
- Async code should be available for API providers and workloads that benefit from concurrent calls.
- The app must also remain executable in non-async or sync-oriented environments, especially local model providers such as Ollama and LM Studio.
- Keep provider boundaries simple enough that OpenAI-compatible hosted models and local OpenAI-compatible endpoints can both be used.
- Do not make async support force unnecessary complexity into modules that can stay synchronous.

## Core Workflow

The standard workflow is:

1. Load a map of demographic persona features.
2. Let the user select or define demographic inputs.
3. Render a persona prompt from those inputs.
4. Ask the LLM to assume the persona.
5. Optionally expand base personas into protocol conditions and iterations for parameter-comparison designs.
6. Load one or more questionnaire definitions.
7. Ask questions one by one using each item's response format.
8. Retain prior question and answer context during the questionnaire chat when the questionnaire requests it.
9. Capture the model answer, structured output if available, logprobs if available, and relevant metadata.
10. Append records to JSONL for later analysis.

## Questionnaire Format

Multiple questionnaires should be runnable through a standardized form.

New questionnaire source data should be stored as Python module-level constants under `src/llm_psych_scales/questionnaires/` and built with the Pydantic models in `src/llm_psych_scales/questionnaires/base/`.

Use `llm_psych_scales.questionnaires.base.scale.Questionnaire` as the source-of-truth object for newly coded questionnaires. It stores:

- `id`, `name`, `version`, optional `language`, `reference`, and `licence`.
- `sections`, where each `Section` groups ordered `item_ids`.
- `items`, where each `Item` has a stable `id`, optional source `code`, display `order`, item `text`, `response_format`, and optional tags or metadata.
- `scales`, where each `Scale` names the measured construct and maps items through `ItemMapping`, including reverse scoring, weights, scoring roles, and item-level scoring metadata.
- `scoring_models`, where each `ScoringModel` defines versioned `ScaleScoringRule` entries using `sum`, `mean`, or `weighted_mean` transformations plus optional output ranges and interpretation bands.
- `metadata` for questionnaire-level details that do not deserve first-class fields yet.

Use `llm_psych_scales.questionnaires.base.response_formats` for response data:

- `LikertFormat` for numeric Likert scales with optional anchor labels.
- `NumericFormat` for free numeric responses and optional units or bounds.
- `SingleChoiceFormat` and `MultipleChoiceFormat` with explicit `Option` objects.
- `TextFormat` for open text responses.

When coding a questionnaire:

- Preserve the wording, order, response anchors, citation/reference, and licence from the source instrument.
- Keep `Item.id`, `Section.id`, `Scale.id`, and `ScoringModel.id` stable because run records and analysis code may refer to them later.
- Put source item numbers or public scale codes in `Item.code`; do not bake them into the stable `id` if the id should remain project-controlled.
- Put reverse scoring, subscale membership, scoring weights, and scoring-specific answer mappings in `Scale.item_mappings`, not in ad hoc comments.
- Prefer explicit metadata fields over implicit naming conventions when extra source details must be retained.
- Validate questionnaire definitions with Pydantic at import or load time before execution.

## Structured Outputs and Logprobs

- Use structured outputs where the selected provider supports them.
- Request logprobs where the selected provider supports them.
- Provider support may vary, especially for local models, so code should degrade cleanly when structured outputs or logprobs are unavailable.
- Persist enough metadata in JSONL to know which provider, model, prompt, questionnaire, persona, and capability settings produced each answer.

## Persistence

- Save questionnaire run data as JSONL.
- Each line should represent a clear analysis unit, preferably one answered item with its questionnaire, prompt messages, typed answer, raw response, structured response, logprobs if available, status, errors, and run/session metadata.
- Keep the JSONL schema stable and validate records with Pydantic before writing.
- Do not duplicate full questionnaire definitions into every result line. Store questionnaire definitions in code under `src/llm_psych_scales/questionnaires/`; store result lines with stable questionnaire, item, scale, model, prompt, persona, provider, and response identifiers or metadata needed for later analysis.
- Current runtime data is stored under `experiments/{experiment_id}/run-{questionnaire}-{model}-{timestamp}/`.
- Experiment roots can also contain `personas.jsonl`, `metadata.jsonl`, and, for protocol runs, `protocol.json`, `base_personas.jsonl`, and `protocol_assignments.jsonl`.
- Experiment IDs must be exactly three lowercase letter-or-digit items separated by hyphens, such as `pilot-study-one`; reject spaces, underscores, dots, path separators, and other symbols.
- Session IDs must be `session-[uuid]`.
- Run IDs currently use the directory form `run-{questionnaire}-{model}-{timestamp}` unless explicitly supplied in lower-level runner calls.
- Run directories currently contain `run.jsonl`, `scale.json`, and `responses/{subject_id}.jsonl`.

## Response and Session Data

Runtime response data should use the Pydantic models in `src/llm_psych_scales/responses/base/`.

- Use `RunRecord` for run-level metadata: experiment and session IDs, run ID, subject IDs, questionnaire ID and version, provider snapshot, timestamps, status, output paths, and metadata.
- Use `SessionRecord` when session-level metadata needs to be persisted separately.
- Use `ItemResponseRecord` for item-level JSONL analysis units: stable questionnaire and item IDs, item order/text, response format type, prompt messages, parsed answer, raw response, structured response, logprobs, status, errors, and metadata.
- Use typed answer values instead of ad hoc nullable fields: `LikertAnswerValue`, `NumericAnswerValue`, `SingleChoiceAnswerValue`, `MultipleChoiceAnswerValue`, and `TextAnswerValue`.
- Keep provider-specific or experiment-specific details inside explicit `metadata` fields unless they become stable first-class fields.
- Do not add scoring outputs to response records until scoring is implemented against questionnaire `scales` and `scoring_models`.

## Testing

- Add unit tests with `pytest` for data validation, questionnaire loading, prompt rendering, provider capability handling, and JSONL serialization.
- Prefer tests that exercise behavior without requiring live model calls.
- Mock or fake OpenAI-compatible API clients in unit tests.
- Run `ruff`, `ty`, and `pytest` before claiming implementation work is complete.
