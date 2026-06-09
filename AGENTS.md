# Agent Guidance

## Project Purpose

This repository runs persona-based LLM questionnaire sessions and stateful
behavioral experiments.

The project should grow beyond a single questionnaire or fixed persona set. It is
intended to support multiple standardized questionnaires, behavioral tasks,
extensible persona feature maps, and experimental designs that compare how
selected persona parameters influence simulated responses and decisions.

The app should allow a user to:

1. Create a persona from demographic inputs selected from a map of available features.
2. Ask an OpenAI-compatible LLM to assume that persona.
3. Optionally expand generated personas through paired or factorial protocol designs
   that manipulate selected fields while keeping the remaining base persona stable.
4. Run a questionnaire or a multi-trial behavioral task while retaining the
   appropriate interaction history.
5. Save item or trial data in JSONL format for later analysis.

## Development Approach

- Put application code under `src/`.
- Before adding or changing code, inspect the current codebase and follow the patterns already present.
- Keep code simple, explicit, and easy to maintain.
- Do not create unnecessary abstractions, framework layers, services, registries, factories, or directory structures.
- Prefer small functions and straightforward data models over complex object hierarchies.
- Add structure only when it is clearly needed by existing behavior or tests.
- Make expansions easy to maintain: new persona fields should fit the existing
  persona models and factory flow, new questionnaires should fit the shared
  questionnaire base models, and repeated-choice tasks should fit the behavioral
  task protocol.

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
6. Load one questionnaire definition or one behavioral task definition.
7. Ask questionnaire items or repeatedly present task observations.
8. Retain prior context requested by the procedure.
9. For tasks, validate the selected action and apply the deterministic
   environment transition before revealing feedback.
10. Capture model output, structured output if available, logprobs if available,
    and relevant metadata.
11. Append item or trial records to JSONL for later analysis.

## Behavioral Task Format

Stateful tasks live under `src/llm_behavior_lab/behavioral_tasks/`.

- Task implementations expose instruction, initial state, observation, action
  application, completion, and summary behavior.
- The task environment owns schedules, rewards, losses, and state transitions.
  The LLM only selects one of the allowed model-facing actions.
- Resolve and persist hidden schedules before the first provider call.
- Keep internal action IDs, future outcomes, advantageous classifications, and
  recognized task names out of model-facing prompts when they could reveal the
  solution.
- Treat a persisted successful transition as the resume commit boundary.
- Resume an existing task only when its explicit run ID is supplied. Replaying
  completed records must reproduce every persisted transition exactly.
- Do not retry a failed or twice-invalid subject unless retry was explicitly
  requested; preserve the prior attempt records when retrying.
- Keep trials sequential within one subject; bounded concurrency may be used
  across subjects.
- Do not interpret provider latency as human reaction time.

## Questionnaire Format

Multiple questionnaires should be runnable through a standardized form.

New questionnaire source data should be stored as Python module-level constants under `src/llm_behavior_lab/questionnaires/` and built with the Pydantic models in `src/llm_behavior_lab/questionnaires/base/`.

Use `llm_behavior_lab.questionnaires.base.scale.Questionnaire` as the source-of-truth object for newly coded questionnaires. It stores:

- `id`, `name`, `version`, optional `language`, `reference`, and `licence`.
- `sections`, where each `Section` groups ordered `item_ids`.
- `items`, where each `Item` has a stable `id`, optional source `code`, display `order`, item `text`, `response_format`, and optional tags or metadata.
- `scales`, where each `Scale` names the measured construct and maps items through `ItemMapping`, including reverse scoring, weights, scoring roles, and item-level scoring metadata.
- `scoring_models`, where each `ScoringModel` defines versioned `ScaleScoringRule` entries using `sum`, `mean`, or `weighted_mean` transformations plus optional output ranges and interpretation bands.
- `metadata` for questionnaire-level details that do not deserve first-class fields yet.

Use `llm_behavior_lab.questionnaires.base.response_formats` for response data:

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

- Save questionnaire and behavioral-task run data as JSONL.
- Each line should represent a clear analysis unit: one answered questionnaire
  item or one task trial with its attempts and deterministic transition.
- Keep the JSONL schema stable and validate records with Pydantic before writing.
- Do not duplicate full questionnaire definitions into every result line. Store questionnaire definitions in code under `src/llm_behavior_lab/questionnaires/`; store result lines with stable questionnaire, item, scale, model, prompt, persona, provider, and response identifiers or metadata needed for later analysis.
- Scale runtime data is stored under
  `experiments/{experiment_id}/run-{questionnaire}-{model}-{timestamp}/`.
- Behavioral-task runtime data is stored under
  `experiments/{experiment_id}/run-task-{task}-{model}-{timestamp}/`.
- Experiment roots can also contain `personas.jsonl`, `metadata.jsonl`, and, for protocol runs, `protocol.json`, `base_personas.jsonl`, and `protocol_assignments.jsonl`.
- Experiment IDs must be exactly three lowercase letter-or-digit items separated by hyphens, such as `pilot-study-one`; reject spaces, underscores, dots, path separators, and other symbols.
- Session IDs must be `session-[uuid]`.
- Scale run directories contain `run.jsonl`, `scale.json`, and
  `responses/{subject_id}.jsonl`.
- Task run directories contain `run.jsonl`, `task.json`, subject schedules,
  compact conversations, and `responses/{subject_id}.jsonl`.

## Response and Session Data

Runtime response data should use the Pydantic models in `src/llm_behavior_lab/responses/base/`.

- Use `RunRecord` for generic procedure metadata: experiment and session IDs,
  run ID, subject IDs, procedure kind/ID/version, provider snapshot, timestamps,
  status, output paths, and metadata.
- Use `SessionRecord` when session-level metadata needs to be persisted separately.
- Use `ItemResponseRecord` for item-level JSONL analysis units: stable questionnaire and item IDs, item order/text, response format type, prompt messages, parsed answer, raw response, structured response, logprobs, status, errors, and metadata.
- Use typed answer values instead of ad hoc nullable fields: `LikertAnswerValue`, `NumericAnswerValue`, `SingleChoiceAnswerValue`, `MultipleChoiceAnswerValue`, and `TextAnswerValue`.
- Keep provider-specific or experiment-specific details inside explicit `metadata` fields unless they become stable first-class fields.
- Do not add scoring outputs to response records until scoring is implemented against questionnaire `scales` and `scoring_models`.
- Use `TaskTrialRecord` for task attempts, observations, transitions, message
  indexes, status, errors, and task metadata.

## Testing

- Add unit tests with `pytest` for data validation, questionnaire/task loading,
  prompt masking, deterministic transitions, resume behavior, provider
  capability handling, and JSONL serialization.
- Prefer tests that exercise behavior without requiring live model calls.
- Mock or fake OpenAI-compatible API clients in unit tests.
- Run `ruff`, `ty`, and `pytest` before claiming implementation work is complete.
