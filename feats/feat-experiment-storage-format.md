# Normalized Experiment Storage Format

## Problem

Experiment-root artifacts currently mix JSON and JSONL even when a file contains only
one snapshot object. Files such as `personas.jsonl` and `base_personas.jsonl` are
complete Pydantic documents rather than streams of independent records, making the
extension misleading and the files less readable.

At the same time, changing every response ledger to JSON would weaken incremental
persistence. Questionnaire items and behavioral-task trials are committed one at a
time, and task resumption depends on replaying successful persisted transitions.

Some legacy response shapes also embed the full persona in each record, unnecessarily
increasing file size and duplicating experiment-level data.

## Proposed Feature

Normalize storage according to artifact semantics:

- Store experiment-root manifests and snapshots as readable, validated JSON documents.
- Store run manifests and immutable procedure snapshots as JSON documents.
- Retain JSONL for append-only item, trial, conversation, and analysis-unit ledgers.
- Reference personas from response records by stable `subject_id`; resolve full persona
  details from the experiment-level persona snapshot when needed.
- Provide readable derived exports without replacing the canonical append-only ledgers.

## Format Decisions

Experiment-root files should be JSON:

- `design.json`
- `personas.json`
- `metadata.json`
- `protocol.json`
- `base_personas.json`
- `protocol_assignments.json`

Run-level snapshot files should also be JSON:

- `run.json`
- `scale.json`
- `task.json`
- `schedules/{subject_id}.json`

Append-only runtime ledgers should remain JSONL:

- `responses/{subject_id}.jsonl`
- `conversations/{subject_id}.jsonl`
- analysis files whose lines are independent subject, item, trial, or scale units

`metadata.json` should be a validated experiment metadata document or run index updated
atomically, not a renamed append log. `protocol_assignments.json` should contain a
validated collection with assignments addressable by stable subject ID.

## Why Responses Remain JSONL

JSONL allows each successful item or trial to be flushed immediately without rewriting
the subject's complete response history. It supports partial-run inspection, bounded
memory use, deterministic replay, and recovery after interruption. A malformed final
line can be isolated more easily than a partially rewritten JSON array.

For human inspection and downstream tools, export commands may produce pretty JSON,
CSV, or other tabular results from the validated JSONL source.

## Data Normalization

- Canonical response records store `subject_id` and required protocol assignment IDs,
  not a full copied persona.
- The complete persona exists once in `personas.json`; protocol base personas exist once
  in `base_personas.json`.
- Provider, model, and procedure snapshots live at run level unless a per-call value
  genuinely differs.
- Item or trial records retain prompts, outputs, status, attempts, transition data, and
  metadata required to interpret that individual analysis unit.
- Loaders join persona or experiment metadata explicitly when producing analysis
  tables.

## Migration and Compatibility

- New runs write only the normalized paths.
- Loaders should recognize legacy `.jsonl` snapshot paths during a documented
  compatibility period.
- Legacy snapshot files that contain one JSON object can be migrated without changing
  their validated payload.
- Migration must not rewrite append-only response ledgers or alter persisted task
  transitions.
- An experiment must not contain both legacy and normalized canonical files without a
  clear precedence rule and validation error for conflicting contents.

## In Scope

- Experiment and run path definitions.
- JSON and JSONL read/write helpers with Pydantic validation.
- Persona-reference normalization in current response models.
- Compatibility loading for existing experiment directories.
- Documentation of canonical and derived artifact formats.
- Readable response export through existing or extended results commands.

## Out of Scope

- Replacing JSONL with a database.
- Removing raw provider responses, prompts, logprobs, or task attempts.
- Compressing or encrypting experiment artifacts.
- Duplicating full questionnaire or task definitions in response rows.
- Rewriting historical experiments in place without an explicit migration command.

## Acceptance Criteria

- All canonical files directly under an experiment root use `.json` and validate as
  complete Pydantic documents.
- Single-object run manifests use `run.json`, not `run.jsonl`.
- New response records contain a stable subject reference and do not duplicate the full
  persona.
- A subject's successful item or trial is durable immediately after its JSONL line is
  persisted.
- Behavioral-task replay reproduces every persisted transition after the format
  changes.
- Analysis and scoring can join response rows to persona and protocol data by stable
  IDs.
- Existing supported legacy experiments remain readable or fail with a precise
  migration message.
- Tests cover atomic JSON writes, JSONL append behavior, legacy loading, metadata joins,
  interrupted writes, and path conflicts.

## Dependencies and Risks

- Atomically updating `metadata.json` requires write-to-temporary-file and replace
  semantics.
- Concurrent subjects must not contend on a shared JSON document after every item or
  trial.
- Removing duplicated fields requires auditing all analysis consumers before changing
  response schemas.
- Compatibility behavior should be time-bounded so legacy path handling does not become
  permanent complexity.
