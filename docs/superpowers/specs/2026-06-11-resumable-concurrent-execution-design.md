# Resumable Concurrent Experiment Execution Design

## Objective

Add bounded concurrent questionnaire execution across subjects, durable
item-level resumption, explicit retry policy, and graceful cancellation without
changing within-subject questionnaire semantics or canonical JSONL ledgers.

Behavioral-task execution already supports bounded subject concurrency and
transition replay. This work extends equivalent operational capabilities to
questionnaire batches and questionnaire protocol steps while preserving the
existing synchronous entry points.

## Scope

### In scope

- Questionnaire batch and factorial-protocol subject concurrency.
- Questionnaire steps inside unified protocols.
- Sequential item execution and retained conversation history within each subject.
- Explicit resumption of an existing run by `run_id`.
- Idempotent item eligibility based on persisted response ledgers.
- Explicit retry of latest failed or invalid item records.
- Application-managed provider retries with configurable backoff.
- Graceful cancellation with resumable persisted progress.
- `partial` and `cancelled` run statuses.
- Equivalent persistence semantics for synchronous and asynchronous entry points.

### Out of scope

- Concurrent items within one subject.
- Distributed or multi-process execution.
- Automatic selection of an incomplete run when `run_id` is omitted.
- Rewriting or deleting prior failed or invalid response records.
- A general workflow engine shared by every procedure type.
- Provider billing controls.

## Architecture

### Subject-level coordinator

Introduce a bounded asynchronous questionnaire batch coordinator. It schedules
subjects concurrently up to the provider-configured limit. Each subject owns one
response ledger and executes questionnaire items sequentially.

The coordinator accepts an async client factory rather than sharing an
implicitly concurrency-safe client. A factory may return one client per subject
or another explicitly safe implementation.

The existing synchronous batch API remains available. It uses the same resume,
retry, history reconstruction, compatibility, and status helpers while
processing subjects one at a time.

### Shared subject runner

Refactor questionnaire execution around shared item eligibility and persistence
logic rather than maintaining independent sync and async behavior.

The sync and async subject runners differ only at the provider-call boundary:

- sync calls `SyncLlmClient.complete`;
- async calls `AsyncLlmClient.complete`.

Both runners:

1. Load and validate existing subject records when resuming.
2. Resolve the latest record for each item.
3. Reconstruct conversation history from completed records in questionnaire order.
4. Execute eligible items sequentially.
5. Append each new response immediately.
6. Return the latest effective records and final conversation history.

No concurrent task writes to the same subject ledger.

## Provider Execution Configuration

Provider configuration gains validated, non-secret execution fields:

- `max_attempts`: total physical attempts for one logical provider call.
- `initial_backoff_seconds`: delay before the first retry.
- `max_backoff_seconds`: upper bound for exponential backoff.
- `max_concurrency`: maximum concurrently executing subjects.

Defaults live in `config.py` and flow through staged designs, unified protocols,
runtime model settings, and persisted provider snapshots.

Both `OpenAI` and `AsyncOpenAI` clients are constructed with SDK
`max_retries=0`. The application retry layer is authoritative so attempts,
delays, fallback behavior, and terminal errors are consistent and testable.

## Retry Policy

The client boundary classifies provider exceptions.

Retryable failures:

- transport and connection errors;
- request timeouts;
- HTTP `408`, `409`, and `429`;
- HTTP `5xx`.

Non-retryable failures:

- other HTTP `4xx`;
- response parsing and answer validation failures;
- application validation errors.

When present, a valid provider `Retry-After` value determines the delay.
Otherwise delay is:

```text
min(initial_backoff_seconds * 2 ** retry_index, max_backoff_seconds)
```

Capability fallback, such as retrying without rejected logprobs, remains a
separate physical attempt and must still respect the overall attempt bound.

The client returns one logical result or raises the terminal exception. The
questionnaire runner persists one terminal item record, not one record per
transport attempt. Detailed call-attempt tracing remains separate proposed work.

## Resume Model

### Explicit run selection

Normal execution without `run_id` always creates a new run. Supplying an
existing `run_id` opts into resumption. The runner never automatically selects
the latest incomplete run.

### Compatibility validation

Before provider calls, resumption validates:

- experiment and run IDs;
- questionnaire ID and version from `scale.json`;
- provider snapshot and execution settings from `run.json`;
- expected subject IDs and persona cohort;
- protocol fingerprint, cohort ID, and step ID where applicable.

Incompatible persisted state fails before appending records.

### Item eligibility

For each `(subject_id, item_id)`, the latest persisted record determines
eligibility:

- `completed`: skip;
- `failed` or `invalid`: skip unless `retry_failed=True`;
- absent: execute;
- duplicate prior records: preserve all, but use the latest record as effective state.

A retry appends a new record. It never modifies or removes the prior attempt.
Repeated resume with no eligible items performs no provider calls and writes no
duplicate response records.

### History reconstruction

Conversation history is rebuilt in questionnaire order from the latest completed
record for each preceding item. Failed and invalid records do not contribute an
assistant turn.

For unified protocol steps with `history: inherit`, reconstructed questionnaire
history begins from the inherited prior-step history. The aggregate protocol
conversation remains consistent regardless of subject scheduling order.

## Cancellation

Cancellation is cooperative at the subject coordinator:

1. Stop scheduling subjects that have not started.
2. Allow in-flight subject provider calls to finish.
3. Persist their resulting item records and conversations.
4. Write a run record with `cancelled` status.
5. Preserve all ledgers for explicit resume.

An abrupt process termination may occur before the final run manifest update.
Resume therefore derives item progress from response ledgers rather than relying
only on `run.json`.

## Run Status

Extend `ResponseStatus` with:

- `partial`: execution stopped with unfinished expected units;
- `cancelled`: cancellation was requested and persisted.

Questionnaire batch status is derived from the latest effective item records:

- `completed`: every expected item has a completed latest record;
- `partial`: expected items are absent and cancellation was not persisted;
- `cancelled`: cancellation was requested;
- `failed`: all expected items have terminal records and one or more latest records failed;
- `invalid`: all expected items have terminal records, none failed, and one or more are invalid.

Protocol status combines step statuses using the existing severity ordering,
updated to include `partial` and `cancelled`.

## Interfaces

### CLI

Questionnaire and protocol execution expose:

- `--run-id` for explicit resume;
- `--retry-failed` for latest failed or invalid items.

Provider execution settings are configured when creating staged designs or
unified protocols and persisted as protocol identity:

- `--max-attempts`;
- `--initial-backoff-seconds`;
- `--max-backoff-seconds`;
- `--max-concurrency`.

Changing these settings for an immutable unified protocol requires a new
experiment ID. Resuming an existing run with different settings is rejected.

### Python

Provide async and sync persisted questionnaire batch entry points with matching
arguments and result models. Unified protocol execution may use an async
internal implementation with a synchronous wrapper for the CLI.

Behavioral-task execution retains its current subject concurrency and transition
resume behavior, but uses the shared provider retry configuration.

## Persistence

- Item and task records remain append-only JSONL.
- One subject owns each response file, avoiding cross-task write contention.
- `run.json` and `metadata.json` remain atomic JSON documents.
- Persisted provider snapshots include retry and concurrency settings.
- Prior response attempts remain available for audit and analysis.
- Analysis and scoring resolve the latest record for each subject/item unit.

## Testing

Use fake sync and async clients; no live provider calls.

Required coverage:

- concurrent subjects never exceed `max_concurrency`;
- items remain sequential within a subject;
- completed items are not repeated on resume;
- failed and invalid items require `retry_failed=True`;
- retries append a new response while preserving prior records;
- repeated resume is idempotent;
- transport, timeout, `408`, `409`, `429`, and `5xx` failures retry;
- other `4xx` and validation failures do not retry;
- `Retry-After` and exponential backoff obey configured limits;
- SDK clients are initialized with `max_retries=0`;
- cancellation stops new subject scheduling and persists `cancelled`;
- abrupt partial ledgers resume without requiring a final run manifest;
- incompatible questionnaire, provider, persona, protocol, or step state is rejected;
- synchronous and asynchronous execution produce equivalent response records;
- mixed protocols preserve reset and inherited histories;
- task replay remains deterministic with shared retry settings.

## Documentation

Update `README.md`, `docs/experiment-workflow.md`, CLI help, and the Ollama
factorial example to explain:

- subject-level concurrency;
- explicit `run_id` resumption;
- retry configuration;
- immutable prior attempt records;
- `partial` and `cancelled` statuses;
- local-provider guidance to keep `max_concurrency` conservative.
