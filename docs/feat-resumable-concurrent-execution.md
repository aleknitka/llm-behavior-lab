# Resumable Concurrent Experiment Execution

## Problem

Batch and protocol experiments can generate thousands of provider calls, but current
execution is primarily sequential. A process interruption or provider outage can leave
an incomplete run that must be inspected and restarted manually, wasting time and
potentially repeating paid requests.

## Proposed Feature

Add bounded asynchronous batch execution with durable progress detection, configurable
retry behavior, and safe resumption of incomplete runs. Preserve synchronous entry
points for local providers and environments where concurrency is undesirable.

## User and Research Value

- Reduces wall-clock time for large experiments.
- Avoids repeating successfully completed calls after interruption.
- Handles transient rate limits and provider failures more predictably.
- Makes long-running local and hosted experiments operationally manageable.

## Core Capabilities

- Configure maximum concurrent requests at the batch runner boundary.
- Resume an explicitly selected run by reading and validating existing records.
- Skip completed subject-item units and retry failed or invalid units by policy.
- Use bounded retries with exponential backoff and optional provider retry hints.
- Append records safely without producing duplicate analysis units.
- Persist progress and final status for completed, partial, cancelled, and failed runs.
- Keep a synchronous execution mode with equivalent persistence semantics.

## In Scope

- Questionnaire batches and paired-factorial protocol runs.
- Concurrency across independent subjects or subject-item calls.
- Idempotent completion keys based on run, subject, questionnaire, and item IDs.
- Graceful cancellation and restart.
- Configurable concurrency, retry count, and retryable error classification.

## Out of Scope

- Distributed execution across multiple machines.
- A general-purpose job queue or workflow orchestration service.
- Automatic provider billing controls.
- Resuming into a run whose questionnaire, persona, protocol, or provider identity has
  changed incompatibly.

## Acceptance Criteria

- A concurrency setting greater than one executes independent calls concurrently while
  never exceeding the configured bound.
- Restarting an interrupted run does not repeat completed subject-item calls.
- Duplicate records are not written when the same resume command is run repeatedly.
- Transient failures are retried; permanent validation failures are recorded without
  uncontrolled retries.
- Cancellation leaves valid JSONL files and a run status that indicates partial work.
- Sync and async modes produce equivalent validated response records.
- Tests use fake clients to verify concurrency bounds, retries, cancellation, and
  idempotent resumption without live model calls.

## Dependencies and Risks

- Concurrent JSONL writes require serialization or per-subject ownership.
- Some local providers perform worse under concurrency, so conservative defaults matter.
- Provider seed support may not guarantee reproducibility under parallel scheduling.
- Resume compatibility checks must prevent mixing results from different experiment
  definitions.

