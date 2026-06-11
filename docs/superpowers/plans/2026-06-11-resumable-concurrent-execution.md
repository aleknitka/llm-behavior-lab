# Resumable Concurrent Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add explicit questionnaire-run resumption, bounded cross-subject concurrency, application-owned provider retries, cooperative cancellation, and retry-safe analysis without changing sequential item behavior within a subject.

**Architecture:** Persist retry and concurrency policy with provider configuration and copy it into every runtime provider snapshot. Keep append-only response ledgers canonical, introduce one shared latest-attempt ledger view, and use that view for eligibility, history reconstruction, run status, scoring, and export. Add an async subject coordinator that limits concurrent subjects while each subject continues to execute questionnaire items sequentially.

**Tech Stack:** Python 3.12, asyncio, OpenAI Python client, Pydantic 2, JSONL, pytest, ruff, ty, uv

---

### Task 1: Persist execution policy and terminal states

**Files:**
- Modify: `src/llm_behavior_lab/config.py`
- Modify: `src/llm_behavior_lab/models.py`
- Modify: `src/llm_behavior_lab/experiments.py`
- Modify: `src/llm_behavior_lab/protocols.py`
- Modify: `src/llm_behavior_lab/responses/base/session.py`
- Test: `tests/llm_behavior_lab/test_models.py`
- Test: `tests/llm_behavior_lab/test_experiments.py`
- Test: `tests/llm_behavior_lab/test_unified_protocol.py`
- Test: `tests/llm_behavior_lab/responses/test_base_models.py`

- [ ] **Step 1: Write failing validation and serialization tests**

```python
def test_model_settings_validate_execution_policy() -> None:
    settings = ModelSettings(
        model="test",
        provider_base_url="http://localhost",
        temperature=0,
        timeout_seconds=10,
        max_attempts=4,
        initial_backoff_seconds=0.25,
        max_backoff_seconds=4,
        max_concurrency=8,
    )
    assert settings.max_attempts == 4
    assert settings.max_concurrency == 8


def test_response_status_supports_partial_and_cancelled() -> None:
    assert ResponseStatus("partial") is ResponseStatus.PARTIAL
    assert ResponseStatus("cancelled") is ResponseStatus.CANCELLED
```

Also assert that `ProviderDesign`, `ProtocolProvider`, and `ProviderSnapshot` round-trip the four fields and reject zero attempts, negative backoff, and zero concurrency.

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `uv run pytest tests/llm_behavior_lab/test_models.py tests/llm_behavior_lab/test_experiments.py tests/llm_behavior_lab/test_unified_protocol.py tests/llm_behavior_lab/responses/test_base_models.py -q`

Expected: FAIL because the execution-policy fields and statuses do not exist.

- [ ] **Step 3: Add constants and validated fields**

```python
# config.py
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_INITIAL_BACKOFF_SECONDS = 1.0
DEFAULT_MAX_BACKOFF_SECONDS = 30.0
DEFAULT_MAX_CONCURRENCY = 4
```

Add these fields to `ModelSettings`, `ProviderDesign`, `ProtocolProvider`, and `ProviderSnapshot`:

```python
max_attempts: int = Field(default=DEFAULT_MAX_ATTEMPTS, ge=1)
initial_backoff_seconds: float = Field(
    default=DEFAULT_INITIAL_BACKOFF_SECONDS, ge=0
)
max_backoff_seconds: float = Field(default=DEFAULT_MAX_BACKOFF_SECONDS, ge=0)
max_concurrency: int = Field(default=DEFAULT_MAX_CONCURRENCY, ge=1)
```

Extend the enum:

```python
class ResponseStatus(StrEnum):
    COMPLETED = "completed"
    INVALID = "invalid"
    FAILED = "failed"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"
```

- [ ] **Step 4: Run tests and commit**

Run: `uv run pytest tests/llm_behavior_lab/test_models.py tests/llm_behavior_lab/test_experiments.py tests/llm_behavior_lab/test_unified_protocol.py tests/llm_behavior_lab/responses/test_base_models.py -q`

Expected: PASS.

```bash
git add src/llm_behavior_lab/config.py src/llm_behavior_lab/models.py src/llm_behavior_lab/experiments.py src/llm_behavior_lab/protocols.py src/llm_behavior_lab/responses/base/session.py tests/llm_behavior_lab
git commit -m "feat: persist provider execution policy"
```

### Task 2: Make application retries authoritative

**Files:**
- Modify: `src/llm_behavior_lab/client.py`
- Test: `tests/llm_behavior_lab/test_client.py`

- [ ] **Step 1: Write failing sync and async retry tests**

Use `httpx.Request`, `httpx.Response`, `openai.APIConnectionError`, `openai.APITimeoutError`, and `openai.APIStatusError` to verify:

```python
@pytest.mark.parametrize("status_code", [408, 409, 429, 500, 503])
def test_sync_client_retries_transient_statuses(status_code: int) -> None:
    completions = SequencedCompletions(
        [status_error(status_code, retry_after="0"), completion("answer-1")]
    )
    client = OpenAiChatClient(
        api_key="test",
        base_url="http://localhost",
        client=FakeOpenAI(completions),
        sleep=lambda _: None,
    )
    result = client.complete(MESSAGES, SETTINGS, ["answer-1"])
    assert result.selected_answer_id == "answer-1"
    assert completions.call_count == 2
```

Add matching async coverage, exhaustion at `max_attempts`, no retry for status 400, `Retry-After` precedence, capped exponential delays, and an assertion that default constructors call `OpenAI(..., max_retries=0)` and `AsyncOpenAI(..., max_retries=0)`.

- [ ] **Step 2: Run the client tests and verify failure**

Run: `uv run pytest tests/llm_behavior_lab/test_client.py -q`

Expected: FAIL because clients do not classify or retry transient failures.

- [ ] **Step 3: Implement shared retry classification and delay calculation**

```python
_RETRYABLE_STATUS_CODES = {408, 409, 429}


def _is_retryable_error(exc: Exception) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return True
    return isinstance(exc, APIStatusError) and (
        exc.status_code in _RETRYABLE_STATUS_CODES or exc.status_code >= 500
    )


def _retry_delay(exc: Exception, settings: ModelSettings, attempt: int) -> float:
    retry_after = _retry_after_seconds(exc)
    if retry_after is not None:
        return min(retry_after, settings.max_backoff_seconds)
    exponential = settings.initial_backoff_seconds * (2 ** (attempt - 1))
    return min(exponential, settings.max_backoff_seconds)
```

Accept injectable `client` and sleep functions for tests, construct SDK clients with `max_retries=0`, and wrap the physical request attempts in bounded sync and async loops. A logprobs capability fallback consumes one physical attempt and remains inside the same `max_attempts` bound.

- [ ] **Step 4: Run tests and commit**

Run: `uv run pytest tests/llm_behavior_lab/test_client.py -q`

Expected: PASS.

```bash
git add src/llm_behavior_lab/client.py tests/llm_behavior_lab/test_client.py
git commit -m "feat: add bounded provider retries"
```

### Task 3: Add the canonical latest-attempt ledger view

**Files:**
- Create: `src/llm_behavior_lab/responses/item_ledgers.py`
- Modify: `src/llm_behavior_lab/responses/__init__.py`
- Test: `tests/llm_behavior_lab/responses/test_item_ledgers.py`

- [ ] **Step 1: Write failing ledger tests**

```python
def test_latest_item_attempts_replace_prior_attempt_without_reordering() -> None:
    first = response("subject-1", "item-1", ResponseStatus.FAILED, item_order=1)
    second = response("subject-1", "item-2", ResponseStatus.COMPLETED, item_order=2)
    retried = response("subject-1", "item-1", ResponseStatus.COMPLETED, item_order=1)

    assert latest_item_attempts([first, second, retried]) == [retried, second]


def test_pending_item_ids_only_retry_failed_when_requested() -> None:
    records = [
        response("subject-1", "item-1", ResponseStatus.COMPLETED, item_order=1),
        response("subject-1", "item-2", ResponseStatus.FAILED, item_order=2),
    ]
    assert pending_item_ids(QUESTIONNAIRE, records, retry_failed=False) == ["item-3"]
    assert pending_item_ids(QUESTIONNAIRE, records, retry_failed=True) == [
        "item-2",
        "item-3",
    ]
```

Cover `INVALID` like `FAILED`, preserve append-only input, reject records whose questionnaire identity or item text/order differs from the current snapshot, and load a missing subject file as an empty ledger.

- [ ] **Step 2: Run the test and verify failure**

Run: `uv run pytest tests/llm_behavior_lab/responses/test_item_ledgers.py -q`

Expected: FAIL because `item_ledgers` does not exist.

- [ ] **Step 3: Implement focused ledger helpers**

```python
def latest_item_attempts(
    records: Sequence[ItemResponseRecord],
) -> list[ItemResponseRecord]:
    latest = {(record.subject_id, record.item_id): record for record in records}
    return sorted(latest.values(), key=lambda record: (record.subject_id, record.item_order))


def pending_item_ids(
    questionnaire: Questionnaire,
    records: Sequence[ItemResponseRecord],
    *,
    retry_failed: bool,
) -> list[str]:
    latest = {record.item_id: record for record in latest_item_attempts(records)}
    retryable = {ResponseStatus.FAILED, ResponseStatus.INVALID}
    return [
        item.id
        for item in questionnaire.items
        if item.id not in latest
        or (retry_failed and latest[item.id].status in retryable)
    ]
```

Add `load_item_ledger(path)` and `validate_item_ledger(questionnaire, subject_id, run_id, records)`. Validation must compare questionnaire ID/version plus every persisted item ID, order, text, and response-format type.

- [ ] **Step 4: Run tests and commit**

Run: `uv run pytest tests/llm_behavior_lab/responses/test_item_ledgers.py -q`

Expected: PASS.

```bash
git add src/llm_behavior_lab/responses tests/llm_behavior_lab/responses/test_item_ledgers.py
git commit -m "feat: add latest-attempt response ledger view"
```

### Task 4: Resume one questionnaire subject safely

**Files:**
- Modify: `src/llm_behavior_lab/runner.py`
- Test: `tests/llm_behavior_lab/test_runner.py`

- [ ] **Step 1: Write failing subject-resume tests**

Add tests that prewrite item 1, call `run_questionnaire(..., run_id=existing_run_id)`, and assert only items 2 onward reach the client. Add explicit `retry_failed=True` coverage proving a failed item is appended as a new attempt, and default coverage proving failed/invalid attempts are not retried.

```python
assert client.requested_item_ids == ["item-2", "item-3"]
assert [record.item_id for record in load_item_ledger(response_path)] == [
    "item-1",
    "item-2",
    "item-3",
]
```

Also assert reconstructed history contains the completed persisted exchange before the next question and does not contain failed/invalid exchanges.

- [ ] **Step 2: Run focused tests and verify failure**

Run: `uv run pytest tests/llm_behavior_lab/test_runner.py -k "resume or retry_failed or history" -q`

Expected: FAIL because the runner asks every item again.

- [ ] **Step 3: Refactor subject preparation and add resume parameters**

Add `retry_failed: bool = False` and `run_root_override` parity to both sync and async signatures. Before the loop:

```python
response_path = paths.response_path_for_subject(persona.persona_id)
persisted = load_item_ledger(response_path)
if persisted:
    validate_item_ledger(
        questionnaire, persona.persona_id, resolved_run_id, persisted
    )
pending = set(
    pending_item_ids(questionnaire, persisted, retry_failed=retry_failed)
)
effective = latest_item_attempts(persisted)
history = _history_after_questionnaire(
    persona, context, initial_history, effective
)
```

Loop over questionnaire order, skip items outside `pending`, append only new attempts, and return the effective latest-attempt records after reloading the ledger. Preserve sequential history updates inside the subject.

- [ ] **Step 4: Run runner tests and commit**

Run: `uv run pytest tests/llm_behavior_lab/test_runner.py -q`

Expected: PASS.

```bash
git add src/llm_behavior_lab/runner.py tests/llm_behavior_lab/test_runner.py
git commit -m "feat: resume questionnaire subjects from ledgers"
```

### Task 5: Compute partial and cancelled run status from effective records

**Files:**
- Modify: `src/llm_behavior_lab/runner.py`
- Test: `tests/llm_behavior_lab/test_runner.py`
- Test: `tests/llm_behavior_lab/test_batch_runner.py`

- [ ] **Step 1: Write failing status tests**

```python
def test_run_status_is_partial_when_expected_items_are_missing() -> None:
    assert _response_status(
        records=[completed("item-1")],
        expected_item_count=3,
        cancelled=False,
    ) is ResponseStatus.PARTIAL


def test_run_status_is_cancelled_when_cancellation_was_requested() -> None:
    assert _response_status(
        [], expected_item_count=3, cancelled=True
    ) is ResponseStatus.CANCELLED
```

Add precedence tests: cancelled first, then partial, then failed, invalid, completed. Assert `item_count` and `error_count` use latest attempts rather than all appended attempts.

- [ ] **Step 2: Run focused tests and verify failure**

Run: `uv run pytest tests/llm_behavior_lab/test_runner.py tests/llm_behavior_lab/test_batch_runner.py -k "status or item_count or error_count" -q`

Expected: FAIL because `_response_status` cannot represent missing items or cancellation.

- [ ] **Step 3: Update status and run-record construction**

```python
def _response_status(
    records: Sequence[ItemResponseRecord],
    *,
    expected_item_count: int,
    cancelled: bool = False,
) -> ResponseStatus:
    if cancelled:
        return ResponseStatus.CANCELLED
    if len(records) < expected_item_count:
        return ResponseStatus.PARTIAL
    if any(record.status == ResponseStatus.FAILED for record in records):
        return ResponseStatus.FAILED
    if any(record.status == ResponseStatus.INVALID for record in records):
        return ResponseStatus.INVALID
    return ResponseStatus.COMPLETED
```

Pass `len(questionnaire.items) * persona_count` for batch run records. Include all four execution-policy values in `_provider_snapshot`.

- [ ] **Step 4: Run tests and commit**

Run: `uv run pytest tests/llm_behavior_lab/test_runner.py tests/llm_behavior_lab/test_batch_runner.py -q`

Expected: PASS.

```bash
git add src/llm_behavior_lab/runner.py tests/llm_behavior_lab/test_runner.py tests/llm_behavior_lab/test_batch_runner.py
git commit -m "feat: persist partial questionnaire run status"
```

### Task 6: Add bounded async execution across subjects

**Files:**
- Modify: `src/llm_behavior_lab/runner.py`
- Modify: `src/llm_behavior_lab/__init__.py`
- Test: `tests/llm_behavior_lab/test_batch_runner.py`

- [ ] **Step 1: Write failing concurrency and cancellation tests**

Use an async fake client that tracks active calls. Run three two-item subjects with `max_concurrency=2` and assert:

```python
assert client_state.max_active == 2
assert client_state.per_subject_items == {
    subject_1: ["item-1", "item-2"],
    subject_2: ["item-1", "item-2"],
    subject_3: ["item-1", "item-2"],
}
```

Set an `asyncio.Event` while two subjects are active; assert waiting subjects make no provider calls, active subjects finish, ledgers are valid, and the batch status is `CANCELLED`.

- [ ] **Step 2: Run focused tests and verify failure**

Run: `uv run pytest tests/llm_behavior_lab/test_batch_runner.py -k "async or concurrency or cancellation" -q`

Expected: FAIL because no async persisted-persona coordinator exists.

- [ ] **Step 3: Implement the coordinator**

```python
async def run_persisted_persona_batch_async(
    *,
    personas: PersonaBatch,
    questionnaire: Questionnaire,
    settings: ModelSettings,
    client_factory: Callable[[], AsyncLlmClient],
    project_root: Path,
    context: str | None = None,
    response_metadata_by_subject: dict[str, dict[str, object]] | None = None,
    initial_histories: dict[str, list[Message]] | None = None,
    run_id: str | None = None,
    run_root_override: Path | None = None,
    retry_failed: bool = False,
    cancel_event: asyncio.Event | None = None,
) -> BatchRunResult:
    semaphore = asyncio.Semaphore(settings.max_concurrency)
```

Create one coroutine per subject. Each coroutine waits on the semaphore, checks `cancel_event` before starting, then calls `run_questionnaire_async` with a client from `client_factory`. Await all coroutines without cancelling in-flight subjects. Aggregate effective ledgers only after all started subjects finish, write one final `run.json`, and export the function from the package.

- [ ] **Step 4: Run tests and commit**

Run: `uv run pytest tests/llm_behavior_lab/test_batch_runner.py tests/llm_behavior_lab/test_package.py -q`

Expected: PASS.

```bash
git add src/llm_behavior_lab/runner.py src/llm_behavior_lab/__init__.py tests/llm_behavior_lab/test_batch_runner.py tests/llm_behavior_lab/test_package.py
git commit -m "feat: run questionnaire subjects concurrently"
```

### Task 7: Validate resume identity and preserve one run index entry

**Files:**
- Modify: `src/llm_behavior_lab/runner.py`
- Modify: `src/llm_behavior_lab/storage.py`
- Test: `tests/llm_behavior_lab/test_batch_runner.py`
- Test: `tests/llm_behavior_lab/test_storage.py`

- [ ] **Step 1: Write failing compatibility tests**

Create an existing run and assert resume fails before provider calls when model, base URL, questionnaire snapshot, persona IDs, run ID, or execution-policy snapshot differs. Add a crash fixture with response JSONL and `scale.json` but no `run.json`; matching resume must continue successfully. Resume a completed run and assert `metadata.jsonl` still contains one logical entry for that run ID.

- [ ] **Step 2: Run focused tests and verify failure**

Run: `uv run pytest tests/llm_behavior_lab/test_batch_runner.py tests/llm_behavior_lab/test_storage.py -k "resume or metadata" -q`

Expected: FAIL because compatibility is not checked and metadata appends duplicates.

- [ ] **Step 3: Implement compatibility and run-index replacement**

Add `validate_questionnaire_resume(...)` that reads `scale.json`, optional `run.json`, and subject ledgers. Exact-match:

```python
expected_provider = _provider_snapshot(settings)
if existing.provider != expected_provider:
    raise ValueError("run provider configuration does not match resume request")
if existing.subject_ids != expected_subject_ids:
    raise ValueError("run persona cohort does not match resume request")
```

When `run.json` is absent, validate from `scale.json`, response records, the persisted experiment design/protocol, and supplied personas. Change `update_experiment_metadata` to replace the existing entry with the same `run_id` instead of appending another entry, preserving the original `started_at` from `run.json` when present.

- [ ] **Step 4: Run tests and commit**

Run: `uv run pytest tests/llm_behavior_lab/test_batch_runner.py tests/llm_behavior_lab/test_storage.py -q`

Expected: PASS.

```bash
git add src/llm_behavior_lab/runner.py src/llm_behavior_lab/storage.py tests/llm_behavior_lab/test_batch_runner.py tests/llm_behavior_lab/test_storage.py
git commit -m "feat: validate questionnaire run resumption"
```

### Task 8: Wire execution policy and resume controls into the CLI

**Files:**
- Modify: `src/llm_behavior_lab/main.py`
- Test: `tests/llm_behavior_lab/test_main.py`

- [ ] **Step 1: Write failing parser and dispatch tests**

Assert both design commands accept:

```text
--max-attempts 5
--initial-backoff 0.5
--max-backoff 8
--max-concurrency 6
```

Assert `scale-run` accepts `--run-id` and `--retry-failed`, and `task-run` uses persisted `max_concurrency` instead of a separate `--concurrency` flag. Mock dispatch and verify `scale-run` creates `AsyncOpenAiChatClient` instances and calls `run_persisted_persona_batch_async`. Add a signal-bridge test that invokes the registered SIGINT callback, sets the shared cancellation event, and does not cancel the running coroutine.

- [ ] **Step 2: Run CLI tests and verify failure**

Run: `uv run pytest tests/llm_behavior_lab/test_main.py -q`

Expected: FAIL because the parser and async scale dispatch are missing.

- [ ] **Step 3: Add shared provider arguments and async dispatch**

```python
def _provider_design_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_API_BASE_URL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument(
        "--initial-backoff",
        type=float,
        default=DEFAULT_INITIAL_BACKOFF_SECONDS,
    )
    parser.add_argument("--max-backoff", type=float, default=DEFAULT_MAX_BACKOFF_SECONDS)
    parser.add_argument("--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY)
```

Use this helper for scale/task design. Add `--run-id` and `--retry-failed` to scale-run. Remove task-run's redundant `--resume` and `--concurrency`; pass `settings.max_concurrency` into the existing task batch API.

Run async batches through a cooperative signal bridge:

```python
async def _run_with_cancellation(
    operation: Callable[[asyncio.Event], Awaitable[BatchRunResult]],
) -> BatchRunResult:
    cancel_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    installed = False
    try:
        loop.add_signal_handler(signal.SIGINT, cancel_event.set)
        installed = True
    except (NotImplementedError, RuntimeError):
        pass
    try:
        return await operation(cancel_event)
    finally:
        if installed:
            loop.remove_signal_handler(signal.SIGINT)
```

Pass the event to `run_persisted_persona_batch_async`. The first Ctrl+C stops new subjects and lets active subjects finish; ordinary exceptions still propagate.

- [ ] **Step 4: Run tests and commit**

Run: `uv run pytest tests/llm_behavior_lab/test_main.py -q`

Expected: PASS.

```bash
git add src/llm_behavior_lab/main.py tests/llm_behavior_lab/test_main.py
git commit -m "feat: expose resumable concurrent scale runs"
```

### Task 9: Resume questionnaire steps inside unified protocols

**Files:**
- Modify: `src/llm_behavior_lab/protocol_runs.py`
- Modify: `src/llm_behavior_lab/main.py`
- Test: `tests/llm_behavior_lab/test_unified_protocol.py`
- Test: `tests/llm_behavior_lab/test_main.py`

- [ ] **Step 1: Write failing protocol-resume tests**

Create a two-step protocol, interrupt during the questionnaire step, and resume with the same explicit protocol run ID. Assert completed questionnaire items are not called twice, inherited history is rebuilt from effective records, completed task steps remain skipped, and a mismatched cohort/provider/protocol fingerprint fails before any call.

Add parser coverage for:

```text
protocol-create --file protocol.json --run-id run-protocol-test-... --retry-failed
```

and reject combining `--run-id` with `--new-run`, `--cohort-id`, or `--persona-seed`.

- [ ] **Step 2: Run focused tests and verify failure**

Run: `uv run pytest tests/llm_behavior_lab/test_unified_protocol.py tests/llm_behavior_lab/test_main.py -k "protocol and resume" -q`

Expected: FAIL because protocol runs always allocate a new run ID.

- [ ] **Step 3: Add explicit protocol-run resumption**

Extend:

```python
def create_protocol_run(
    project_root: Path,
    protocol: UnifiedExperimentProtocol,
    *,
    run_id: str | None = None,
    retry_failed: bool = False,
    ...
) -> ProtocolRunResult:
```

For new runs, write an initial `PARTIAL` top-level `run.json` before provider calls. For resume, require the run directory, load its top-level record, validate protocol fingerprint, cohort ID, effective seeds, provider snapshot, and ordered step IDs, then execute only incomplete steps. Use `run_persisted_persona_batch_async` for questionnaire steps and the existing task resumption API for task steps. Rebuild aggregate conversations after each step and replace the one protocol run-index entry on completion or cancellation.

- [ ] **Step 4: Run tests and commit**

Run: `uv run pytest tests/llm_behavior_lab/test_unified_protocol.py tests/llm_behavior_lab/test_main.py -q`

Expected: PASS.

```bash
git add src/llm_behavior_lab/protocol_runs.py src/llm_behavior_lab/main.py tests/llm_behavior_lab/test_unified_protocol.py tests/llm_behavior_lab/test_main.py
git commit -m "feat: resume unified protocol runs"
```

### Task 10: Make scoring, export, examples, and feature tracking retry-safe

**Files:**
- Modify: `src/llm_behavior_lab/analysis.py`
- Modify: `src/llm_behavior_lab/scoring/run.py`
- Modify: `README.md`
- Modify: `examples/README.md`
- Modify: `examples/ollama-bfi10-factorial/README.md`
- Modify: `examples/ollama-bfi10-factorial/notebook.py`
- Modify: `examples/ollama-bfi10-factorial/protocol.json`
- Delete: `feats/feat-resumable-concurrent-execution.md`
- Test: `tests/llm_behavior_lab/test_analysis.py`
- Test: `tests/llm_behavior_lab/scoring/test_run.py`

- [ ] **Step 1: Write failing downstream-reader tests**

Append a failed item attempt followed by a completed retry. Assert `load_response_table` emits one row containing the completed retry and `score_run` scores one effective item rather than both ledger lines.

- [ ] **Step 2: Run focused tests and verify failure**

Run: `uv run pytest tests/llm_behavior_lab/test_analysis.py tests/llm_behavior_lab/scoring/test_run.py -q`

Expected: FAIL because readers consume every append-only attempt.

- [ ] **Step 3: Apply the latest-attempt view and update user-facing material**

Wrap both loaders:

```python
return latest_item_attempts(records)
```

Document persisted retry/concurrency settings, `scale-run --run-id ...`, explicit `--retry-failed`, cancellation status, and the rule that a missing `--run-id` always starts a new run. Update the Ollama example provider configuration with all four fields, recommend conservative local concurrency, and show a resumed async scale run. Delete the completed feature document only after all behavior and docs are present.

- [ ] **Step 4: Run the full quality gate**

Run:

```bash
uv run ruff check .
uv run ty check
uv run pytest
```

Expected: all commands exit 0.

- [ ] **Step 5: Commit**

```bash
git add src/llm_behavior_lab/analysis.py src/llm_behavior_lab/scoring/run.py README.md examples/README.md examples/ollama-bfi10-factorial tests/llm_behavior_lab/test_analysis.py tests/llm_behavior_lab/scoring/test_run.py
git rm feats/feat-resumable-concurrent-execution.md
git commit -m "docs: complete resumable concurrent execution"
```
