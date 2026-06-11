from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from llm_behavior_lab.behavioral_tasks.base import (
    TaskAttemptRecord,
    TaskRunResult,
    TaskState,
    TaskTransition,
    TaskTrialRecord,
)
from llm_behavior_lab.behavioral_tasks.iowa_gambling import (
    IowaGamblingTask,
    ResolvedIowaSchedule,
)
from llm_behavior_lab.client import Message, SyncLlmClient
from llm_behavior_lab.models import LlmQuestionResult, ModelSettings, Persona
from llm_behavior_lab.prompting import render_persona_intro
from llm_behavior_lab.responses.base import ChatMessage, ResponseStatus
from llm_behavior_lab.storage import (
    append_jsonl_record,
    normalize_prefixed_uuid,
    resolve_experiment_paths,
    validate_experiment_id,
    write_json_document,
)


def run_behavioral_task(
    *,
    persona: Persona,
    task: IowaGamblingTask,
    settings: ModelSettings,
    client: SyncLlmClient,
    project_root: Path,
    experiment_id: str,
    run_id: str,
    resolved_schedule: ResolvedIowaSchedule,
    session_id: str | None = None,
    resume: bool = False,
    retry_failed: bool = False,
    max_trials: int | None = None,
    response_metadata: dict[str, object] | None = None,
    initial_history: Sequence[Message] | None = None,
    run_root_override: Path | None = None,
) -> TaskRunResult:
    """Run or resume one persona through a stateful behavioral task.

    Successful transitions are the commit boundary. On resume, state and compact
    chat history are reconstructed by replaying those transitions in order.
    """
    experiment_id = validate_experiment_id(experiment_id)
    paths = resolve_experiment_paths(
        project_root,
        experiment_id,
        run_id,
        run_root_override=run_root_override,
    )
    response_path = paths.response_path_for_subject(persona.persona_id)
    conversation_path = (
        paths.run_root / "conversations" / f"{persona.persona_id}.jsonl"
    )
    schedule_path = paths.run_root / "schedules" / f"{persona.persona_id}.json"
    task_path = paths.run_root / "task.json"
    _write_once(task_path, task.config)
    _write_once(schedule_path, resolved_schedule)

    existing = _load_records(response_path) if resume else []
    session_id = (
        existing[0].session_id
        if existing and session_id is None
        else normalize_prefixed_uuid("session-", session_id)
    )
    if existing and not resume:
        raise FileExistsError(f"task response ledger already exists: {response_path}")
    if (
        existing
        and existing[-1].status in {ResponseStatus.FAILED, ResponseStatus.INVALID}
        and not retry_failed
    ):
        return TaskRunResult(records=existing, status=existing[-1].status)
    completed = [record for record in existing if record.status == ResponseStatus.COMPLETED]
    state, transitions = _replay(task, resolved_schedule, completed)
    history = _history(persona, task, transitions, initial_history)
    _write_conversation(conversation_path, history)

    records = list(existing)
    successful_this_call = 0
    while not task.is_complete(state):
        if max_trials is not None and successful_this_call >= max_trials:
            break
        observation = task.observe(state, resolved_schedule)
        messages = [*history, {"role": "user", "content": observation.prompt}]
        message_start = len(history)
        attempts: list[TaskAttemptRecord] = []
        selected: str | None = None
        result = LlmQuestionResult()

        for attempt_number in (1, 2):
            try:
                result = client.complete(
                    messages,
                    settings,
                    observation.allowed_action_ids,
                )
            except Exception as exc:
                result = LlmQuestionResult(error=str(exc))

            selected = result.selected_answer_id
            status = ResponseStatus.COMPLETED
            if result.error:
                status = ResponseStatus.FAILED
            elif selected not in observation.allowed_action_ids:
                status = ResponseStatus.INVALID
            attempts.append(
                TaskAttemptRecord(
                    attempt=attempt_number,
                    status=status,
                    selected_action_id=selected,
                    raw_response=result.raw_response,
                    structured_response=result.structured_response,
                    logprobs=result.logprobs,
                    error=result.error,
                )
            )
            if status == ResponseStatus.COMPLETED:
                break
            if status == ResponseStatus.FAILED:
                break
            messages = [
                *messages,
                {
                    "role": "assistant",
                    "content": result.raw_response or "",
                },
                {
                    "role": "user",
                    "content": (
                        "That was not a valid option. Reply with exactly one of: "
                        + ", ".join(observation.allowed_action_ids)
                    ),
                },
            ]

        final_attempt = attempts[-1]
        if final_attempt.status != ResponseStatus.COMPLETED or selected is None:
            record = TaskTrialRecord(
                experiment_id=experiment_id,
                session_id=session_id,
                run_id=run_id,
                subject_id=persona.persona_id,
                task_id=task.id,
                task_version=task.version,
                schedule_id=resolved_schedule.id,
                trial_index=state.trial_index + 1,
                observation=observation,
                attempts=attempts,
                message_start_index=message_start,
                message_end_index=len(messages),
                status=final_attempt.status,
                error=final_attempt.error or "invalid action after corrective retry",
                metadata=response_metadata or {},
            )
            append_jsonl_record(response_path, record)
            records.append(record)
            _append_messages(conversation_path, messages[message_start:])
            return TaskRunResult(records=records, status=final_attempt.status)

        transition = task.apply_action(state, selected, resolved_schedule)
        committed_messages: list[Message] = [
            {"role": "user", "content": observation.prompt},
            {"role": "assistant", "content": selected},
            {"role": "user", "content": transition.feedback},
        ]
        record = TaskTrialRecord(
            experiment_id=experiment_id,
            session_id=session_id,
            run_id=run_id,
            subject_id=persona.persona_id,
            task_id=task.id,
            task_version=task.version,
            schedule_id=resolved_schedule.id,
            trial_index=transition.trial_index,
            observation=observation,
            attempts=attempts,
            transition=transition,
            message_start_index=message_start,
            message_end_index=message_start + len(committed_messages),
            status=ResponseStatus.COMPLETED,
            metadata=response_metadata or {},
        )
        append_jsonl_record(response_path, record)
        _append_messages(conversation_path, committed_messages)
        records.append(record)
        transitions.append(transition)
        state = transition.state
        history.extend(committed_messages)
        successful_this_call += 1

    summary = task.summarize(transitions) if task.is_complete(state) else None
    return TaskRunResult(
        records=records,
        summary=summary,
        status=(
            ResponseStatus.COMPLETED
            if task.is_complete(state)
            else ResponseStatus.SKIPPED
        ),
    )


def _replay(
    task: IowaGamblingTask,
    schedule: ResolvedIowaSchedule,
    records: list[TaskTrialRecord],
) -> tuple[TaskState, list[TaskTransition]]:
    state = task.initial_state(schedule)
    transitions: list[TaskTransition] = []
    for expected_index, record in enumerate(records, start=1):
        if record.trial_index != expected_index or record.transition is None:
            raise ValueError("task ledger is not a contiguous sequence")
        replayed = task.apply_action(
            state,
            record.transition.visible_action_id,
            schedule,
        )
        if replayed != record.transition:
            raise ValueError(f"task ledger diverges at trial {expected_index}")
        transitions.append(replayed)
        state = replayed.state
    return state, transitions


def _history(
    persona: Persona,
    task: IowaGamblingTask,
    transitions: list[TaskTransition],
    initial_history: Sequence[Message] | None = None,
) -> list[Message]:
    history: list[Message] = (
        [*initial_history, {"role": "user", "content": task.instruction()}]
        if initial_history is not None
        else [
            {"role": "system", "content": render_persona_intro(persona)},
            {"role": "user", "content": task.instruction()},
        ]
    )
    for transition in transitions:
        history.extend(
            [
                {
                    "role": "user",
                    "content": (
                        f"Trial {transition.trial_index}. Choose one available option."
                    ),
                },
                {
                    "role": "assistant",
                    "content": transition.visible_action_id,
                },
                {"role": "user", "content": transition.feedback},
            ]
        )
    return history


def _load_records(path: Path) -> list[TaskTrialRecord]:
    if not path.exists():
        return []
    return [
        TaskTrialRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_once(path: Path, model: BaseModel) -> None:
    if path.exists():
        existing = type(model).model_validate_json(path.read_text(encoding="utf-8"))
        if existing != model:
            raise ValueError(f"persisted artifact does not match requested value: {path}")
        return
    write_json_document(path, model)


def _write_conversation(path: Path, messages: list[Message]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for message in messages:
            file.write(ChatMessage(**message).model_dump_json() + "\n")


def _append_messages(path: Path, messages: list[Message]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        for message in messages:
            file.write(ChatMessage(**message).model_dump_json() + "\n")
