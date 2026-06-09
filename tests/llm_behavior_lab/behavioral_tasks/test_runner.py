from collections.abc import Sequence
from pathlib import Path

from llm_behavior_lab.behavioral_tasks.iowa_gambling import IowaGamblingConfig, IowaGamblingTask
from llm_behavior_lab.behavioral_tasks.runner import run_behavioral_task
from llm_behavior_lab.models import LlmQuestionResult, ModelSettings, Persona


class SequenceClient:
    def __init__(self, answers: Sequence[str | None]) -> None:
        self.answers = list(answers)
        self.messages: list[list[dict[str, str]]] = []

    def complete(self, messages, settings, allowed_answer_ids):
        self.messages.append(list(messages))
        answer = self.answers.pop(0)
        return LlmQuestionResult(selected_answer_id=answer, raw_response=answer)


def settings() -> ModelSettings:
    return ModelSettings(
        model="test-model",
        provider_base_url="http://localhost",
        temperature=0,
        timeout_seconds=10,
        seed=17,
    )


def test_runner_retains_persona_and_trial_feedback_history(tmp_path: Path) -> None:
    config = IowaGamblingConfig(trial_count=2)
    task = IowaGamblingTask(config)
    schedule = task.resolve_schedule(seed=17, subject_id="subject-1")
    labels = list(schedule.label_mapping)
    client = SequenceClient(labels)

    result = run_behavioral_task(
        persona=Persona(persona_id="subject-1", features={"age": "42"}),
        task=task,
        settings=settings(),
        client=client,
        project_root=tmp_path,
        experiment_id="task-test-one",
        run_id="run-task-test",
        resolved_schedule=schedule,
    )

    assert len(result.records) == 2
    assert client.messages[0][0]["role"] == "system"
    assert "42" in client.messages[0][0]["content"]
    assert len(client.messages[1]) > len(client.messages[0])
    assert any("Running balance" in message["content"] for message in client.messages[1])
    assert all("Iowa" not in message["content"] for call in client.messages for message in call)


def test_runner_retries_once_without_advancing_state(tmp_path: Path) -> None:
    config = IowaGamblingConfig(trial_count=1)
    task = IowaGamblingTask(config)
    schedule = task.resolve_schedule(seed=17, subject_id="subject-1")
    valid_label = next(iter(schedule.label_mapping))
    client = SequenceClient([None, valid_label])

    result = run_behavioral_task(
        persona=Persona(persona_id="subject-1", features={}),
        task=task,
        settings=settings(),
        client=client,
        project_root=tmp_path,
        experiment_id="task-test-two",
        run_id="run-task-test",
        resolved_schedule=schedule,
    )

    assert len(result.records) == 1
    assert result.records[0].trial_index == 1
    assert len(result.records[0].attempts) == 2
    assert result.records[0].attempts[0].status == "invalid"
    assert result.records[0].attempts[1].status == "completed"


def test_runner_resumes_from_completed_trial_ledger(tmp_path: Path) -> None:
    config = IowaGamblingConfig(trial_count=3)
    task = IowaGamblingTask(config)
    schedule = task.resolve_schedule(seed=17, subject_id="subject-1")
    labels = list(schedule.label_mapping)

    first = run_behavioral_task(
        persona=Persona(persona_id="subject-1", features={}),
        task=task,
        settings=settings(),
        client=SequenceClient(labels[:2]),
        project_root=tmp_path,
        experiment_id="task-test-three",
        run_id="run-task-test",
        resolved_schedule=schedule,
        max_trials=2,
    )
    resumed_client = SequenceClient([labels[2]])
    resumed = run_behavioral_task(
        persona=Persona(persona_id="subject-1", features={}),
        task=task,
        settings=settings(),
        client=resumed_client,
        project_root=tmp_path,
        experiment_id="task-test-three",
        run_id="run-task-test",
        resolved_schedule=schedule,
        resume=True,
    )

    assert len(first.records) == 2
    assert len(resumed.records) == 3
    assert resumed.records[-1].trial_index == 3
    assert {record.session_id for record in resumed.records} == {
        first.records[0].session_id
    }
    assert len(resumed_client.messages) == 1
    assert any("Running balance" in item["content"] for item in resumed_client.messages[0])


def test_failed_subject_requires_explicit_retry(tmp_path: Path) -> None:
    task = IowaGamblingTask(IowaGamblingConfig(trial_count=1))
    schedule = task.resolve_schedule(seed=17, subject_id="subject-1")
    persona = Persona(persona_id="subject-1", features={})
    run_behavioral_task(
        persona=persona,
        task=task,
        settings=settings(),
        project_root=tmp_path,
        experiment_id="task-test-four",
        run_id="run-task-test",
        resolved_schedule=schedule,
        client=SequenceClient([None, None]),
    )
    blocked_client = SequenceClient([next(iter(schedule.label_mapping))])

    blocked = run_behavioral_task(
        persona=persona,
        task=task,
        settings=settings(),
        project_root=tmp_path,
        experiment_id="task-test-four",
        run_id="run-task-test",
        resolved_schedule=schedule,
        client=blocked_client,
        resume=True,
    )
    assert blocked_client.messages == []
    retried = run_behavioral_task(
        persona=persona,
        task=task,
        settings=settings(),
        project_root=tmp_path,
        experiment_id="task-test-four",
        run_id="run-task-test",
        resolved_schedule=schedule,
        client=blocked_client,
        resume=True,
        retry_failed=True,
    )

    assert blocked.status == "invalid"
    assert blocked_client.messages
    assert retried.status == "completed"
