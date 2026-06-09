import asyncio
import json

from llm_behavior_lab.behavioral_tasks.batch import run_persisted_task_batch_async
from llm_behavior_lab.behavioral_tasks.iowa_gambling import (
    IowaGamblingConfig,
    IowaGamblingTask,
)
from llm_behavior_lab.models import LlmQuestionResult, ModelSettings
from llm_behavior_lab.personas.factory import (
    PersonaFactory,
    PersonaFactoryRequest,
    RequestedDemographicField,
)


class FirstChoiceClient:
    def complete(self, messages, settings, allowed_answer_ids):
        selected = allowed_answer_ids[0]
        return LlmQuestionResult(selected_answer_id=selected, raw_response=selected)


def test_async_batch_isolates_subject_ledgers_and_writes_run_metadata(tmp_path) -> None:
    personas = PersonaFactory().create_demographics_batch(
        PersonaFactoryRequest(
            count=2,
            requested_fields={RequestedDemographicField.AGE},
            seed=3,
            experiment_id="card-batch-one",
        )
    )
    task = IowaGamblingTask(IowaGamblingConfig(trial_count=2))

    run = asyncio.run(
        run_persisted_task_batch_async(
            personas=personas,
            task=task,
            settings=ModelSettings(
                model="test",
                provider_base_url="http://localhost",
                temperature=0,
                timeout_seconds=10,
                seed=3,
            ),
            client_factory=FirstChoiceClient,
            project_root=tmp_path,
            concurrency=2,
        )
    )

    run_root = tmp_path / "experiments" / "card-batch-one" / run.run_id
    response_paths = sorted((run_root / "responses").glob("*.jsonl"))
    schedule_paths = sorted((run_root / "schedules").glob("*.json"))
    row = json.loads((run_root / "run.jsonl").read_text(encoding="utf-8"))

    assert len(response_paths) == 2
    assert all(path.read_text(encoding="utf-8").count("\n") == 2 for path in response_paths)
    assert len(schedule_paths) == 2
    assert row["procedure_kind"] == "task"
    assert row["procedure_id"] == task.id
    assert row["item_count"] == 4


def test_batch_resume_preserves_run_session_id(tmp_path) -> None:
    personas = PersonaFactory().create_demographics_batch(
        PersonaFactoryRequest(
            count=1,
            requested_fields={RequestedDemographicField.AGE},
            seed=4,
            experiment_id="card-batch-two",
        )
    )
    task = IowaGamblingTask(IowaGamblingConfig(trial_count=1))
    settings = ModelSettings(
        model="test",
        provider_base_url="http://localhost",
        temperature=0,
        timeout_seconds=10,
        seed=4,
    )
    first = asyncio.run(
        run_persisted_task_batch_async(
            personas=personas,
            task=task,
            settings=settings,
            client_factory=FirstChoiceClient,
            project_root=tmp_path,
            concurrency=1,
        )
    )
    asyncio.run(
        run_persisted_task_batch_async(
            personas=personas,
            task=task,
            settings=settings,
            client_factory=FirstChoiceClient,
            project_root=tmp_path,
            concurrency=1,
            run_id=first.run_id,
        )
    )

    run_path = (
        tmp_path / "experiments" / "card-batch-two" / first.run_id / "run.jsonl"
    )
    rows = [
        json.loads(line)
        for line in run_path.read_text(encoding="utf-8").splitlines()
    ]
    assert {row["session_id"] for row in rows} == {first.session_id}
