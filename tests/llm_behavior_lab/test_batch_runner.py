import asyncio
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest
from loguru import logger

from llm_behavior_lab.models import LlmQuestionResult, ModelSettings
from llm_behavior_lab.personas.factory import (
    PersonaFactory,
    PersonaFactoryRequest,
    PersonaGenerationConfig,
    RequestedDemographicField,
)
from llm_behavior_lab.protocols import ExperimentProtocol
from llm_behavior_lab.questionnaires.base.scale import Questionnaire
from llm_behavior_lab.questionnaires.bfi10 import BFI_10
from llm_behavior_lab.responses.base import ResponseStatus
from llm_behavior_lab.runner import (
    run_persisted_persona_batch_async,
    run_persona_questionnaire_batch,
    run_protocol_experiment,
)


class FakeBatchClient:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    def complete(
        self,
        messages: Sequence[dict[str, str]],
        settings: ModelSettings,
        allowed_answer_ids: Sequence[str],
    ) -> LlmQuestionResult:
        self.calls.append(list(messages))
        return LlmQuestionResult(
            selected_answer_id=allowed_answer_ids[0],
            raw_response=allowed_answer_ids[0],
        )


def test_batch_generates_personas_and_runs_each_one_under_experiment(tmp_path) -> None:
    settings = ModelSettings(
        model="openai/gpt-oss-20b",
        provider_base_url="http://localhost:1234/v1",
        temperature=0.2,
        timeout_seconds=60.0,
    )

    result = run_persona_questionnaire_batch(
        questionnaire=BFI_10,
        settings=settings,
        client=FakeBatchClient(),
        project_root=tmp_path,
        experiment_id="bfi10-lmstudio-test",
        persona_count=100,
        seed=11,
    )

    experiment_root = tmp_path / "experiments" / "bfi10-lmstudio-test"
    run_roots = sorted(path for path in experiment_root.iterdir() if path.is_dir())
    run_root = run_roots[0]
    responses_root = run_root / "responses"

    assert result.experiment_id == "bfi10-lmstudio-test"
    assert len(result.personas.personas) == 100
    assert result.personas.metadata.requested_fields == sorted(
        RequestedDemographicField, key=lambda field: field.value
    )
    assert len(result.runs) == 1
    assert len(run_roots) == 1
    assert (experiment_root / "personas.json").exists()
    assert (experiment_root / "metadata.json").exists()
    assert not (experiment_root / "sessions").exists()

    metadata_rows = json.loads((experiment_root / "metadata.json").read_text(encoding="utf-8"))[
        "runs"
    ]
    assert len(metadata_rows) == 1

    assert run_root.name.startswith("run-bfi10-openai-gpt-oss-20b-")
    assert {path.name for path in run_root.iterdir()} == {
        "run.json",
        "responses",
        "scale.json",
    }
    assert (
        json.loads((run_root / "run.json").read_text(encoding="utf-8"))["run_id"]
        == result.runs[0].run_id
    )
    assert len(list(responses_root.glob("*.jsonl"))) == 100
    for response_path in responses_root.glob("*.jsonl"):
        assert response_path.read_text(encoding="utf-8").count("\n") == 10
    scale = Questionnaire.model_validate_json((run_root / "scale.json").read_text())
    assert scale.id == BFI_10.id
    assert scale.shorthand == "bfi10"


def test_batch_response_records_share_experiment_session_and_run_ids(tmp_path) -> None:
    settings = ModelSettings(
        model="openai/gpt-oss-20b",
        provider_base_url="http://localhost:1234/v1",
        temperature=0.2,
        timeout_seconds=60.0,
    )

    result = run_persona_questionnaire_batch(
        questionnaire=BFI_10,
        settings=settings,
        client=FakeBatchClient(),
        project_root=tmp_path,
        experiment_id="bfi10-lmstudio-test",
        persona_count=2,
        seed=12,
    )

    run = result.runs[0]
    responses_root = tmp_path / "experiments" / result.experiment_id / run.run_id / "responses"
    response_paths = sorted(responses_root.glob("*.jsonl"))
    assert len(response_paths) == 2

    for responses_path in response_paths:
        response_rows = [
            json.loads(line) for line in responses_path.read_text(encoding="utf-8").splitlines()
        ]
        assert responses_path.name == f"{response_rows[0]['subject_id']}.jsonl"
        assert {row["session_id"] for row in response_rows} == {result.session_id}
        assert {row["run_id"] for row in response_rows} == {run.run_id}
        assert {row["metadata"]["experiment_id"] for row in response_rows} == {result.experiment_id}
        assert list(response_rows[0])[:3] == ["subject_id", "session_id", "run_id"]


def test_batch_runner_passes_context_to_questionnaire_prompt(tmp_path) -> None:
    settings = ModelSettings(
        model="openai/gpt-oss-20b",
        provider_base_url="http://localhost:1234/v1",
        temperature=0.2,
        timeout_seconds=60.0,
    )
    client = FakeBatchClient()

    run_persona_questionnaire_batch(
        questionnaire=BFI_10,
        settings=settings,
        client=client,
        project_root=tmp_path,
        experiment_id="bfi10-lmstudio-test",
        persona_count=1,
        seed=15,
        context="Read this batch vignette before answering.",
    )

    assert "Additional context:" in client.calls[0][0]["content"]
    assert "Read this batch vignette before answering." in client.calls[0][0]["content"]


def test_batch_run_record_does_not_duplicate_persona_snapshot(tmp_path) -> None:
    settings = ModelSettings(
        model="openai/gpt-oss-20b",
        provider_base_url="http://localhost:1234/v1",
        temperature=0.2,
        timeout_seconds=60.0,
    )

    result = run_persona_questionnaire_batch(
        questionnaire=BFI_10,
        settings=settings,
        client=FakeBatchClient(),
        project_root=tmp_path,
        experiment_id="bfi10-lmstudio-test",
        persona_count=2,
        seed=14,
    )

    run_path = tmp_path / "experiments" / result.experiment_id / result.runs[0].run_id / "run.json"
    run_row = json.loads(run_path.read_text(encoding="utf-8"))

    assert "persona_snapshot" not in run_row
    assert run_row["persona_count"] == 2
    assert run_row["subject_ids"] == [
        str(persona.subject_id) for persona in result.personas.personas
    ]


def test_batch_runner_logs_progress(tmp_path) -> None:
    messages: list[str] = []
    handler_id = logger.add(lambda message: messages.append(str(message)), level="INFO")
    settings = ModelSettings(
        model="openai/gpt-oss-20b",
        provider_base_url="http://localhost:1234/v1",
        temperature=0.2,
        timeout_seconds=60.0,
    )

    try:
        run_persona_questionnaire_batch(
            questionnaire=BFI_10,
            settings=settings,
            client=FakeBatchClient(),
            project_root=tmp_path,
            experiment_id="bfi10-lmstudio-test",
            persona_count=2,
            seed=13,
        )
    finally:
        logger.remove(handler_id)

    log_text = "\n".join(messages)
    assert "Starting persona questionnaire batch" in log_text
    assert "Starting persona run 1/2" in log_text
    assert "Starting persona run 2/2" in log_text
    assert "Completed persona questionnaire batch" in log_text


def test_batch_runner_uses_persona_generation_config(tmp_path) -> None:
    settings = ModelSettings(
        model="openai/gpt-oss-20b",
        provider_base_url="http://localhost:1234/v1",
        temperature=0.2,
        timeout_seconds=60.0,
    )

    result = run_persona_questionnaire_batch(
        questionnaire=BFI_10,
        settings=settings,
        client=FakeBatchClient(),
        project_root=tmp_path,
        experiment_id="bfi10-lmstudio-test",
        persona_count=3,
        seed=17,
        persona_config=PersonaGenerationConfig(
            field_probabilities={
                RequestedDemographicField.COUNTRY: {"PL": 1.0},
                RequestedDemographicField.AFFLUENCE_LEVEL: {"middle": 1.0},
            }
        ),
    )

    assert {persona.features.country for persona in result.personas.personas} == {"PL"}
    assert {persona.features.affluence_level for persona in result.personas.personas} == {"middle"}


def test_protocol_runner_writes_protocol_artifacts_and_metadata(tmp_path) -> None:
    settings = ModelSettings(
        model="openai/gpt-oss-20b",
        provider_base_url="http://localhost:1234/v1",
        temperature=0.2,
        timeout_seconds=60.0,
        seed=99,
    )
    protocol = ExperimentProtocol.model_validate(
        {
            "version": "1.0",
            "name": "gender-affluence-factorial",
            "base_persona_count": 1,
            "seed": 123,
            "iterations": 2,
            "requested_fields": ["age", "country", "gender", "affluence_level"],
            "base_persona_config": {"field_probabilities": {"country": {"PL": 1.0}}},
            "factors": [
                {
                    "name": "gender",
                    "field": "gender",
                    "levels": [
                        {"id": "female", "value": "female"},
                        {"id": "male", "value": "male"},
                    ],
                },
                {
                    "name": "affluence",
                    "field": "affluence_level",
                    "levels": [
                        {"id": "low", "value": "low"},
                        {"id": "middle", "value": "middle"},
                    ],
                },
            ],
        }
    )

    client = FakeBatchClient()
    result = run_protocol_experiment(
        protocol=protocol,
        questionnaire=BFI_10,
        settings=settings,
        client=client,
        project_root=tmp_path,
        experiment_id="proto-study-one",
        context="Read this protocol vignette before answering.",
    )

    experiment_root = tmp_path / "experiments" / "proto-study-one"
    run_root = experiment_root / result.runs[0].run_id
    assert (experiment_root / "protocol.json").exists()
    assert (experiment_root / "base_personas.json").exists()
    assert (experiment_root / "personas.json").exists()
    assert (experiment_root / "protocol_assignments.json").exists()
    assert len(list((run_root / "responses").glob("*.jsonl"))) == 1 * 2 * 2 * 2

    assignment_rows = json.loads(
        (experiment_root / "protocol_assignments.json").read_text(encoding="utf-8")
    )["assignments"]
    assert len(assignment_rows) == 8
    response_path = sorted((run_root / "responses").glob("*.jsonl"))[0]
    response_rows = [json.loads(line) for line in response_path.read_text().splitlines()]
    assignment = next(
        row for row in assignment_rows if row["subject_id"] == response_rows[0]["subject_id"]
    )
    assert response_rows[0]["metadata"]["protocol_name"] == "gender-affluence-factorial"
    assert response_rows[0]["metadata"]["base_subject_id"] == assignment["base_subject_id"]
    assert response_rows[0]["metadata"]["condition_id"] == assignment["condition_id"]
    assert response_rows[0]["metadata"]["iteration_index"] == assignment["iteration_index"]
    assert response_rows[0]["metadata"]["factor_values"]["gender"] in {"female", "male"}
    assert "persona_snapshot" not in response_rows[0]
    assert "Additional context:" in client.calls[0][0]["content"]
    assert "Read this protocol vignette before answering." in client.calls[0][0]["content"]


@dataclass
class AsyncCallState:
    active: int = 0
    max_active: int = 0
    per_subject_items: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    two_active: asyncio.Event = field(default_factory=asyncio.Event)


class TrackingAsyncClient:
    def __init__(self, state: AsyncCallState) -> None:
        self.state = state

    async def complete(
        self,
        messages: Sequence[dict[str, str]],
        settings: ModelSettings,
        allowed_answer_ids: Sequence[str],
    ) -> LlmQuestionResult:
        system = messages[0]["content"]
        subject_id = system.split("Persona ID: ", 1)[1].splitlines()[0]
        prompt = messages[-1]["content"]
        item_id = next(item.id for item in BFI_10.items if prompt.startswith(item.text))
        self.state.active += 1
        self.state.max_active = max(self.state.max_active, self.state.active)
        if self.state.active == 2:
            self.state.two_active.set()
        try:
            await asyncio.sleep(0.005)
            self.state.per_subject_items[subject_id].append(item_id)
            return LlmQuestionResult(
                selected_answer_id=allowed_answer_ids[0],
                raw_response=allowed_answer_ids[0],
            )
        finally:
            self.state.active -= 1


def _persona_batch(count: int):
    return PersonaFactory().create_demographics_batch(
        PersonaFactoryRequest(
            count=count,
            requested_fields={RequestedDemographicField.AGE},
            seed=11,
            experiment_id="async-study-one",
        )
    )


@pytest.mark.anyio
async def test_async_batch_bounds_concurrency_and_keeps_subject_items_sequential(
    tmp_path,
) -> None:
    personas = _persona_batch(3)
    state = AsyncCallState()

    result = await run_persisted_persona_batch_async(
        personas=personas,
        questionnaire=BFI_10,
        settings=ModelSettings(
            model="test",
            provider_base_url="http://localhost",
            temperature=0,
            timeout_seconds=10,
            max_concurrency=2,
        ),
        client_factory=lambda: TrackingAsyncClient(state),
        project_root=tmp_path,
    )

    assert state.max_active == 2
    assert state.per_subject_items == {
        str(persona.subject_id): [item.id for item in BFI_10.items] for persona in personas.personas
    }
    assert result.runs[0].status is ResponseStatus.COMPLETED


@pytest.mark.anyio
async def test_async_batch_cancellation_stops_waiting_subjects(tmp_path) -> None:
    personas = _persona_batch(3)
    state = AsyncCallState()
    cancel_event = asyncio.Event()

    async def request_cancellation() -> None:
        await state.two_active.wait()
        cancel_event.set()

    cancellation = asyncio.create_task(request_cancellation())
    result = await run_persisted_persona_batch_async(
        personas=personas,
        questionnaire=BFI_10,
        settings=ModelSettings(
            model="test",
            provider_base_url="http://localhost",
            temperature=0,
            timeout_seconds=10,
            max_concurrency=2,
        ),
        client_factory=lambda: TrackingAsyncClient(state),
        project_root=tmp_path,
        cancel_event=cancel_event,
    )
    await cancellation

    assert len(state.per_subject_items) == 2
    assert all(
        item_ids == [item.id for item in BFI_10.items]
        for item_ids in state.per_subject_items.values()
    )
    assert result.runs[0].status is ResponseStatus.CANCELLED
    responses_root = (
        tmp_path / "experiments" / personas.experiment_id / result.runs[0].run_id / "responses"
    )
    assert len(list(responses_root.glob("*.jsonl"))) == 2


@pytest.mark.anyio
async def test_async_batch_resume_is_idempotent_and_replaces_run_index_entry(
    tmp_path,
) -> None:
    personas = _persona_batch(1)
    state = AsyncCallState()
    settings = ModelSettings(
        model="test",
        provider_base_url="http://localhost",
        temperature=0,
        timeout_seconds=10,
    )
    first = await run_persisted_persona_batch_async(
        personas=personas,
        questionnaire=BFI_10,
        settings=settings,
        client_factory=lambda: TrackingAsyncClient(state),
        project_root=tmp_path,
    )
    first_started_at = first.runs[0].started_at
    resumed_state = AsyncCallState()

    resumed = await run_persisted_persona_batch_async(
        personas=personas,
        questionnaire=BFI_10,
        settings=settings,
        client_factory=lambda: TrackingAsyncClient(resumed_state),
        project_root=tmp_path,
        run_id=first.runs[0].run_id,
    )

    metadata = json.loads(
        (tmp_path / "experiments" / personas.experiment_id / "metadata.json").read_text()
    )
    assert resumed_state.per_subject_items == {}
    assert resumed.runs[0].started_at == first_started_at
    assert [run["run_id"] for run in metadata["runs"]] == [first.runs[0].run_id]


@pytest.mark.anyio
@pytest.mark.parametrize("mismatch", ["provider", "questionnaire", "personas"])
async def test_async_batch_rejects_incompatible_resume_before_provider_calls(
    tmp_path,
    mismatch: str,
) -> None:
    personas = _persona_batch(1)
    settings = ModelSettings(
        model="test",
        provider_base_url="http://localhost",
        temperature=0,
        timeout_seconds=10,
    )
    first = await run_persisted_persona_batch_async(
        personas=personas,
        questionnaire=BFI_10,
        settings=settings,
        client_factory=lambda: TrackingAsyncClient(AsyncCallState()),
        project_root=tmp_path,
    )
    resumed_personas = personas
    resumed_questionnaire = BFI_10
    resumed_settings = settings
    if mismatch == "provider":
        resumed_settings = settings.model_copy(update={"max_concurrency": 7})
    elif mismatch == "questionnaire":
        resumed_questionnaire = BFI_10.model_copy(update={"version": "2.0"})
    else:
        resumed_personas = _persona_batch(2)
    factory_calls = 0

    def client_factory() -> TrackingAsyncClient:
        nonlocal factory_calls
        factory_calls += 1
        return TrackingAsyncClient(AsyncCallState())

    with pytest.raises(ValueError, match=mismatch.removesuffix("s")):
        await run_persisted_persona_batch_async(
            personas=resumed_personas,
            questionnaire=resumed_questionnaire,
            settings=resumed_settings,
            client_factory=client_factory,
            project_root=tmp_path,
            run_id=first.runs[0].run_id,
        )

    assert factory_calls == 0


@pytest.mark.anyio
async def test_async_batch_rejects_partial_ledgers_without_run_manifest(
    tmp_path,
) -> None:
    personas = _persona_batch(1)
    settings = ModelSettings(
        model="test",
        provider_base_url="http://localhost",
        temperature=0,
        timeout_seconds=10,
    )
    first = await run_persisted_persona_batch_async(
        personas=personas,
        questionnaire=BFI_10,
        settings=settings,
        client_factory=lambda: TrackingAsyncClient(AsyncCallState()),
        project_root=tmp_path,
    )
    run_root = tmp_path / "experiments" / personas.experiment_id / first.runs[0].run_id
    response_path = next((run_root / "responses").glob("*.jsonl"))
    first_line = response_path.read_text().splitlines()[0]
    response_path.write_text(first_line + "\n", encoding="utf-8")
    (run_root / "run.json").unlink()
    with pytest.raises(ValueError, match="missing their run manifest"):
        await run_persisted_persona_batch_async(
            personas=personas,
            questionnaire=BFI_10,
            settings=settings,
            client_factory=lambda: TrackingAsyncClient(AsyncCallState()),
            project_root=tmp_path,
            run_id=first.runs[0].run_id,
        )


@pytest.mark.anyio
async def test_async_batch_persists_partial_manifest_before_first_provider_call(
    tmp_path,
) -> None:
    personas = _persona_batch(1)
    settings = ModelSettings(
        model="test",
        provider_base_url="http://localhost",
        temperature=0,
        timeout_seconds=10,
    )

    class InspectingClient:
        async def complete(self, messages, settings, allowed_answer_ids):
            run_root = next((tmp_path / "experiments" / personas.experiment_id).glob("run-*"))
            manifest = json.loads((run_root / "run.json").read_text())
            assert manifest["status"] == "partial"
            assert manifest["completed_at"] is None
            assert (run_root / "scale.json").exists()
            return LlmQuestionResult(
                selected_answer_id=allowed_answer_ids[0],
                raw_response=allowed_answer_ids[0],
            )

    await run_persisted_persona_batch_async(
        personas=personas,
        questionnaire=BFI_10,
        settings=settings,
        client_factory=InspectingClient,
        project_root=tmp_path,
    )
