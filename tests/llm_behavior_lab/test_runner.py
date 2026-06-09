import json
from collections.abc import Sequence

import pytest

from llm_behavior_lab.models import LlmQuestionResult, ModelSettings, Persona, ProviderCapabilities
from llm_behavior_lab.questionnaires.base.scale import Questionnaire
from llm_behavior_lab.questionnaires.bfi10 import BFI_10
from llm_behavior_lab.responses.base import LikertAnswerValue, ResponseStatus
from llm_behavior_lab.runner import run_questionnaire, run_questionnaire_async

TEST_PERSONA = Persona(
    persona_id="test_persona",
    features={
        "age": "35",
        "country": "Poland",
        "education": "university degree",
        "employment": "full-time",
    },
)


class FakeSyncClient:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []
        self.seeds: list[int | None] = []

    def complete(
        self,
        messages: Sequence[dict[str, str]],
        settings: ModelSettings,
        allowed_answer_ids: Sequence[str],
    ) -> LlmQuestionResult:
        self.calls.append(list(messages))
        self.seeds.append(settings.seed)
        return LlmQuestionResult(
            selected_answer_id=allowed_answer_ids[0],
            raw_response=allowed_answer_ids[0],
            structured_response={"selected_answer_id": allowed_answer_ids[0]},
        )


def test_run_questionnaire_retains_context_and_saves_records(tmp_path) -> None:
    settings = ModelSettings(
        model="llama3.1",
        provider_base_url="http://localhost:11434/v1",
        temperature=0.2,
        timeout_seconds=60.0,
        capabilities=ProviderCapabilities(supports_structured_outputs=True),
    )
    client = FakeSyncClient()

    records = run_questionnaire(
        persona=TEST_PERSONA,
        questionnaire=BFI_10,
        settings=settings,
        client=client,
        project_root=tmp_path,
        experiment_id="pilot-study-one",
        context="Read this vignette before answering.",
    )

    assert len(records) == len(BFI_10.items)
    assert records[0].run_id.startswith("run-bfi10-llama3-1-")
    assert records[0].session_id.startswith("session-")
    assert records[0].subject_id == TEST_PERSONA.persona_id
    assert records[0].run_id != records[0].session_id
    assert {record.run_id for record in records} == {records[0].run_id}
    assert {record.session_id for record in records} == {records[0].session_id}
    assert records[0].questionnaire_id == BFI_10.id
    assert records[0].item_id == BFI_10.items[0].id
    assert records[0].status == ResponseStatus.COMPLETED
    assert isinstance(records[0].answer, LikertAnswerValue)
    assert records[0].answer.value == 1
    assert records[0].answer.label == "Strongly agree"
    assert "Additional context:" in client.calls[0][0]["content"]
    assert "Read this vignette before answering." in client.calls[0][0]["content"]
    assert len(client.calls[1]) > len(client.calls[0])
    response_path = (
        tmp_path
        / "experiments"
        / "pilot-study-one"
        / records[0].run_id
        / "responses"
        / f"{TEST_PERSONA.persona_id}.jsonl"
    )
    run_path = response_path.parents[1] / "run.jsonl"
    metadata_path = response_path.parents[2] / "metadata.jsonl"
    scale_path = response_path.parents[1] / "scale.json"
    assert response_path.read_text(encoding="utf-8").count("\n") == len(BFI_10.items)
    assert run_path.read_text(encoding="utf-8").count("\n") == 1
    metadata_rows = [
        json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()
    ]
    run_rows = [json.loads(line) for line in run_path.read_text(encoding="utf-8").splitlines()]
    response_rows = [json.loads(line) for line in response_path.read_text().splitlines()]
    assert len(metadata_rows) == 1
    assert metadata_rows[0]["run_id"] == records[0].run_id
    assert run_rows[0]["experiment_id"] == "pilot-study-one"
    assert run_rows[0]["session_id"] == records[0].session_id
    assert run_rows[0]["run_id"] == records[0].run_id
    assert "persona_snapshot" not in run_rows[0]
    assert run_rows[0]["subject_ids"] == [TEST_PERSONA.persona_id]
    assert run_rows[0]["persona_count"] == 1
    assert run_rows[0]["questionnaire_shorthand"] == "bfi10"
    assert run_rows[0]["model_slug"] == "llama3-1"
    assert run_rows[0]["output_paths"]["responses"].endswith("/responses")
    assert Questionnaire.model_validate_json(scale_path.read_text()).id == BFI_10.id
    assert list(response_rows[0])[:3] == ["subject_id", "session_id", "run_id"]
    assert "Poland" not in response_path.read_text(encoding="utf-8")
    assert "university degree" not in response_path.read_text(encoding="utf-8")
    assert run_rows[0]["error_count"] == 0
    assert run_rows[0]["item_count"] == len(BFI_10.items)


def test_run_questionnaire_sets_and_persists_seed_for_each_item(tmp_path) -> None:
    settings = ModelSettings(
        model="llama3.1",
        provider_base_url="http://localhost:11434/v1",
        temperature=0.2,
        timeout_seconds=60.0,
        seed=123,
    )
    client = FakeSyncClient()

    records = run_questionnaire(
        persona=TEST_PERSONA,
        questionnaire=BFI_10,
        settings=settings,
        client=client,
        project_root=tmp_path,
        experiment_id="pilot-study-one",
    )

    record_seeds = [record.metadata["seed"] for record in records]
    assert client.seeds == record_seeds
    assert len(set(record_seeds)) == len(BFI_10.items)
    run_path = tmp_path / "experiments" / "pilot-study-one" / records[0].run_id / "run.jsonl"
    run_row = json.loads(run_path.read_text(encoding="utf-8"))
    assert run_row["metadata"]["base_seed"] == 123


class FailingSyncClient:
    def complete(
        self,
        messages: Sequence[dict[str, str]],
        settings: ModelSettings,
        allowed_answer_ids: Sequence[str],
    ) -> LlmQuestionResult:
        raise RuntimeError("provider unavailable")


def test_run_questionnaire_saves_failure_records(tmp_path) -> None:
    settings = ModelSettings(
        model="llama3.1",
        provider_base_url="http://localhost:11434/v1",
        temperature=0.2,
        timeout_seconds=60.0,
    )

    records = run_questionnaire(
        persona=TEST_PERSONA,
        questionnaire=BFI_10,
        settings=settings,
        client=FailingSyncClient(),
        project_root=tmp_path,
        experiment_id="pilot-study-one",
    )

    assert len(records) == len(BFI_10.items)
    assert records[0].status == ResponseStatus.FAILED
    assert records[0].error == "provider unavailable"
    assert records[0].answer is None
    response_path = (
        tmp_path
        / "experiments"
        / "pilot-study-one"
        / records[0].run_id
        / "responses"
        / f"{TEST_PERSONA.persona_id}.jsonl"
    )
    assert response_path.read_text(encoding="utf-8").count("\n") == len(BFI_10.items)
    run_row = json.loads((response_path.parents[1] / "run.jsonl").read_text(encoding="utf-8"))
    assert run_row["status"] == ResponseStatus.FAILED
    assert run_row["error_count"] == len(BFI_10.items)


class FakeAsyncClient:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    async def complete(
        self,
        messages: Sequence[dict[str, str]],
        settings: ModelSettings,
        allowed_answer_ids: Sequence[str],
    ) -> LlmQuestionResult:
        self.calls.append(list(messages))
        return LlmQuestionResult(
            selected_answer_id=allowed_answer_ids[-1],
            raw_response=allowed_answer_ids[-1],
            logprobs={"tokens": []},
        )


@pytest.mark.anyio
async def test_run_questionnaire_async_saves_records(tmp_path) -> None:
    settings = ModelSettings(
        model="gpt-4.1-mini",
        provider_base_url="https://api.openai.com/v1",
        temperature=0.2,
        timeout_seconds=60.0,
        capabilities=ProviderCapabilities(supports_logprobs=True),
    )
    client = FakeAsyncClient()

    records = await run_questionnaire_async(
        persona=TEST_PERSONA,
        questionnaire=BFI_10,
        settings=settings,
        client=client,
        project_root=tmp_path,
        experiment_id="pilot-study-one",
        context="Read this async vignette before answering.",
    )

    assert len(records) == len(BFI_10.items)
    assert records[0].status == ResponseStatus.COMPLETED
    assert isinstance(records[0].answer, LikertAnswerValue)
    assert records[0].answer.value == 5
    assert records[0].answer.label == "Strongly disagree"
    assert records[0].logprobs == {"tokens": []}
    assert "Additional context:" in client.calls[0][0]["content"]
    assert "Read this async vignette before answering." in client.calls[0][0]["content"]
    response_path = (
        tmp_path
        / "experiments"
        / "pilot-study-one"
        / records[0].run_id
        / "responses"
        / f"{TEST_PERSONA.persona_id}.jsonl"
    )
    assert response_path.read_text(encoding="utf-8").count("\n") == len(BFI_10.items)


def test_run_questionnaire_accepts_explicit_run_directory_id(tmp_path) -> None:
    run_id = "run-bfi10-llama3-1-20260603142709"
    settings = ModelSettings(
        model="llama3.1",
        provider_base_url="http://localhost:11434/v1",
        temperature=0.2,
        timeout_seconds=60.0,
    )

    records = run_questionnaire(
        run_id=run_id,
        persona=TEST_PERSONA,
        questionnaire=BFI_10,
        settings=settings,
        client=FakeSyncClient(),
        project_root=tmp_path,
        experiment_id="pilot-study-one",
    )

    assert records[0].run_id == run_id
    assert records[0].session_id.startswith("session-")


def test_run_questionnaire_reuses_session_for_multiple_runs(tmp_path) -> None:
    session_id = "session-00000000-0000-4000-8000-000000000002"
    first_run_id = "run-bfi10-llama3-1-20260603142709"
    second_run_id = "run-bfi10-llama3-1-20260603142710"
    settings = ModelSettings(
        model="llama3.1",
        provider_base_url="http://localhost:11434/v1",
        temperature=0.2,
        timeout_seconds=60.0,
    )

    first = run_questionnaire(
        run_id=first_run_id,
        session_id=session_id,
        persona=TEST_PERSONA,
        questionnaire=BFI_10,
        settings=settings,
        client=FakeSyncClient(),
        project_root=tmp_path,
        experiment_id="pilot-study-one",
    )
    second = run_questionnaire(
        run_id=second_run_id,
        session_id=session_id,
        persona=TEST_PERSONA,
        questionnaire=BFI_10,
        settings=settings,
        client=FakeSyncClient(),
        project_root=tmp_path,
        experiment_id="pilot-study-one",
    )

    assert first[0].session_id == session_id
    assert second[0].session_id == session_id
    experiment_root = tmp_path / "experiments" / "pilot-study-one"
    assert (
        experiment_root / first_run_id / "responses" / f"{TEST_PERSONA.persona_id}.jsonl"
    ).exists()
    assert (
        experiment_root / second_run_id / "responses" / f"{TEST_PERSONA.persona_id}.jsonl"
    ).exists()
    assert (experiment_root / first_run_id / "run.jsonl").exists()
    assert (experiment_root / second_run_id / "run.jsonl").exists()
    assert (experiment_root / "metadata.jsonl").read_text(encoding="utf-8").count("\n") == 2
