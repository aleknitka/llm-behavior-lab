import json
from collections.abc import Sequence

from loguru import logger

from llm_psych_scales.models import LlmQuestionResult, ModelSettings
from llm_psych_scales.personas.factory import PersonaGenerationConfig, RequestedDemographicField
from llm_psych_scales.questionnaires.base.scale import Questionnaire
from llm_psych_scales.questionnaires.bfi10 import BFI_10
from llm_psych_scales.runner import run_persona_questionnaire_batch


class FakeBatchClient:
    def complete(
        self,
        messages: Sequence[dict[str, str]],
        settings: ModelSettings,
        allowed_answer_ids: Sequence[str],
    ) -> LlmQuestionResult:
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
    assert (experiment_root / "personas.jsonl").exists()
    assert (experiment_root / "metadata.jsonl").exists()
    assert not (experiment_root / "sessions").exists()

    metadata_rows = [
        json.loads(line)
        for line in (experiment_root / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(metadata_rows) == 1

    assert run_root.name.startswith("run-bfi10-openai-gpt-oss-20b-")
    assert {path.name for path in run_root.iterdir()} == {
        "run.jsonl",
        "responses",
        "scale.json",
    }
    assert (run_root / "run.jsonl").read_text(encoding="utf-8").count("\n") == 1
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
    responses_root = (
            tmp_path
            / "experiments"
            / result.experiment_id
            / run.run_id
            / "responses"
    )
    response_paths = sorted(responses_root.glob("*.jsonl"))
    assert len(response_paths) == 2

    for responses_path in response_paths:
        response_rows = [
            json.loads(line) for line in responses_path.read_text(encoding="utf-8").splitlines()
        ]
        assert responses_path.name == f"{response_rows[0]['subject_id']}.jsonl"
        assert {row["session_id"] for row in response_rows} == {result.session_id}
        assert {row["run_id"] for row in response_rows} == {run.run_id}
        assert {row["metadata"]["experiment_id"] for row in response_rows} == {
            result.experiment_id
        }
        assert list(response_rows[0])[:3] == ["subject_id", "session_id", "run_id"]


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

    run_path = (
        tmp_path
        / "experiments"
        / result.experiment_id
        / result.runs[0].run_id
        / "run.jsonl"
    )
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
    assert {persona.features.affluence_level for persona in result.personas.personas} == {
        "middle"
    }
