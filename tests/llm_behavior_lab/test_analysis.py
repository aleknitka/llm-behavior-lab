import csv

from llm_behavior_lab.analysis import load_response_table, write_response_table_csv
from llm_behavior_lab.personas.factory import (
    PersonaFactory,
    PersonaFactoryRequest,
    RequestedDemographicField,
)
from llm_behavior_lab.responses.base import (
    ChatMessage,
    ItemResponseRecord,
    LikertAnswerValue,
    ResponseStatus,
    TextAnswerValue,
)


def _write_response(path, record: ItemResponseRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(record.model_dump_json() + "\n")


def test_load_response_table_flattens_response_directory(tmp_path) -> None:
    responses_root = tmp_path / "run-bfi10-model-20260603120000" / "responses"
    _write_response(
        responses_root / "subject-1.jsonl",
        ItemResponseRecord(
            subject_id="subject-1",
            session_id="session-1",
            run_id="run-bfi10-model-20260603120000",
            questionnaire_id="bfi_10",
            questionnaire_version="1.0",
            item_id="bfi10_01_reserved",
            item_order=1,
            item_text="I see myself as someone who is reserved",
            response_format_type="likert",
            messages=[ChatMessage(role="user", content="Question")],
            answer=LikertAnswerValue(value=2, label="Agree"),
            raw_response="2",
            structured_response={"selected_answer_id": "2"},
            logprobs={"content": []},
            status=ResponseStatus.COMPLETED,
            error=None,
            metadata={"experiment_id": "pilot-study-one"},
        ),
    )
    _write_response(
        responses_root / "subject-2.jsonl",
        ItemResponseRecord(
            subject_id="subject-2",
            session_id="session-1",
            run_id="run-bfi10-model-20260603120000",
            questionnaire_id="bfi_10",
            questionnaire_version="1.0",
            item_id="free_text",
            item_order=2,
            item_text="Explain",
            response_format_type="text",
            messages=[],
            answer=TextAnswerValue(text="hello"),
            raw_response="hello",
            structured_response=None,
            logprobs=None,
            status=ResponseStatus.COMPLETED,
            error=None,
            metadata={"experiment_id": "pilot-study-one"},
        ),
    )

    rows = load_response_table(responses_root)

    assert rows == [
        {
            "subject_id": "subject-1",
            "session_id": "session-1",
            "run_id": "run-bfi10-model-20260603120000",
            "questionnaire_id": "bfi_10",
            "questionnaire_version": "1.0",
            "item_id": "bfi10_01_reserved",
            "item_order": 1,
            "item_text": "I see myself as someone who is reserved",
            "response_format_type": "likert",
            "answer_type": "likert",
            "answer_value": 2,
            "answer_label": "Agree",
            "raw_response": "2",
            "status": "completed",
            "error": None,
            "logprobs_available": True,
            "experiment_id": "pilot-study-one",
        },
        {
            "subject_id": "subject-2",
            "session_id": "session-1",
            "run_id": "run-bfi10-model-20260603120000",
            "questionnaire_id": "bfi_10",
            "questionnaire_version": "1.0",
            "item_id": "free_text",
            "item_order": 2,
            "item_text": "Explain",
            "response_format_type": "text",
            "answer_type": "text",
            "answer_value": "hello",
            "answer_label": None,
            "raw_response": "hello",
            "status": "completed",
            "error": None,
            "logprobs_available": False,
            "experiment_id": "pilot-study-one",
        },
    ]


def test_load_response_table_accepts_run_directory(tmp_path) -> None:
    run_root = tmp_path / "run-bfi10-model-20260603120000"
    _write_response(
        run_root / "responses" / "subject-1.jsonl",
        ItemResponseRecord(
            subject_id="subject-1",
            session_id="session-1",
            run_id=run_root.name,
            questionnaire_id="bfi_10",
            questionnaire_version="1.0",
            item_id="bfi10_01_reserved",
            item_order=1,
            item_text="Question",
            response_format_type="likert",
            messages=[],
            answer=LikertAnswerValue(value=1, label="Strongly agree"),
            raw_response="1",
            structured_response=None,
            logprobs=None,
            status=ResponseStatus.COMPLETED,
            error=None,
            metadata={},
        ),
    )

    assert len(load_response_table(run_root)) == 1


def test_load_response_table_uses_latest_item_attempt(tmp_path) -> None:
    responses_root = tmp_path / "run-bfi10-model-20260603120000" / "responses"
    failed = ItemResponseRecord(
        subject_id="subject-1",
        session_id="session-1",
        run_id="run-bfi10-model-20260603120000",
        questionnaire_id="bfi_10",
        questionnaire_version="1.0",
        item_id="bfi10_01_reserved",
        item_order=1,
        item_text="Question",
        response_format_type="likert",
        messages=[],
        status=ResponseStatus.FAILED,
        error="provider unavailable",
    )
    completed = failed.model_copy(
        update={
            "answer": LikertAnswerValue(value=2, label="Agree"),
            "raw_response": "2",
            "status": ResponseStatus.COMPLETED,
            "error": None,
        }
    )
    _write_response(responses_root / "subject-1.jsonl", failed)
    _write_response(responses_root / "subject-1.jsonl", completed)

    rows = load_response_table(responses_root)

    assert len(rows) == 1
    assert rows[0]["status"] == "completed"
    assert rows[0]["answer_value"] == 2


def test_load_response_table_joins_persona_features_by_subject_id(tmp_path) -> None:
    experiment_root = tmp_path / "experiments" / "join-study-one"
    run_root = experiment_root / "run-bfi10-model-20260603120000"
    batch = PersonaFactory().create_demographics_batch(
        PersonaFactoryRequest(
            count=1,
            requested_fields={
                RequestedDemographicField.AGE,
                RequestedDemographicField.COUNTRY,
            },
            seed=7,
            experiment_id="join-study-one",
        )
    )
    subject_id = str(batch.personas[0].subject_id)
    (experiment_root / "personas.json").parent.mkdir(parents=True)
    (experiment_root / "personas.json").write_text(batch.model_dump_json())
    _write_response(
        run_root / "responses" / f"{subject_id}.jsonl",
        ItemResponseRecord(
            subject_id=subject_id,
            session_id="session-1",
            run_id=run_root.name,
            questionnaire_id="bfi_10",
            questionnaire_version="1.0",
            item_id="bfi10_01_reserved",
            item_order=1,
            item_text="Question",
            response_format_type="likert",
            messages=[],
            answer=LikertAnswerValue(value=1),
            raw_response="1",
            status=ResponseStatus.COMPLETED,
        ),
    )

    rows = load_response_table(run_root)

    assert rows[0]["persona_age"] == batch.personas[0].features.age
    assert rows[0]["persona_country"] == batch.personas[0].features.country


def test_load_response_table_joins_protocol_step_to_cohort_persona(tmp_path) -> None:
    experiment_root = tmp_path / "experiments" / "join-protocol-one"
    protocol_run_root = experiment_root / "run-protocol-model-20260603120000"
    step_root = protocol_run_root / "steps" / "questionnaire"
    cohort_root = experiment_root / "cohorts" / "cohort-test"
    batch = PersonaFactory().create_demographics_batch(
        PersonaFactoryRequest(
            count=1,
            requested_fields={RequestedDemographicField.AGE},
            seed=8,
            experiment_id="join-protocol-one",
        )
    )
    subject_id = str(batch.personas[0].subject_id)
    cohort_root.mkdir(parents=True)
    (cohort_root / "personas.json").write_text(batch.model_dump_json())
    (experiment_root / "protocol.json").write_text("{}")
    protocol_run_root.mkdir(parents=True)
    (protocol_run_root / "run.json").write_text(
        '{"metadata":{"cohort_id":"cohort-test"}}'
    )
    _write_response(
        step_root / "responses" / f"{subject_id}.jsonl",
        ItemResponseRecord(
            subject_id=subject_id,
            session_id="session-1",
            run_id=protocol_run_root.name,
            questionnaire_id="bfi_10",
            questionnaire_version="1.0",
            item_id="item-1",
            item_order=1,
            item_text="Question",
            response_format_type="likert",
            messages=[],
            answer=LikertAnswerValue(value=1),
            status=ResponseStatus.COMPLETED,
        ),
    )

    rows = load_response_table(step_root)

    assert rows[0]["persona_age"] == batch.personas[0].features.age


def test_write_response_table_csv_writes_stable_columns(tmp_path) -> None:
    rows = [
        {
            "subject_id": "subject-1",
            "session_id": "session-1",
            "run_id": "run-1",
            "questionnaire_id": "bfi_10",
            "questionnaire_version": "1.0",
            "item_id": "item-1",
            "item_order": 1,
            "item_text": "Question",
            "response_format_type": "likert",
            "answer_type": "likert",
            "answer_value": 1,
            "answer_label": "Strongly agree",
            "raw_response": "1",
            "status": "completed",
            "error": None,
            "logprobs_available": False,
            "experiment_id": "pilot-study-one",
        }
    ]
    output_path = tmp_path / "responses.csv"

    write_response_table_csv(rows, output_path)

    with output_path.open(encoding="utf-8", newline="") as file:
        csv_rows = list(csv.DictReader(file))
    assert csv_rows[0]["subject_id"] == "subject-1"
    assert csv_rows[0]["answer_value"] == "1"


def test_load_response_table_includes_protocol_columns(tmp_path) -> None:
    responses_root = tmp_path / "run-bfi10-model-20260603120000" / "responses"
    _write_response(
        responses_root / "subject-1.jsonl",
        ItemResponseRecord(
            subject_id="subject-1",
            session_id="session-1",
            run_id="run-bfi10-model-20260603120000",
            questionnaire_id="bfi_10",
            questionnaire_version="1.0",
            item_id="bfi10_01_reserved",
            item_order=1,
            item_text="Question",
            response_format_type="likert",
            messages=[],
            answer=LikertAnswerValue(value=1, label="Strongly agree"),
            raw_response="1",
            structured_response=None,
            logprobs=None,
            status=ResponseStatus.COMPLETED,
            error=None,
            metadata={
                "experiment_id": "proto-study-one",
                "protocol_name": "gender-affluence-factorial",
                "base_subject_id": "base-1",
                "condition_id": "gender-female__affluence-low",
                "iteration_index": 2,
                "factor_values": {
                    "gender": "female",
                    "affluence_level": "low",
                },
            },
        ),
    )

    rows = load_response_table(responses_root)

    assert rows[0]["protocol_name"] == "gender-affluence-factorial"
    assert rows[0]["base_subject_id"] == "base-1"
    assert rows[0]["condition_id"] == "gender-female__affluence-low"
    assert rows[0]["iteration_index"] == 2
    assert rows[0]["factor_gender"] == "female"
    assert rows[0]["factor_affluence_level"] == "low"
