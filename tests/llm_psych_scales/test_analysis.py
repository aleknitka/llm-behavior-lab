import csv

from llm_psych_scales.analysis import load_response_table, write_response_table_csv
from llm_psych_scales.responses.base import (
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
