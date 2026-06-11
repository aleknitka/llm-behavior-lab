import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from llm_behavior_lab.responses.base import (
    ChatMessage,
    ItemResponseRecord,
    LikertAnswerValue,
    MultipleChoiceAnswerValue,
    NumericAnswerValue,
    ProviderSnapshot,
    ResponseStatus,
    RunRecord,
    SessionRecord,
    SingleChoiceAnswerValue,
    TextAnswerValue,
)


def test_answer_values_store_typed_response_data() -> None:
    likert = LikertAnswerValue(value=4, label="Agree")
    numeric = NumericAnswerValue(value=3.5, unit="hours")
    single_choice = SingleChoiceAnswerValue(
        option_id="a",
        label="First option",
        value=1,
    )
    multiple_choice = MultipleChoiceAnswerValue(
        option_ids=["a", "c"],
        labels=["First", "Third"],
        values=[1, 3],
    )
    text = TextAnswerValue(text="Free text answer")

    assert likert.type == "likert"
    assert numeric.model_dump() == {
        "type": "numeric",
        "value": 3.5,
        "unit": "hours",
    }
    assert single_choice.option_id == "a"
    assert multiple_choice.option_ids == ["a", "c"]
    assert text.text == "Free text answer"


def test_session_record_captures_batch_session_metadata() -> None:
    started_at = datetime(2026, 6, 3, 12, 0, tzinfo=UTC)

    session = SessionRecord(
        experiment_id="pilot-study-one",
        session_id="session-00000000-0000-4000-8000-000000000002",
        started_at=started_at,
        status=ResponseStatus.COMPLETED,
        run_count=100,
        metadata={"questionnaire_id": "bfi_10"},
    )

    assert session.experiment_id == "pilot-study-one"
    assert session.completed_at is None
    assert session.run_count == 100
    assert session.metadata["questionnaire_id"] == "bfi_10"


def test_response_status_supports_partial_and_cancelled() -> None:
    assert ResponseStatus("partial") is ResponseStatus.PARTIAL
    assert ResponseStatus("cancelled") is ResponseStatus.CANCELLED


def test_run_record_captures_persona_questionnaire_provider_metadata() -> None:
    started_at = datetime(2026, 6, 3, 12, 0, tzinfo=UTC)
    provider = ProviderSnapshot(
        provider_base_url="http://localhost:11434/v1",
        model="llama3.1",
        temperature=0.2,
        timeout_seconds=60.0,
        supports_structured_outputs=False,
        supports_logprobs=True,
        max_attempts=5,
        initial_backoff_seconds=0.5,
        max_backoff_seconds=8,
        max_concurrency=6,
    )

    run = RunRecord(
        experiment_id="pilot-study-one",
        session_id="session-00000000-0000-4000-8000-000000000002",
        run_id="run-00000000-0000-4000-8000-000000000001",
        subject_ids=["persona-1"],
        persona_count=1,
        questionnaire_id="bfi_10",
        questionnaire_shorthand="bfi10",
        questionnaire_version="1.0",
        model_slug="llama3-1",
        provider=provider,
        started_at=started_at,
        status=ResponseStatus.COMPLETED,
        error_count=0,
        item_count=10,
        output_paths={
            "run": "experiments/pilot-study-one/run-bfi10-llama3-1-20260603120000/run.json",
            "responses": "experiments/pilot-study-one/run-bfi10-llama3-1-20260603120000/responses",
            "scale": "experiments/pilot-study-one/run-bfi10-llama3-1-20260603120000/scale.json",
        },
        metadata={"scoring_model_id": None},
    )

    assert run.provider.model == "llama3.1"
    assert run.completed_at is None
    assert run.subject_ids == ["persona-1"]
    assert run.persona_count == 1
    assert run.questionnaire_shorthand == "bfi10"
    assert run.model_slug == "llama3-1"
    assert run.item_count == 10
    assert run.provider.max_attempts == 5
    assert run.provider.max_concurrency == 6


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_attempts", 0),
        ("initial_backoff_seconds", -0.1),
        ("max_backoff_seconds", -0.1),
        ("max_concurrency", 0),
    ],
)
def test_provider_snapshot_rejects_invalid_execution_policy(
    field: str,
    value: int | float,
) -> None:
    with pytest.raises(ValidationError):
        ProviderSnapshot(
            provider_base_url="http://localhost",
            model="test",
            temperature=0,
            timeout_seconds=10,
            **{field: value},
        )


def test_item_response_record_stores_raw_and_parsed_output() -> None:
    record = ItemResponseRecord(
        subject_id="subject-1",
        session_id="session-00000000-0000-4000-8000-000000000002",
        run_id="run-00000000-0000-4000-8000-000000000001",
        questionnaire_id="bfi_10",
        questionnaire_version="1.0",
        item_id="bfi10_01_reserved",
        item_order=1,
        item_text="I see myself as someone who is reserved",
        response_format_type="likert",
        messages=[
            ChatMessage(role="system", content="Assume the persona."),
            ChatMessage(role="user", content="Question text"),
        ],
        answer=LikertAnswerValue(value=2, label="Agree"),
        raw_response="2",
        structured_response={"selected_answer_id": "2"},
        logprobs={"tokens": []},
        status=ResponseStatus.COMPLETED,
        error=None,
        metadata={"prompt_template": "question_mcq"},
    )

    dumped = json.loads(record.model_dump_json())

    assert dumped["answer"] == {"type": "likert", "value": 2, "label": "Agree"}
    assert list(dumped)[:3] == ["subject_id", "session_id", "run_id"]
    assert dumped["messages"][0]["role"] == "system"
    assert dumped["structured_response"] == {"selected_answer_id": "2"}
    assert dumped["metadata"]["prompt_template"] == "question_mcq"


def test_item_response_record_parses_answer_by_discriminator() -> None:
    record = ItemResponseRecord.model_validate(
        {
            "subject_id": "subject-1",
            "session_id": "session-00000000-0000-4000-8000-000000000002",
            "run_id": "run-00000000-0000-4000-8000-000000000001",
            "questionnaire_id": "bfi_10",
            "questionnaire_version": "1.0",
            "item_id": "bfi10_01_reserved",
            "item_order": 1,
            "item_text": "I see myself as someone who is reserved",
            "response_format_type": "likert",
            "messages": [],
            "answer": {
                "type": "single_choice",
                "option_id": "2",
                "label": "Agree",
                "value": 2,
            },
            "raw_response": "2",
            "structured_response": None,
            "logprobs": None,
            "status": ResponseStatus.COMPLETED,
            "error": None,
        }
    )

    assert isinstance(record.answer, SingleChoiceAnswerValue)
    assert record.answer.option_id == "2"


def test_persisted_records_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ItemResponseRecord.model_validate(
            {
                "session_id": "session-00000000-0000-4000-8000-000000000002",
                "subject_id": "subject-1",
                "run_id": "run-00000000-0000-4000-8000-000000000001",
                "questionnaire_id": "bfi_10",
                "questionnaire_version": "1.0",
                "item_id": "bfi10_01_reserved",
                "item_order": 1,
                "item_text": "I see myself as someone who is reserved",
                "response_format_type": "likert",
                "messages": [],
                "answer": None,
                "raw_response": None,
                "structured_response": None,
                "logprobs": None,
                "status": ResponseStatus.FAILED,
                "error": "provider unavailable",
                "unexpected": "not allowed",
            }
        )
