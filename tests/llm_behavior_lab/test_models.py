import pytest
from pydantic import ValidationError

from llm_behavior_lab.models import (
    AllowedAnswer,
    JsonlQuestionRecord,
    ModelSettings,
    Persona,
    ProviderCapabilities,
    Questionnaire,
    QuestionnaireQuestion,
)


def test_questionnaire_requires_allowed_answer_ids() -> None:
    questionnaire = Questionnaire(
        questionnaire_id="mini_big_five",
        name="Mini Big Five",
        questions=[
            QuestionnaireQuestion(
                question_id="q1",
                text="I see myself as someone who is talkative.",
                allowed_answers=[
                    AllowedAnswer(answer_id="1", label="Strongly disagree", score=1),
                    AllowedAnswer(answer_id="5", label="Strongly agree", score=5),
                ],
            )
        ],
        retain_history=True,
    )

    assert questionnaire.questions[0].allowed_answers[1].answer_id == "5"


def test_question_rejects_empty_allowed_answers() -> None:
    with pytest.raises(ValidationError):
        QuestionnaireQuestion(question_id="q1", text="Question", allowed_answers=[])


def test_jsonl_record_can_store_failure() -> None:
    record = JsonlQuestionRecord(
        run_id="run-1",
        persona=Persona(persona_id="p1", features={"age": "35", "country": "Poland"}),
        questionnaire_id="mini_big_five",
        question_id="q1",
        question_text="I see myself as someone who is talkative.",
        allowed_answers=[AllowedAnswer(answer_id="1", label="Strongly disagree", score=1)],
        messages=[{"role": "user", "content": "Question"}],
        model="llama3.1",
        provider_base_url="http://localhost:11434/v1",
        temperature=0.2,
        selected_answer_id=None,
        raw_response=None,
        structured_response=None,
        logprobs=None,
        error="model call failed",
    )

    assert record.error == "model call failed"


def test_provider_capabilities_defaults_to_local_safe_behavior() -> None:
    capabilities = ProviderCapabilities()

    assert capabilities.supports_structured_outputs is False
    assert capabilities.supports_logprobs is False


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
    assert settings.initial_backoff_seconds == 0.25
    assert settings.max_backoff_seconds == 4
    assert settings.max_concurrency == 8


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_attempts", 0),
        ("initial_backoff_seconds", -0.1),
        ("max_backoff_seconds", -0.1),
        ("max_concurrency", 0),
    ],
)
def test_model_settings_reject_invalid_execution_policy(
    field: str,
    value: int | float,
) -> None:
    values = {
        "model": "test",
        "provider_base_url": "http://localhost",
        "temperature": 0,
        "timeout_seconds": 10,
        field: value,
    }

    with pytest.raises(ValidationError):
        ModelSettings(**values)
