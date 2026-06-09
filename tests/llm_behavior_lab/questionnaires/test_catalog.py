import pytest

from llm_behavior_lab.questionnaires.catalog import (
    describe_questionnaire,
    list_questionnaires,
    questionnaire_ids,
    resolve_questionnaire,
)


def test_list_questionnaires_returns_complete_descriptors_in_stable_order() -> None:
    descriptors = list_questionnaires()

    assert [descriptor.id for descriptor in descriptors] == [
        "bfi_10",
        "consumer_involvement",
        "purchase_decision_making_inventory",
    ]
    assert questionnaire_ids() == [descriptor.id for descriptor in descriptors]
    assert all(descriptor.name for descriptor in descriptors)
    assert all(descriptor.version for descriptor in descriptors)
    assert all(descriptor.reference for descriptor in descriptors)
    assert all(descriptor.licence for descriptor in descriptors)
    assert all(descriptor.item_count > 0 for descriptor in descriptors)
    assert all(descriptor.response_format_types for descriptor in descriptors)


def test_describe_questionnaire_reports_scoring_and_response_metadata() -> None:
    descriptor = describe_questionnaire("bfi_10")

    assert descriptor.id == "bfi_10"
    assert descriptor.language == "en"
    assert descriptor.source_url
    assert descriptor.item_count == 10
    assert descriptor.scale_ids == [
        "extraversion",
        "agreeableness",
        "conscientiousness",
        "emotional_stability",
        "openness",
    ]
    assert descriptor.scoring_model_ids == ["default"]
    assert descriptor.response_format_types == ["likert"]
    assert descriptor.parameters == []


def test_describe_parameterized_questionnaire_reports_required_parameter() -> None:
    descriptor = describe_questionnaire("consumer_involvement")

    assert descriptor.item_count == 12
    assert descriptor.scoring_model_ids == []
    assert len(descriptor.parameters) == 1
    parameter = descriptor.parameters[0]
    assert parameter.name == "target"
    assert parameter.required is True
    assert parameter.description
    assert parameter.example == "meal delivery services"


def test_describe_questionnaire_rejects_unknown_id() -> None:
    with pytest.raises(ValueError, match="unknown questionnaire_id: pdmi"):
        describe_questionnaire("pdmi")


@pytest.mark.parametrize("questionnaire_id", ["bfi_10", "purchase_decision_making_inventory"])
def test_static_questionnaires_reject_parameters(questionnaire_id: str) -> None:
    with pytest.raises(
        ValueError,
        match=f"{questionnaire_id} does not accept questionnaire parameters",
    ):
        resolve_questionnaire(questionnaire_id, {"target": "coffee"})


def test_parameterized_questionnaire_rejects_missing_blank_and_unknown_parameters() -> None:
    with pytest.raises(
        ValueError,
        match="consumer_involvement requires questionnaire parameter 'target'",
    ):
        resolve_questionnaire("consumer_involvement")

    with pytest.raises(ValueError, match="target must be a non-empty"):
        resolve_questionnaire("consumer_involvement", {"target": "   "})

    with pytest.raises(
        ValueError,
        match="consumer_involvement received unknown questionnaire parameters: extra",
    ):
        resolve_questionnaire(
            "consumer_involvement",
            {"target": "coffee", "extra": "value"},
        )


def test_parameterized_questionnaire_normalizes_valid_parameter() -> None:
    questionnaire = resolve_questionnaire(
        "consumer_involvement",
        {"target": "  meal   delivery services "},
    )

    assert questionnaire.metadata["target"] == "meal delivery services"
