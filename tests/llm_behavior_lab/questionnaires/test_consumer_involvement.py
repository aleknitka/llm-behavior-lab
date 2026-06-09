import pytest

from llm_behavior_lab.questionnaires.base.response_formats import LikertFormat
from llm_behavior_lab.questionnaires.consumer_involvement import (
    CONSUMER_INVOLVEMENT_TEMPLATE,
    build_consumer_involvement_questionnaire,
)


def test_consumer_involvement_builder_rejects_blank_target() -> None:
    with pytest.raises(ValueError, match="target must be a non-empty"):
        build_consumer_involvement_questionnaire("   ")


def test_consumer_involvement_builder_fills_target_in_source_items() -> None:
    questionnaire = build_consumer_involvement_questionnaire("meal delivery services")

    assert questionnaire.id == "consumer_involvement"
    assert questionnaire.shorthand == "cinv"
    assert questionnaire.name == "Consumer Involvement Scale"
    assert questionnaire.metadata["target"] == "meal delivery services"
    assert questionnaire.metadata["source_doi"] == "10.1080/10696679.1999.11501855"
    assert questionnaire.metadata["source_table"] == "Table 2"
    assert len(questionnaire.items) == 12
    assert all("{target}" not in item.text for item in questionnaire.items)
    assert all("meal delivery services" in item.text for item in questionnaire.items)
    assert questionnaire.sections[0].item_ids == [item.id for item in questionnaire.items]


def test_consumer_involvement_uses_source_order_and_wording() -> None:
    questionnaire = build_consumer_involvement_questionnaire("coffee")

    assert [item.text for item in questionnaire.items] == [
        "I can tell a lot about a person by the type of coffee s/he chooses",
        "Buying coffee helps me express my personality",
        "Buying coffee requires a lot of thought",
        "It is extremely important that I make the right choice of coffee",
        "Choosing between coffee is a very important decision",
        "I attach great importance to coffee",
        "I have a strong interest in coffee",
        "I enjoy buying coffee",
        "All brands of coffee would not be equally enjoyable",
        "I believe that differing brands of coffee would give different amounts of pleasure",
        "In purchasing coffee I am certain of my choice",
        "It is really annoying to make an unsuitable purchase of coffee",
    ]
    assert [item.code for item in questionnaire.items] == [str(index) for index in range(1, 13)]


def test_consumer_involvement_uses_source_likert_format() -> None:
    questionnaire = build_consumer_involvement_questionnaire("coffee")
    response_format = questionnaire.items[0].response_format

    assert isinstance(response_format, LikertFormat)
    assert response_format.min_value == 1
    assert response_format.max_value == 5
    assert response_format.labels == {
        1: "Strongly agree",
        2: "Agree",
        3: "Neither agree nor disagree",
        4: "Disagree",
        5: "Strongly disagree",
    }
    assert all(item.response_format == response_format for item in questionnaire.items)


def test_consumer_involvement_scales_match_source_subconstructs() -> None:
    questionnaire = build_consumer_involvement_questionnaire("coffee")

    assert [scale.id for scale in questionnaire.scales] == [
        "normative",
        "situational",
        "enduring",
        "risk",
    ]
    assert {
        scale.id: [mapping.item_id for mapping in scale.item_mappings]
        for scale in questionnaire.scales
    } == {
        "normative": [
            "cinv_01_normative_type_choice",
            "cinv_02_normative_personality",
        ],
        "situational": [
            "cinv_03_situational_thought",
            "cinv_04_situational_right_choice",
            "cinv_05_situational_decision",
        ],
        "enduring": [
            "cinv_06_enduring_importance",
            "cinv_07_enduring_interest",
            "cinv_08_enduring_enjoy_buying",
        ],
        "risk": [
            "cinv_09_risk_brands_enjoyable",
            "cinv_10_risk_brand_pleasure",
            "cinv_11_risk_certain_choice",
            "cinv_12_risk_unsuitable_purchase",
        ],
    }


def test_consumer_involvement_factor_loadings_are_preserved_in_metadata() -> None:
    questionnaire = build_consumer_involvement_questionnaire("coffee")

    loadings = {
        mapping.item_id: mapping.metadata["factor_loading"]
        for scale in questionnaire.scales
        for mapping in scale.item_mappings
    }

    assert loadings == {
        "cinv_01_normative_type_choice": 0.93386,
        "cinv_02_normative_personality": 0.76836,
        "cinv_03_situational_thought": 0.82871,
        "cinv_04_situational_right_choice": 0.81888,
        "cinv_05_situational_decision": 0.77612,
        "cinv_06_enduring_importance": 0.85650,
        "cinv_07_enduring_interest": 0.85166,
        "cinv_08_enduring_enjoy_buying": 0.63860,
        "cinv_09_risk_brands_enjoyable": 0.82910,
        "cinv_10_risk_brand_pleasure": 0.68223,
        "cinv_11_risk_certain_choice": 0.59584,
        "cinv_12_risk_unsuitable_purchase": 0.58704,
    }


def test_consumer_involvement_template_keeps_target_placeholders() -> None:
    assert all("{target}" in item.text for item in CONSUMER_INVOLVEMENT_TEMPLATE.items)
    assert CONSUMER_INVOLVEMENT_TEMPLATE.metadata["is_template"] is True
