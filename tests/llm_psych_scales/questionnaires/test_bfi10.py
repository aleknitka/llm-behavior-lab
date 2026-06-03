import pytest
from pydantic import ValidationError

from llm_psych_scales.questionnaires.base.response_formats import LikertFormat
from llm_psych_scales.questionnaires.base.scale import Questionnaire
from llm_psych_scales.questionnaires.bfi10 import BFI_10


def test_bfi10_uses_source_order_and_wording() -> None:
    assert isinstance(BFI_10, Questionnaire)
    assert BFI_10.id == "bfi_10"
    assert BFI_10.shorthand == "bfi10"
    assert BFI_10.name == "Big Five Inventory 10 Item Scale"
    assert BFI_10.version == "1.0"
    assert BFI_10.sections[0].item_ids == [item.id for item in BFI_10.items]
    assert [item.text for item in BFI_10.items] == [
        "I see myself as someone who is reserved",
        "I see myself as someone who is generally trusting",
        "I see myself as someone who does a thorough job",
        "I see myself as someone who is relaxed, handles stress well",
        "I see myself as someone who has an active imagination",
        "I see myself as someone who is outgoing, sociable",
        "I see myself as someone who tends to find fault with others",
        "I see myself as someone who tends to be lazy",
        "I see myself as someone who gets nervous easily",
        "I see myself as someone who has few artistic interests",
    ]


def test_bfi10_uses_source_response_order() -> None:
    response_format = BFI_10.items[0].response_format

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
    assert all(item.response_format == response_format for item in BFI_10.items)


def test_bfi10_includes_citation_and_item_metadata() -> None:
    assert "Rammstedt" in BFI_10.reference
    assert BFI_10.metadata["source_url"].endswith("BFI-10_English_Items.pdf")
    assert BFI_10.licence == "See source instrument terms."
    assert [
        (item.metadata["trait"], item.metadata["reverse_scored"]) for item in BFI_10.items
    ] == [
        ("extraversion", True),
        ("agreeableness", False),
        ("conscientiousness", False),
        ("emotional_stability", False),
        ("openness", False),
        ("extraversion", False),
        ("agreeableness", True),
        ("conscientiousness", True),
        ("emotional_stability", True),
        ("openness", True),
    ]


@pytest.mark.parametrize("shorthand", ["bf", "abcdefgh", "bfi-10", "BFI10"])
def test_questionnaire_rejects_invalid_shorthand(shorthand: str) -> None:
    with pytest.raises(ValidationError):
        Questionnaire(
            id="example",
            shorthand=shorthand,
            name="Example",
            version="1.0",
            sections=[],
            items=[],
            reference="Reference",
            licence="Licence",
        )
