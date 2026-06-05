from pathlib import Path

from llm_psych_scales.questionnaires import BFI_10
from llm_psych_scales.questionnaires.base.response_formats import LikertFormat
from llm_psych_scales.questionnaires.base.scale import Questionnaire
from llm_psych_scales.questionnaires.pdmi import PDMI, PURCHASE_DECISION_MAKING_INVENTORY

EMOTIONAL_SOURCE_ITEMS = [
    ("C56", "I bought compulsively."),
    ("C47", "I buy things without thinking about the effect on my economy."),
    ("C55", "When I do my shopping, I do not limit myself"),
    ("C50", "I shop excessively."),
    ("C57", "When I do my shopping, I think that if I want it I have it."),
    ("C59", "I avoid thinking about the consequences when making my purchases."),
    ("C51", "I decide fast without thinking, when I make purchases."),
    ("C58", "When I want something, I immediately buy it"),
    ("C60", "I have bought using instinct."),
    ("C45", "I have accumulated things that I have bought and do not need."),
    ("C46", "I have bought things that I like at the moment but not later."),
    ("C34", "I have generated debts when buying things."),
    ("C30", "I have spent money that I do not have when buying things."),
    ("C43", "I have delayed making payments for services"),
    ("C44", "I have had a service suspended for not paying on time."),
    ("C37", "It's common for me to end up in debt to satisfy my whims."),
    ("C31", "I have borrowed money to make a purchase."),
    ("C11", "When I'm depressed, I buy things."),
    ("C19", "When I'm angry, I spend more than I should."),
    ("C14", "When I'm desperate I buy the first thing that I see."),
    ("C20", "I have bought things on a whim."),
    ("C2", "I have bought by anxiety."),
    ("C15", "My purchase is initiated by emotional needs."),
    ("C9", "I have bought things due to an outburst."),
    ("C5", "It anguishes me to have to decide between several products."),
    ("C10", "It frustrates me to have to decide between several products."),
    ("C18", "It makes me nervous to choose between different products."),
    ("C7", "Buying is a pleasant activity for me."),
    ("C8", "Shopping is one of the activities I enjoy most in life."),
    ("C12", "I feel happy when I buy things."),
]

REASONED_SOURCE_ITEMS = [
    ("C82", "I take care every spend that I do."),
    ("C83", "I control every expense before buying."),
    ("C84", "When purchasing things, I look for products that help me economize."),
    ("C74", "I organize every expense that I carry out."),
    ("C85", "When I do my shopping, I spend the right amount."),
    ("C80", "When choosing what to buy, I make choices taking into consideration my economy."),
    ("C91", "I make my purchases carefully to save money."),
    ("C88", "I avoid buying things that aren't on my shopping list."),
    ("C89", "I avoid buying something so as not to get into debt."),
    ("C96", "I previously list the products that I need before buying them."),
    ("C64", "I think about every purchase that I make."),
    ("C63", "I'm analytical when I make my purchases."),
    ("C65", "I reason with myself before buying something."),
    ("C66", "I decide calmly when buying a product."),
    ("C62", "I consciously decide what to buy when I make a purchase."),
    ("C67", "When I go shopping, I only buy the things that I had willing to buy."),
    ("C104", "I am informed about the products before buying them."),
    ("C102", "I ask about the products before deciding to buy them."),
    ("C97", "I investigate information about the products before buying them."),
    ("C105", "I'm critical when deciding between which products to buy."),
]


def test_pdmi_validates_and_exports_alias() -> None:
    questionnaire = PURCHASE_DECISION_MAKING_INVENTORY

    assert isinstance(questionnaire, Questionnaire)
    assert PDMI is questionnaire
    assert Questionnaire.model_validate(questionnaire.model_dump()) == questionnaire
    assert questionnaire.id == "purchase_decision_making_inventory"
    assert questionnaire.shorthand == "pdmi"
    assert questionnaire.name == "Purchase Decision-Making Inventory"
    assert questionnaire.version == "1.0"
    assert questionnaire.language == "en"


def test_pdmi_sections_match_source_order() -> None:
    questionnaire = PURCHASE_DECISION_MAKING_INVENTORY

    assert len(questionnaire.items) == 50
    assert [section.id for section in questionnaire.sections] == [
        "pdmi_emotional_items",
        "pdmi_reasoned_items",
    ]
    assert questionnaire.sections[0].item_ids == [item.id for item in questionnaire.items[:30]]
    assert questionnaire.sections[1].item_ids == [item.id for item in questionnaire.items[30:]]
    assert [item.order for item in questionnaire.items] == list(range(1, 51))


def test_pdmi_uses_source_likert_frequency_scale() -> None:
    questionnaire = PURCHASE_DECISION_MAKING_INVENTORY
    response_format = questionnaire.items[0].response_format

    assert isinstance(response_format, LikertFormat)
    assert response_format.min_value == 1
    assert response_format.max_value == 5
    assert response_format.labels == {
        1: "Never",
        2: "Rarely",
        3: "Sometimes",
        4: "Often",
        5: "Always",
    }
    assert all(item.response_format == response_format for item in questionnaire.items)


def test_pdmi_item_wording_and_source_codes_match_tables_2_and_3() -> None:
    questionnaire = PURCHASE_DECISION_MAKING_INVENTORY

    assert [(item.code, item.text) for item in questionnaire.items[:30]] == EMOTIONAL_SOURCE_ITEMS
    assert [(item.code, item.text) for item in questionnaire.items[30:]] == REASONED_SOURCE_ITEMS
    assert {item.metadata["source_table"] for item in questionnaire.items[:30]} == {"Table 2"}
    assert {item.metadata["source_table"] for item in questionnaire.items[30:]} == {"Table 3"}


def test_pdmi_scales_match_source_subscales() -> None:
    questionnaire = PURCHASE_DECISION_MAKING_INVENTORY

    assert [scale.id for scale in questionnaire.scales] == [
        "impulsivity",
        "indebtedness",
        "negative_emotions",
        "frustration",
        "hedonism",
        "saving",
        "reasoning",
        "search_of_information",
    ]
    assert {
        scale.id: [mapping.item_id for mapping in scale.item_mappings]
        for scale in questionnaire.scales
    } == {
        "impulsivity": [f"pdmi_{index:02d}" for index in range(1, 12)],
        "indebtedness": [f"pdmi_{index:02d}" for index in range(12, 18)],
        "negative_emotions": [f"pdmi_{index:02d}" for index in range(18, 25)],
        "frustration": [f"pdmi_{index:02d}" for index in range(25, 28)],
        "hedonism": [f"pdmi_{index:02d}" for index in range(28, 31)],
        "saving": [f"pdmi_{index:02d}" for index in range(31, 41)],
        "reasoning": [f"pdmi_{index:02d}" for index in range(41, 47)],
        "search_of_information": [f"pdmi_{index:02d}" for index in range(47, 51)],
    }
    assert {scale.id: scale.metadata["cronbach_alpha"] for scale in questionnaire.scales} == {
        "impulsivity": 0.883,
        "indebtedness": 0.825,
        "negative_emotions": 0.82,
        "frustration": 0.851,
        "hedonism": 0.819,
        "saving": 0.895,
        "reasoning": 0.879,
        "search_of_information": 0.841,
    }
    assert all(
        scale.metadata["source_article"] == questionnaire.reference
        for scale in questionnaire.scales
    )
    assert all(scale.metadata["licence"] == questionnaire.licence for scale in questionnaire.scales)


def test_pdmi_preserves_representative_factor_loading_metadata() -> None:
    questionnaire = PURCHASE_DECISION_MAKING_INVENTORY
    loadings = {
        mapping.item_id: mapping.metadata["factor_loading"]
        for scale in questionnaire.scales
        for mapping in scale.item_mappings
    }

    assert loadings["pdmi_01"] == 0.656
    assert loadings["pdmi_11"] == 0.454
    assert loadings["pdmi_12"] == 0.733
    assert loadings["pdmi_17"] == 0.567
    assert loadings["pdmi_18"] == 0.595
    assert loadings["pdmi_24"] == 0.471
    assert loadings["pdmi_25"] == 0.810
    assert loadings["pdmi_27"] == 0.707
    assert loadings["pdmi_28"] == 0.783
    assert loadings["pdmi_30"] == 0.644
    assert loadings["pdmi_31"] == 0.790
    assert loadings["pdmi_40"] == 0.448
    assert loadings["pdmi_41"] == 0.903
    assert loadings["pdmi_46"] == 0.553
    assert loadings["pdmi_47"] == 0.908
    assert loadings["pdmi_50"] == 0.593
    assert questionnaire.metadata["source_doi"] == "10.14349/sumapsi.2019.v26.n2.3"
    assert questionnaire.metadata["emotional_variance_explained"] == 48.75
    assert questionnaire.metadata["reasoned_variance_explained"] == 53.72
    assert questionnaire.scoring_models == []
    assert questionnaire.metadata["scoring_models_status"] == "deferred"


def test_pdmi_readme_exists_and_root_readme_links_to_it() -> None:
    questionnaire_readme = Path("src/llm_psych_scales/questionnaires/pdmi/README.md")
    root_readme = Path("README.md")
    root_text = root_readme.read_text()

    assert questionnaire_readme.exists()
    assert "10.14349/sumapsi.2019.v26.n2.3" in questionnaire_readme.read_text()
    assert "questionnaires/pdmi/README.md" in root_text
    assert root_text.index("questionnaires/pdmi/README.md") > root_text.index(
        "## Questionnaire Definitions"
    )
    assert "BFI-10 is the first coded questionnaire" not in root_text


def test_questionnaires_package_exports_static_questionnaires() -> None:
    assert BFI_10.shorthand == "bfi10"
    assert PDMI.shorthand == "pdmi"
