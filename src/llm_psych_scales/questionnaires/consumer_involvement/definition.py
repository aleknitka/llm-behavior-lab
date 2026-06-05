from __future__ import annotations

from llm_psych_scales.questionnaires.base.response_formats import LikertFormat
from llm_psych_scales.questionnaires.base.scale import (
    Item,
    ItemMapping,
    Questionnaire,
    Scale,
    Section,
)

CONSUMER_INVOLVEMENT_SOURCE_DOI = "10.1080/10696679.1999.11501855"
CONSUMER_INVOLVEMENT_SOURCE_URL = (
    "https://www.tandfonline.com/doi/abs/10.1080/10696679.1999.11501855"
)
CONSUMER_INVOLVEMENT_CITATION = (
    "Broderick, A. J., & Mueller, R. D. (1999). A theoretical and empirical "
    "exegesis of the consumer involvement construct: The psychology of the food "
    "shopper. Journal of Marketing Theory and Practice, 7(4), 97-108."
)
CONSUMER_INVOLVEMENT_LICENCE = "See source article terms."

CONSUMER_INVOLVEMENT_RESPONSE_FORMAT = LikertFormat(
    min_value=1,
    max_value=5,
    labels={
        1: "Strongly agree",
        2: "Agree",
        3: "Neither agree nor disagree",
        4: "Disagree",
        5: "Strongly disagree",
    },
)

_ITEM_DEFINITIONS = [
    (
        "cinv_01_normative_type_choice",
        "1",
        "normative",
        "I can tell a lot about a person by the type of {target} s/he chooses",
        0.93386,
    ),
    (
        "cinv_02_normative_personality",
        "2",
        "normative",
        "Buying {target} helps me express my personality",
        0.76836,
    ),
    (
        "cinv_03_situational_thought",
        "3",
        "situational",
        "Buying {target} requires a lot of thought",
        0.82871,
    ),
    (
        "cinv_04_situational_right_choice",
        "4",
        "situational",
        "It is extremely important that I make the right choice of {target}",
        0.81888,
    ),
    (
        "cinv_05_situational_decision",
        "5",
        "situational",
        "Choosing between {target} is a very important decision",
        0.77612,
    ),
    (
        "cinv_06_enduring_importance",
        "6",
        "enduring",
        "I attach great importance to {target}",
        0.85650,
    ),
    (
        "cinv_07_enduring_interest",
        "7",
        "enduring",
        "I have a strong interest in {target}",
        0.85166,
    ),
    (
        "cinv_08_enduring_enjoy_buying",
        "8",
        "enduring",
        "I enjoy buying {target}",
        0.63860,
    ),
    (
        "cinv_09_risk_brands_enjoyable",
        "9",
        "risk",
        "All brands of {target} would not be equally enjoyable",
        0.82910,
    ),
    (
        "cinv_10_risk_brand_pleasure",
        "10",
        "risk",
        "I believe that differing brands of {target} would give different amounts of pleasure",
        0.68223,
    ),
    (
        "cinv_11_risk_certain_choice",
        "11",
        "risk",
        "In purchasing {target} I am certain of my choice",
        0.59584,
    ),
    (
        "cinv_12_risk_unsuitable_purchase",
        "12",
        "risk",
        "It is really annoying to make an unsuitable purchase of {target}",
        0.58704,
    ),
]


def _items_with_target(target: str) -> list[Item]:
    return [
        Item(
            id=item_id,
            code=code,
            order=index,
            text=text.format(target=target),
            response_format=CONSUMER_INVOLVEMENT_RESPONSE_FORMAT,
            metadata={
                "subconstruct": subconstruct,
                "factor_loading": factor_loading,
                "source_table": "Table 2",
            },
        )
        for index, (item_id, code, subconstruct, text, factor_loading) in enumerate(
            _ITEM_DEFINITIONS, start=1
        )
    ]


def _scales() -> list[Scale]:
    scale_names = {
        "normative": (
            "Normative involvement",
            "The relevance of a product to an individual's values and emotions",
        ),
        "situational": (
            "Situational involvement",
            "Interest between brands or types of products at a point in time",
        ),
        "enduring": (
            "Enduring involvement",
            "Interest or familiarity with the product class as a whole",
        ),
        "risk": (
            "Risk involvement",
            "Importance or probability of making an incorrect product choice",
        ),
    }
    scales: list[Scale] = []
    for scale_id, (name, construct) in scale_names.items():
        mappings = [
            ItemMapping(
                item_id=item_id,
                metadata={
                    "factor_loading": factor_loading,
                    "source_factor": scale_id,
                    "source_table": "Table 2",
                },
            )
            for item_id, _code, subconstruct, _text, factor_loading in _ITEM_DEFINITIONS
            if subconstruct == scale_id
        ]
        scales.append(
            Scale(
                id=scale_id,
                name=name,
                construct=construct,
                item_mappings=mappings,
            )
        )
    return scales


def _questionnaire(target: str, *, is_template: bool) -> Questionnaire:
    items = _items_with_target(target)
    metadata = {
        "retain_history": True,
        "target": target,
        "target_kind": "product_or_service_category",
        "source_doi": CONSUMER_INVOLVEMENT_SOURCE_DOI,
        "source_url": CONSUMER_INVOLVEMENT_SOURCE_URL,
        "source_table": "Table 2",
        "source_likert_note": "1 indicates strong agreement; 5 indicates strong disagreement.",
        "is_template": is_template,
    }
    return Questionnaire(
        id="consumer_involvement",
        shorthand="cinv",
        name="Consumer Involvement Scale",
        version="1.0",
        language="en",
        sections=[
            Section(
                id="consumer_involvement_items",
                title="Consumer involvement items",
                item_ids=[item.id for item in items],
            )
        ],
        items=items,
        scales=_scales(),
        metadata=metadata,
        reference=CONSUMER_INVOLVEMENT_CITATION,
        licence=CONSUMER_INVOLVEMENT_LICENCE,
    )


CONSUMER_INVOLVEMENT_TEMPLATE = _questionnaire("{target}", is_template=True)


def build_consumer_involvement_questionnaire(target: str) -> Questionnaire:
    normalized_target = " ".join(target.split())
    if not normalized_target:
        raise ValueError("target must be a non-empty product or service category")
    return _questionnaire(normalized_target, is_template=False)
