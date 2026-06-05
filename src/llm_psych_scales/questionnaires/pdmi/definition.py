from llm_psych_scales.questionnaires.base.response_formats import LikertFormat
from llm_psych_scales.questionnaires.base.scale import (
    Item,
    ItemMapping,
    Questionnaire,
    Scale,
    Section,
)

PDMI_SOURCE_DOI = "10.14349/sumapsi.2019.v26.n2.3"
PDMI_SOURCE_URL = "http://dx.doi.org/10.14349/sumapsi.2019.v26.n2.3"
PDMI_CITATION = (
    "Soler Anguiano, F. L., Bustos Aguayo, J. M., Palacios, J., Zeelenberg, M., "
    "& Diaz Loving, R. (2019). Development and validation of the Inventory of "
    "Emotional and Reasoned Purchases Decision-Making Styles (PDMI). "
    "Suma Psicologica, 26(2), 75-85."
)
PDMI_LICENCE = "CC BY-NC-ND 4.0"

PDMI_RESPONSE_FORMAT = LikertFormat(
    min_value=1,
    max_value=5,
    labels={
        1: "Never",
        2: "Rarely",
        3: "Sometimes",
        4: "Often",
        5: "Always",
    },
)

_EMOTIONAL_ITEMS = [
    (
        "C56",
        "impulsivity",
        "I bought compulsively.",
        0.656,
    ),
    (
        "C47",
        "impulsivity",
        "I buy things without thinking about the effect on my economy.",
        0.628,
    ),
    (
        "C55",
        "impulsivity",
        "When I do my shopping, I do not limit myself",
        0.598,
    ),
    ("C50", "impulsivity", "I shop excessively.", 0.577),
    (
        "C57",
        "impulsivity",
        "When I do my shopping, I think that if I want it I have it.",
        0.574,
    ),
    (
        "C59",
        "impulsivity",
        "I avoid thinking about the consequences when making my purchases.",
        0.569,
    ),
    (
        "C51",
        "impulsivity",
        "I decide fast without thinking, when I make purchases.",
        0.551,
    ),
    ("C58", "impulsivity", "When I want something, I immediately buy it", 0.549),
    ("C60", "impulsivity", "I have bought using instinct.", 0.538),
    (
        "C45",
        "impulsivity",
        "I have accumulated things that I have bought and do not need.",
        0.519,
    ),
    (
        "C46",
        "impulsivity",
        "I have bought things that I like at the moment but not later.",
        0.454,
    ),
    (
        "C34",
        "indebtedness",
        "I have generated debts when buying things.",
        0.733,
    ),
    (
        "C30",
        "indebtedness",
        "I have spent money that I do not have when buying things.",
        0.686,
    ),
    ("C43", "indebtedness", "I have delayed making payments for services", 0.609),
    (
        "C44",
        "indebtedness",
        "I have had a service suspended for not paying on time.",
        0.598,
    ),
    (
        "C37",
        "indebtedness",
        "It's common for me to end up in debt to satisfy my whims.",
        0.569,
    ),
    (
        "C31",
        "indebtedness",
        "I have borrowed money to make a purchase.",
        0.567,
    ),
    ("C11", "negative_emotions", "When I'm depressed, I buy things.", 0.595),
    (
        "C19",
        "negative_emotions",
        "When I'm angry, I spend more than I should.",
        0.542,
    ),
    (
        "C14",
        "negative_emotions",
        "When I'm desperate I buy the first thing that I see.",
        0.536,
    ),
    (
        "C20",
        "negative_emotions",
        "I have bought things on a whim.",
        0.527,
    ),
    ("C2", "negative_emotions", "I have bought by anxiety.", 0.507),
    (
        "C15",
        "negative_emotions",
        "My purchase is initiated by emotional needs.",
        0.506,
    ),
    (
        "C9",
        "negative_emotions",
        "I have bought things due to an outburst.",
        0.471,
    ),
    (
        "C5",
        "frustration",
        "It anguishes me to have to decide between several products.",
        0.810,
    ),
    (
        "C10",
        "frustration",
        "It frustrates me to have to decide between several products.",
        0.787,
    ),
    (
        "C18",
        "frustration",
        "It makes me nervous to choose between different products.",
        0.707,
    ),
    ("C7", "hedonism", "Buying is a pleasant activity for me.", 0.783),
    (
        "C8",
        "hedonism",
        "Shopping is one of the activities I enjoy most in life.",
        0.774,
    ),
    ("C12", "hedonism", "I feel happy when I buy things.", 0.644),
]

_REASONED_ITEMS = [
    ("C82", "saving", "I take care every spend that I do.", 0.790),
    ("C83", "saving", "I control every expense before buying.", 0.777),
    (
        "C84",
        "saving",
        "When purchasing things, I look for products that help me economize.",
        0.696,
    ),
    ("C74", "saving", "I organize every expense that I carry out.", 0.689),
    (
        "C85",
        "saving",
        "When I do my shopping, I spend the right amount.",
        0.647,
    ),
    (
        "C80",
        "saving",
        "When choosing what to buy, I make choices taking into consideration my economy.",
        0.643,
    ),
    ("C91", "saving", "I make my purchases carefully to save money.", 0.623),
    (
        "C88",
        "saving",
        "I avoid buying things that aren't on my shopping list.",
        0.462,
    ),
    (
        "C89",
        "saving",
        "I avoid buying something so as not to get into debt.",
        0.454,
    ),
    (
        "C96",
        "saving",
        "I previously list the products that I need before buying them.",
        0.448,
    ),
    ("C64", "reasoning", "I think about every purchase that I make.", 0.903),
    ("C63", "reasoning", "I'm analytical when I make my purchases.", 0.770),
    (
        "C65",
        "reasoning",
        "I reason with myself before buying something.",
        0.756,
    ),
    ("C66", "reasoning", "I decide calmly when buying a product.", 0.643),
    (
        "C62",
        "reasoning",
        "I consciously decide what to buy when I make a purchase.",
        0.565,
    ),
    (
        "C67",
        "reasoning",
        "When I go shopping, I only buy the things that I had willing to buy.",
        0.553,
    ),
    (
        "C104",
        "search_of_information",
        "I am informed about the products before buying them.",
        0.908,
    ),
    (
        "C102",
        "search_of_information",
        "I ask about the products before deciding to buy them.",
        0.687,
    ),
    (
        "C97",
        "search_of_information",
        "I investigate information about the products before buying them.",
        0.621,
    ),
    (
        "C105",
        "search_of_information",
        "I'm critical when deciding between which products to buy.",
        0.593,
    ),
]

_SCALE_DETAILS = {
    "impulsivity": {
        "name": "Impulsivity",
        "construct": "Emotional purchase decision-making",
        "description": (
            "Buying without thinking about the economy, and buying at the time "
            "the product is desired."
        ),
        "source_scale": "PDMI-Emotional",
        "source_table": "Table 2",
        "cronbach_alpha": 0.883,
        "variance_explained": 14.26,
    },
    "indebtedness": {
        "name": "Indebtedness",
        "construct": "Emotional purchase decision-making",
        "description": (
            "Generating debts, spending money one does not have, and borrowing "
            "money to satisfy whims."
        ),
        "source_scale": "PDMI-Emotional",
        "source_table": "Table 2",
        "cronbach_alpha": 0.825,
        "variance_explained": 9.98,
    },
    "negative_emotions": {
        "name": "Negative emotions",
        "construct": "Emotional purchase decision-making",
        "description": "Purchasing associated with negative emotional states.",
        "source_scale": "PDMI-Emotional",
        "source_table": "Table 2",
        "cronbach_alpha": 0.82,
        "variance_explained": 9.79,
    },
    "frustration": {
        "name": "Frustration",
        "construct": "Emotional purchase decision-making",
        "description": (
            "Anguish, frustration, and nervousness when choosing between different products."
        ),
        "source_scale": "PDMI-Emotional",
        "source_table": "Table 2",
        "cronbach_alpha": 0.851,
        "variance_explained": 7.36,
    },
    "hedonism": {
        "name": "Hedonism",
        "construct": "Emotional purchase decision-making",
        "description": "Shopping for pleasure, happiness, and enjoyment.",
        "source_scale": "PDMI-Emotional",
        "source_table": "Table 2",
        "cronbach_alpha": 0.819,
        "variance_explained": 7.34,
    },
    "saving": {
        "name": "Saving",
        "construct": "Reasoned purchase decision-making",
        "description": (
            "Purchasing consciously and analysing expenses for the benefit of "
            "one's personal economy."
        ),
        "source_scale": "PDMI-Reasoned",
        "source_table": "Table 3",
        "cronbach_alpha": 0.895,
        "variance_explained": 22.78,
    },
    "reasoning": {
        "name": "Reasoning",
        "construct": "Reasoned purchase decision-making",
        "description": (
            "Processing, being analytical, and calmly and consciously deciding each purchase."
        ),
        "source_scale": "PDMI-Reasoned",
        "source_table": "Table 3",
        "cronbach_alpha": 0.879,
        "variance_explained": 18.14,
    },
    "search_of_information": {
        "name": "Search of information",
        "construct": "Reasoned purchase decision-making",
        "description": (
            "Consulting, being critical, and finding out about products before making the purchase."
        ),
        "source_scale": "PDMI-Reasoned",
        "source_table": "Table 3",
        "cronbach_alpha": 0.841,
        "variance_explained": 12.78,
    },
}


def _items() -> list[Item]:
    definitions = [*_EMOTIONAL_ITEMS, *_REASONED_ITEMS]
    items: list[Item] = []
    for order, (code, scale_id, text, factor_loading) in enumerate(definitions, start=1):
        source_table = "Table 2" if order <= len(_EMOTIONAL_ITEMS) else "Table 3"
        items.append(
            Item(
                id=f"pdmi_{order:02d}",
                code=code,
                order=order,
                text=text,
                response_format=PDMI_RESPONSE_FORMAT,
                metadata={
                    "source_scale": _SCALE_DETAILS[scale_id]["source_scale"],
                    "source_subscale": scale_id,
                    "source_table": source_table,
                    "factor_loading": factor_loading,
                },
            )
        )
    return items


def _scales(items: list[Item]) -> list[Scale]:
    scales: list[Scale] = []
    for scale_id, details in _SCALE_DETAILS.items():
        mappings = [
            ItemMapping(
                item_id=item.id,
                metadata={
                    "factor_loading": item.metadata["factor_loading"],
                    "source_scale": item.metadata["source_scale"],
                    "source_subscale": scale_id,
                    "source_table": item.metadata["source_table"],
                },
            )
            for item in items
            if item.metadata["source_subscale"] == scale_id
        ]
        scales.append(
            Scale(
                id=scale_id,
                name=str(details["name"]),
                construct=str(details["construct"]),
                description=str(details["description"]),
                item_mappings=mappings,
                metadata={
                    "source_scale": details["source_scale"],
                    "source_table": details["source_table"],
                    "cronbach_alpha": details["cronbach_alpha"],
                    "variance_explained": details["variance_explained"],
                    "source_doi": PDMI_SOURCE_DOI,
                    "source_url": PDMI_SOURCE_URL,
                    "source_article": PDMI_CITATION,
                    "licence": PDMI_LICENCE,
                },
            )
        )
    return scales


_PDMI_ITEMS = _items()

PURCHASE_DECISION_MAKING_INVENTORY = Questionnaire(
    id="purchase_decision_making_inventory",
    shorthand="pdmi",
    name="Purchase Decision-Making Inventory",
    version="1.0",
    language="en",
    sections=[
        Section(
            id="pdmi_emotional_items",
            title="Scale of Emotional Decisions in Purchases",
            item_ids=[item.id for item in _PDMI_ITEMS[: len(_EMOTIONAL_ITEMS)]],
        ),
        Section(
            id="pdmi_reasoned_items",
            title="Scale of Reasoned Decisions in Purchases",
            item_ids=[item.id for item in _PDMI_ITEMS[len(_EMOTIONAL_ITEMS) :]],
        ),
    ],
    items=_PDMI_ITEMS,
    scales=_scales(_PDMI_ITEMS),
    metadata={
        "retain_history": True,
        "source_doi": PDMI_SOURCE_DOI,
        "source_url": PDMI_SOURCE_URL,
        "source_tables": ["Table 2", "Table 3"],
        "source_article": PDMI_CITATION,
        "source_sample": {
            "country": "Mexico",
            "total_n": 518,
            "efa_n": 300,
            "cfa_n": 218,
        },
        "source_response_format": "Five-point frequency scale from 1 = never to 5 = always.",
        "emotional_variance_explained": 48.75,
        "reasoned_variance_explained": 53.72,
        "emotional_cronbach_alpha": 0.915,
        "reasoned_cronbach_alpha": 0.919,
        "scoring_models_status": "deferred",
        "scoring_models_note": (
            "The questionnaire preserves source subscale mappings, but executable "
            "scoring models are deferred until scoring behavior is implemented."
        ),
    },
    reference=PDMI_CITATION,
    licence=PDMI_LICENCE,
)

PDMI = PURCHASE_DECISION_MAKING_INVENTORY
