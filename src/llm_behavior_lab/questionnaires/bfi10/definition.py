from llm_behavior_lab.questionnaires.base.response_formats import LikertFormat
from llm_behavior_lab.questionnaires.base.scale import (
    Item,
    ItemMapping,
    Questionnaire,
    Scale,
    ScaleScoringRule,
    ScoringModel,
    Section,
    Transformation,
)

BFI_10_SOURCE_URL = (
    "https://www.gesis.org/fileadmin/admin/Dateikatalog/pdf/guidelines/"
    "SDMwiki/BFI-10/BFI-10_English_Items.pdf"
)
BFI_10_CITATION = (
    "Rammstedt, B. & John, O.P. (2007). Measuring personality in one minute or less: "
    "A 10-item short version of the Big Five Inventory in English and German. "
    "Journal of Research in Personality, 41, 203-212."
)
BFI_10_LICENCE = "See source instrument terms."

BFI_10_RESPONSE_FORMAT = LikertFormat(
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

BFI_10_ITEMS = [
    Item(
        id="bfi10_01_reserved",
        code="1",
        order=1,
        text="I see myself as someone who is reserved",
        response_format=BFI_10_RESPONSE_FORMAT,
        metadata={"trait": "extraversion", "reverse_scored": True},
    ),
    Item(
        id="bfi10_02_generally_trusting",
        code="2",
        order=2,
        text="I see myself as someone who is generally trusting",
        response_format=BFI_10_RESPONSE_FORMAT,
        metadata={"trait": "agreeableness", "reverse_scored": False},
    ),
    Item(
        id="bfi10_03_thorough_job",
        code="3",
        order=3,
        text="I see myself as someone who does a thorough job",
        response_format=BFI_10_RESPONSE_FORMAT,
        metadata={"trait": "conscientiousness", "reverse_scored": False},
    ),
    Item(
        id="bfi10_04_relaxed_handles_stress",
        code="4",
        order=4,
        text="I see myself as someone who is relaxed, handles stress well",
        response_format=BFI_10_RESPONSE_FORMAT,
        metadata={"trait": "emotional_stability", "reverse_scored": False},
    ),
    Item(
        id="bfi10_05_active_imagination",
        code="5",
        order=5,
        text="I see myself as someone who has an active imagination",
        response_format=BFI_10_RESPONSE_FORMAT,
        metadata={"trait": "openness", "reverse_scored": False},
    ),
    Item(
        id="bfi10_06_outgoing_sociable",
        code="6",
        order=6,
        text="I see myself as someone who is outgoing, sociable",
        response_format=BFI_10_RESPONSE_FORMAT,
        metadata={"trait": "extraversion", "reverse_scored": False},
    ),
    Item(
        id="bfi10_07_finds_fault",
        code="7",
        order=7,
        text="I see myself as someone who tends to find fault with others",
        response_format=BFI_10_RESPONSE_FORMAT,
        metadata={"trait": "agreeableness", "reverse_scored": True},
    ),
    Item(
        id="bfi10_08_lazy",
        code="8",
        order=8,
        text="I see myself as someone who tends to be lazy",
        response_format=BFI_10_RESPONSE_FORMAT,
        metadata={"trait": "conscientiousness", "reverse_scored": True},
    ),
    Item(
        id="bfi10_09_nervous_easily",
        code="9",
        order=9,
        text="I see myself as someone who gets nervous easily",
        response_format=BFI_10_RESPONSE_FORMAT,
        metadata={"trait": "emotional_stability", "reverse_scored": True},
    ),
    Item(
        id="bfi10_10_few_artistic_interests",
        code="10",
        order=10,
        text="I see myself as someone who has few artistic interests",
        response_format=BFI_10_RESPONSE_FORMAT,
        metadata={"trait": "openness", "reverse_scored": True},
    ),
]

_TRAIT_ITEMS = {
    "extraversion": [("bfi10_01_reserved", False), ("bfi10_06_outgoing_sociable", True)],
    "agreeableness": [("bfi10_02_generally_trusting", True), ("bfi10_07_finds_fault", False)],
    "conscientiousness": [("bfi10_03_thorough_job", True), ("bfi10_08_lazy", False)],
    "emotional_stability": [
        ("bfi10_04_relaxed_handles_stress", True),
        ("bfi10_09_nervous_easily", False),
    ],
    "openness": [
        ("bfi10_05_active_imagination", True),
        ("bfi10_10_few_artistic_interests", False),
    ],
}

BFI_10_SCALES = [
    Scale(
        id=trait,
        name=trait.replace("_", " ").title(),
        construct=trait.replace("_", " "),
        item_mappings=[
            ItemMapping(item_id=item_id, reverse_scored=reverse_scored)
            for item_id, reverse_scored in mappings
        ],
    )
    for trait, mappings in _TRAIT_ITEMS.items()
]

BFI_10_SCORING_MODEL = ScoringModel(
    id="default",
    name="BFI-10 construct-high mean scores",
    version="1.0",
    provenance="source_defined",
    scale_rules=[
        ScaleScoringRule(
            scale_id=scale.id,
            transformation=Transformation.MEAN,
            output_min=1,
            output_max=5,
        )
        for scale in BFI_10_SCALES
    ],
)

BFI_10 = Questionnaire(
    id="bfi_10",
    shorthand="bfi10",
    name="Big Five Inventory 10 Item Scale",
    version="1.0",
    language="en",
    sections=[
        Section(
            id="bfi10_items",
            title="BFI-10 items",
            item_ids=[item.id for item in BFI_10_ITEMS],
        )
    ],
    items=BFI_10_ITEMS,
    scales=BFI_10_SCALES,
    scoring_models=[BFI_10_SCORING_MODEL],
    metadata={
        "retain_history": True,
        "source_url": BFI_10_SOURCE_URL,
        "score_direction": "Higher scores indicate more of the named construct.",
    },
    reference=BFI_10_CITATION,
    licence=BFI_10_LICENCE,
)
