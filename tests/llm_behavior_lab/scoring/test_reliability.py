import pytest

from llm_behavior_lab.scoring import (
    ScaleReliabilityStatus,
    calculate_reliability,
)


def test_calculate_reliability_returns_alpha_and_confidence_interval() -> None:
    reliability = calculate_reliability(
        scale_id="trait",
        scoring_model_id="default",
        scoring_model_version="1.0",
        item_values={
            "subject-1": {"item-1": 1.0, "item-2": 2.0, "item-3": 1.0},
            "subject-2": {"item-1": 2.0, "item-2": 3.0, "item-3": 2.0},
            "subject-3": {"item-1": 3.0, "item-2": 4.0, "item-3": 4.0},
            "subject-4": {"item-1": 4.0, "item-2": 5.0, "item-3": 5.0},
        },
    )

    assert reliability.status == ScaleReliabilityStatus.COMPLETED
    assert reliability.alpha == pytest.approx(0.9827586)
    assert reliability.confidence_interval is not None
    assert reliability.subject_count == 4
    assert reliability.item_count == 3


def test_calculate_reliability_requires_two_items() -> None:
    reliability = calculate_reliability(
        scale_id="trait",
        scoring_model_id="default",
        scoring_model_version="1.0",
        item_values={"subject-1": {"item-1": 1.0}, "subject-2": {"item-1": 2.0}},
    )

    assert reliability.status == ScaleReliabilityStatus.NOT_COMPUTABLE
    assert reliability.reason == "at least two items are required"
