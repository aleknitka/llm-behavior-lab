import pytest
from pydantic import ValidationError

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
from llm_behavior_lab.responses.base import (
    ChatMessage,
    ItemResponseRecord,
    LikertAnswerValue,
    ResponseStatus,
)
from llm_behavior_lab.scoring import ScaleScoreStatus, score_records


def _questionnaire() -> Questionnaire:
    response_format = LikertFormat(min_value=1, max_value=5)
    return Questionnaire(
        id="example",
        shorthand="exam",
        name="Example",
        version="1.0",
        sections=[Section(id="main", item_ids=["positive", "negative"])],
        items=[
            Item(
                id="positive",
                order=1,
                text="Positive",
                response_format=response_format,
            ),
            Item(
                id="negative",
                order=2,
                text="Negative",
                response_format=response_format,
            ),
        ],
        scales=[
            Scale(
                id="trait",
                name="Trait",
                construct="Trait",
                item_mappings=[
                    ItemMapping(item_id="positive", reverse_scored=True),
                    ItemMapping(item_id="negative"),
                ],
            )
        ],
        scoring_models=[
            ScoringModel(
                id="default",
                name="Default",
                version="1.0",
                provenance="project_defined",
                scale_rules=[
                    ScaleScoringRule(
                        scale_id="trait",
                        transformation=Transformation.MEAN,
                        output_min=1,
                        output_max=5,
                    )
                ],
            )
        ],
        reference="Reference",
        licence="Licence",
    )


def _record(item_id: str, value: int, status: ResponseStatus = ResponseStatus.COMPLETED):
    return ItemResponseRecord(
        subject_id="subject-1",
        session_id="session-1",
        run_id="run-1",
        questionnaire_id="example",
        questionnaire_version="1.0",
        item_id=item_id,
        item_order=1,
        item_text=item_id,
        response_format_type="likert",
        messages=[ChatMessage(role="user", content="Question")],
        answer=LikertAnswerValue(value=value),
        raw_response=str(value),
        status=status,
        metadata={"experiment_id": "pilot-study-one", "condition_id": "control"},
    )


def test_score_records_reverses_and_averages_item_values() -> None:
    questionnaire = _questionnaire()

    scores = score_records(
        questionnaire,
        questionnaire.scoring_models[0],
        [_record("positive", 1), _record("negative", 5)],
    )

    assert len(scores) == 1
    assert scores[0].status == ScaleScoreStatus.COMPLETED
    assert scores[0].score == 5
    assert [item.keyed_value for item in scores[0].items] == [5, 5]
    assert scores[0].metadata["condition_id"] == "control"


def test_score_records_marks_incomplete_scale_unscorable() -> None:
    questionnaire = _questionnaire()

    scores = score_records(
        questionnaire,
        questionnaire.scoring_models[0],
        [_record("positive", 1)],
    )

    assert scores[0].status == ScaleScoreStatus.UNSCORABLE
    assert scores[0].score is None
    assert scores[0].error == "missing or unscorable items: negative"


def test_questionnaire_rejects_scoring_rule_for_unknown_scale() -> None:
    questionnaire = _questionnaire()
    payload = questionnaire.model_dump()
    payload["scoring_models"][0]["scale_rules"][0]["scale_id"] = "missing"

    with pytest.raises(ValidationError, match="unknown scale"):
        Questionnaire.model_validate(payload)


def test_questionnaire_rejects_non_unit_mean_weight() -> None:
    questionnaire = _questionnaire()
    payload = questionnaire.model_dump()
    payload["scales"][0]["item_mappings"][0]["weight"] = 2

    with pytest.raises(ValidationError, match="unit item weights"):
        Questionnaire.model_validate(payload)
