from collections import defaultdict
from collections.abc import Sequence

from llm_behavior_lab.questionnaires.base.response_formats import (
    LikertFormat,
    NumericFormat,
    SingleChoiceFormat,
)
from llm_behavior_lab.questionnaires.base.scale import (
    Item,
    ItemMapping,
    Questionnaire,
    Scale,
    ScaleScoringRule,
    ScoringModel,
    Transformation,
)
from llm_behavior_lab.responses.base import (
    ItemResponseRecord,
    LikertAnswerValue,
    NumericAnswerValue,
    ResponseStatus,
    SingleChoiceAnswerValue,
)
from llm_behavior_lab.scoring.models import (
    ScaleScoreRecord,
    ScaleScoreStatus,
    ScoredItemContribution,
)


def score_records(
    questionnaire: Questionnaire,
    scoring_model: ScoringModel,
    records: Sequence[ItemResponseRecord],
) -> list[ScaleScoreRecord]:
    """Score validated item responses without modifying source records."""
    by_subject: dict[str, list[ItemResponseRecord]] = defaultdict(list)
    for record in records:
        if record.questionnaire_id != questionnaire.id:
            raise ValueError("response questionnaire_id does not match questionnaire")
        if record.questionnaire_version != questionnaire.version:
            raise ValueError("response questionnaire_version does not match questionnaire")
        by_subject[record.subject_id].append(record)

    scales = {scale.id: scale for scale in questionnaire.scales}
    rules = {rule.scale_id: rule for rule in scoring_model.scale_rules}
    items = {item.id: item for item in questionnaire.items}
    output: list[ScaleScoreRecord] = []
    for subject_id, subject_records in sorted(by_subject.items()):
        responses = {record.item_id: record for record in subject_records}
        for scale_id, rule in rules.items():
            output.append(
                _score_scale(
                    questionnaire,
                    scoring_model,
                    scales[scale_id],
                    rule,
                    items,
                    responses,
                    subject_id,
                    subject_records[0],
                )
            )
    return output


def _score_scale(
    questionnaire: Questionnaire,
    scoring_model: ScoringModel,
    scale: Scale,
    rule: ScaleScoringRule,
    items: dict[str, Item],
    responses: dict[str, ItemResponseRecord],
    subject_id: str,
    exemplar: ItemResponseRecord,
) -> ScaleScoreRecord:
    contributions: list[ScoredItemContribution] = []
    missing: list[str] = []
    for mapping in scale.item_mappings:
        record = responses.get(mapping.item_id)
        if record is None or record.status != ResponseStatus.COMPLETED or record.answer is None:
            missing.append(mapping.item_id)
            continue
        try:
            raw_value = _scalar_value(items[mapping.item_id], mapping, record)
            keyed_value = _reverse_value(items[mapping.item_id], raw_value, mapping)
        except ValueError:
            missing.append(mapping.item_id)
            continue
        contributions.append(
            ScoredItemContribution(
                item_id=mapping.item_id,
                raw_value=raw_value,
                keyed_value=keyed_value,
                weight=mapping.weight,
                reverse_scored=mapping.reverse_scored,
            )
        )

    if missing:
        return ScaleScoreRecord(
            subject_id=subject_id,
            session_id=exemplar.session_id,
            run_id=exemplar.run_id,
            questionnaire_id=questionnaire.id,
            questionnaire_version=questionnaire.version,
            scoring_model_id=scoring_model.id,
            scoring_model_version=scoring_model.version,
            scale_id=scale.id,
            transformation=rule.transformation,
            status=ScaleScoreStatus.UNSCORABLE,
            items=contributions,
            error=f"missing or unscorable items: {', '.join(missing)}",
            metadata=dict(exemplar.metadata),
        )

    score = _aggregate(contributions, rule.transformation)
    interpretation = next(
        (
            band.label
            for band in rule.interpretation_bands
            if (band.min_value is None or score >= band.min_value)
            and (band.max_value is None or score <= band.max_value)
        ),
        None,
    )
    return ScaleScoreRecord(
        subject_id=subject_id,
        session_id=exemplar.session_id,
        run_id=exemplar.run_id,
        questionnaire_id=questionnaire.id,
        questionnaire_version=questionnaire.version,
        scoring_model_id=scoring_model.id,
        scoring_model_version=scoring_model.version,
        scale_id=scale.id,
        transformation=rule.transformation,
        status=ScaleScoreStatus.COMPLETED,
        score=score,
        interpretation=interpretation,
        items=contributions,
        metadata=dict(exemplar.metadata),
    )


def _scalar_value(item: Item, mapping: ItemMapping, record: ItemResponseRecord) -> float:
    answer = record.answer
    if isinstance(answer, LikertAnswerValue | NumericAnswerValue):
        return float(answer.value)
    if isinstance(answer, SingleChoiceAnswerValue):
        if mapping.scoring_key is None or answer.option_id not in mapping.scoring_key:
            raise ValueError("single-choice answer requires a scoring key")
        value = mapping.scoring_key[answer.option_id]
        if not isinstance(value, int | float):
            raise ValueError("single-choice score must be numeric")
        return float(value)
    raise ValueError("answer type is not scalar")


def _reverse_value(item: Item, value: float, mapping: ItemMapping) -> float:
    if not mapping.reverse_scored:
        return value
    response_format = item.response_format
    if isinstance(response_format, LikertFormat):
        return float(response_format.min_value + response_format.max_value) - value
    if isinstance(response_format, NumericFormat):
        if response_format.min_value is None or response_format.max_value is None:
            raise ValueError("reverse-scored numeric item requires bounds")
        return response_format.min_value + response_format.max_value - value
    if isinstance(response_format, SingleChoiceFormat):
        values = list(mapping.scoring_key.values()) if mapping.scoring_key else []
        if not values or not all(isinstance(candidate, int | float) for candidate in values):
            raise ValueError("reverse-scored choice item requires numeric scoring key")
        return float(min(values) + max(values)) - value
    raise ValueError("response format cannot be reverse scored")


def _aggregate(
    contributions: Sequence[ScoredItemContribution],
    transformation: Transformation,
) -> float:
    if transformation == Transformation.SUM:
        return sum(item.keyed_value for item in contributions)
    if transformation == Transformation.MEAN:
        return sum(item.keyed_value for item in contributions) / len(contributions)
    weighted_total = sum(item.keyed_value * item.weight for item in contributions)
    return weighted_total / sum(item.weight for item in contributions)
