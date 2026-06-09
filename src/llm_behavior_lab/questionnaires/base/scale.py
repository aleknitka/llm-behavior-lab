import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from llm_behavior_lab.questionnaires.base.response_formats import ResponseFormat


class Item(BaseModel):
    id: str
    code: str | None = None
    order: int
    instructions: str | None = None
    help_text: str | None = None
    text: str
    response_format: ResponseFormat
    required: bool = True
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Section(BaseModel):
    id: str
    title: str | None = None
    description: str | None = None
    item_ids: list[str]


class ItemMapping(BaseModel):
    """Map one questionnaire item into a scale score."""
    item_id: str
    role: str = "score_item"
    reverse_scored: bool = False
    weight: float = Field(default=1.0, gt=0)
    scoring_key: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Scale(BaseModel):
    """Named construct measured by an ordered set of item mappings."""
    id: str
    name: str
    construct: str
    description: str | None = None
    item_mappings: list[ItemMapping]
    metadata: dict[str, Any] = Field(default_factory=dict)


class Transformation(StrEnum):
    SUM = "sum"
    MEAN = "mean"
    WEIGHTED_MEAN = "weighted_mean"


class InterpretationBand(BaseModel):
    label: str
    min_value: float | None = None
    max_value: float | None = None
    description: str | None = None


class ScaleScoringRule(BaseModel):
    """Versioned aggregation rule for one scale."""
    scale_id: str
    transformation: Transformation
    output_min: float | None = None
    output_max: float | None = None
    interpretation_bands: list[InterpretationBand] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScoringModel(BaseModel):
    """Collection of executable scale rules with explicit provenance."""
    id: str
    name: str
    version: str
    provenance: str = "source_defined"
    scale_rules: list[ScaleScoringRule]


class Questionnaire(BaseModel):
    """Validated questionnaire definition and optional executable scoring models."""
    id: str
    shorthand: str
    name: str
    version: str
    language: str | None = None
    sections: list[Section]
    items: list[Item]
    scales: list[Scale] = Field(default_factory=list)
    scoring_models: list[ScoringModel] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    reference: str
    licence: str

    @field_validator("shorthand")
    @classmethod
    def validate_shorthand(cls, value: str) -> str:
        if not re.fullmatch(r"[a-z0-9]{3,7}", value):
            raise ValueError("questionnaire shorthand must be 3-7 lowercase letters or digits")
        return value

    @model_validator(mode="after")
    def validate_references_and_scoring(self) -> "Questionnaire":
        item_ids = [item.id for item in self.items]
        if len(set(item_ids)) != len(item_ids):
            raise ValueError("questionnaire contains duplicate item ids")
        scale_ids = [scale.id for scale in self.scales]
        if len(set(scale_ids)) != len(scale_ids):
            raise ValueError("questionnaire contains duplicate scale ids")
        known_items = set(item_ids)
        for section in self.sections:
            unknown = set(section.item_ids) - known_items
            if unknown:
                raise ValueError(f"section {section.id!r} references unknown items: {unknown}")
        scales_by_id = {scale.id: scale for scale in self.scales}
        for scale in self.scales:
            unknown = {mapping.item_id for mapping in scale.item_mappings} - known_items
            if unknown:
                raise ValueError(f"scale {scale.id!r} references unknown items: {unknown}")
        model_ids = [model.id for model in self.scoring_models]
        if len(set(model_ids)) != len(model_ids):
            raise ValueError("questionnaire contains duplicate scoring model ids")
        for scoring_model in self.scoring_models:
            rule_ids = [rule.scale_id for rule in scoring_model.scale_rules]
            if len(set(rule_ids)) != len(rule_ids):
                raise ValueError(f"scoring model {scoring_model.id!r} has duplicate scale rules")
            for rule in scoring_model.scale_rules:
                scale = scales_by_id.get(rule.scale_id)
                if scale is None:
                    raise ValueError(
                        f"scoring model {scoring_model.id!r} references unknown scale "
                        f"{rule.scale_id!r}"
                    )
                if rule.transformation != Transformation.WEIGHTED_MEAN and any(
                    mapping.weight != 1 for mapping in scale.item_mappings
                ):
                    raise ValueError(
                        f"{rule.transformation} requires unit item weights for scale "
                        f"{scale.id!r}"
                    )
                _validate_interpretation_bands(rule)
        return self


def _validate_interpretation_bands(rule: ScaleScoringRule) -> None:
    ordered = sorted(
        rule.interpretation_bands,
        key=lambda band: float("-inf") if band.min_value is None else band.min_value,
    )
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if previous.max_value is None or current.min_value is None:
            raise ValueError(f"scale {rule.scale_id!r} has overlapping interpretation bands")
        if current.min_value <= previous.max_value:
            raise ValueError(f"scale {rule.scale_id!r} has overlapping interpretation bands")
