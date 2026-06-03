import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from llm_psych_scales.questionnaires.base.response_formats import ResponseFormat


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
    item_id: str
    role: str = "score_item"
    reverse_scored: bool = False
    weight: float = 1.0
    scoring_key: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Scale(BaseModel):
    id: str
    name: str
    construct: str
    description: str | None = None
    item_mappings: list[ItemMapping]


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
    scale_id: str
    transformation: Transformation
    output_min: float | None = None
    output_max: float | None = None
    interpretation_bands: list[InterpretationBand] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScoringModel(BaseModel):
    id: str
    name: str
    version: str
    scale_rules: list[ScaleScoringRule]


class Questionnaire(BaseModel):
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
            raise ValueError(
                "questionnaire shorthand must be 3-7 lowercase letters or digits"
            )
        return value
