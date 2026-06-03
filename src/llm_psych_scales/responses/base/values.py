from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class AnswerValueType(StrEnum):
    LIKERT = "likert"
    NUMERIC = "numeric"
    SINGLE_CHOICE = "single_choice"
    MULTIPLE_CHOICE = "multiple_choice"
    TEXT = "text"


class LikertAnswerValue(BaseModel):
    type: Literal[AnswerValueType.LIKERT] = AnswerValueType.LIKERT
    value: int
    label: str | None = None


class NumericAnswerValue(BaseModel):
    type: Literal[AnswerValueType.NUMERIC] = AnswerValueType.NUMERIC
    value: float
    unit: str | None = None


class SingleChoiceAnswerValue(BaseModel):
    type: Literal[AnswerValueType.SINGLE_CHOICE] = AnswerValueType.SINGLE_CHOICE
    option_id: str
    label: str | None = None
    value: int | float | str | None = None


class MultipleChoiceAnswerValue(BaseModel):
    type: Literal[AnswerValueType.MULTIPLE_CHOICE] = AnswerValueType.MULTIPLE_CHOICE
    option_ids: list[str]
    labels: list[str] = Field(default_factory=list)
    values: list[int | float | str | None] = Field(default_factory=list)


class TextAnswerValue(BaseModel):
    type: Literal[AnswerValueType.TEXT] = AnswerValueType.TEXT
    text: str


AnswerValue = (
    LikertAnswerValue
    | NumericAnswerValue
    | SingleChoiceAnswerValue
    | MultipleChoiceAnswerValue
    | TextAnswerValue
)
