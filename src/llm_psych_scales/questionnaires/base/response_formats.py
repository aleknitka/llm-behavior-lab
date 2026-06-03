from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

__all__ = ["ResponseFormat"]


class ResponseType(StrEnum):
    LIKERT = "likert"
    NUMERIC = "numeric"
    SINGLE_CHOICE = "single_choice"
    MULTIPLE_CHOICE = "multiple_choice"
    TEXT = "text"


class Option(BaseModel):
    id: str
    label: str
    value: int | float | str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LikertFormat(BaseModel):
    type: Literal[ResponseType.LIKERT] = ResponseType.LIKERT
    min_value: int
    max_value: int
    labels: dict[int, str] = Field(default_factory=dict)


class NumericFormat(BaseModel):
    type: Literal[ResponseType.NUMERIC] = ResponseType.NUMERIC
    min_value: float | None = None
    max_value: float | None = None
    unit: str | None = None


class SingleChoiceFormat(BaseModel):
    type: Literal[ResponseType.SINGLE_CHOICE] = ResponseType.SINGLE_CHOICE
    options: list[Option]


class MultipleChoiceFormat(BaseModel):
    type: Literal[ResponseType.MULTIPLE_CHOICE] = ResponseType.MULTIPLE_CHOICE
    options: list[Option]
    min_selected: int | None = None
    max_selected: int | None = None


class TextFormat(BaseModel):
    type: Literal[ResponseType.TEXT] = ResponseType.TEXT
    max_length: int | None = None


ResponseFormat = (
    LikertFormat | NumericFormat | SingleChoiceFormat | MultipleChoiceFormat | TextFormat
)
