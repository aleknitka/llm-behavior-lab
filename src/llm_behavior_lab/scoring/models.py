from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from llm_behavior_lab.questionnaires.base.scale import Transformation


class ScaleScoreStatus(StrEnum):
    COMPLETED = "completed"
    UNSCORABLE = "unscorable"


class ScoredItemContribution(BaseModel):
    """Auditable raw and keyed value used in one scale score."""
    item_id: str
    raw_value: float
    keyed_value: float
    weight: float = 1.0
    reverse_scored: bool = False


class ScaleScoreRecord(BaseModel):
    """Persisted subject-scale scoring result."""
    subject_id: str
    session_id: str
    run_id: str
    questionnaire_id: str
    questionnaire_version: str
    scoring_model_id: str
    scoring_model_version: str
    scale_id: str
    transformation: Transformation
    status: ScaleScoreStatus
    score: float | None = None
    interpretation: str | None = None
    items: list[ScoredItemContribution] = Field(default_factory=list)
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScaleReliabilityStatus(StrEnum):
    COMPLETED = "completed"
    NOT_COMPUTABLE = "not_computable"


class ScaleReliabilityRecord(BaseModel):
    """Persisted reliability result for a scale and optional condition."""
    scale_id: str
    scoring_model_id: str
    scoring_model_version: str
    status: ScaleReliabilityStatus
    subject_count: int = Field(ge=0)
    item_count: int = Field(ge=0)
    alpha: float | None = None
    confidence_interval: tuple[float, float] | None = None
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
