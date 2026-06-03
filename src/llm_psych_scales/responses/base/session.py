from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ResponseStatus(StrEnum):
    COMPLETED = "completed"
    INVALID = "invalid"
    FAILED = "failed"
    SKIPPED = "skipped"


class ProviderSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_base_url: str
    model: str
    temperature: float
    timeout_seconds: float
    supports_structured_outputs: bool = False
    supports_logprobs: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    session_id: str
    started_at: datetime
    completed_at: datetime | None = None
    status: ResponseStatus
    run_count: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    session_id: str
    run_id: str
    subject_ids: list[str]
    persona_count: int = Field(ge=1)
    questionnaire_id: str
    questionnaire_shorthand: str
    questionnaire_version: str
    model_slug: str
    provider: ProviderSnapshot
    started_at: datetime
    completed_at: datetime | None = None
    status: ResponseStatus
    error_count: int = Field(ge=0)
    item_count: int = Field(ge=0)
    output_paths: dict[str, str]
    metadata: dict[str, Any] = Field(default_factory=dict)
