from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from llm_psych_scales.responses.base.session import ResponseStatus
from llm_psych_scales.responses.base.values import AnswerValue


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    content: str


class ItemResponseRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject_id: str
    session_id: str
    run_id: str
    questionnaire_id: str
    questionnaire_version: str
    item_id: str
    item_order: int
    item_text: str
    response_format_type: str
    messages: list[ChatMessage]
    answer: Annotated[AnswerValue, Field(discriminator="type")] | None = None
    raw_response: str | None = None
    structured_response: dict[str, Any] | None = None
    logprobs: dict[str, Any] | None = None
    status: ResponseStatus
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
