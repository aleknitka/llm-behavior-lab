from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AllowedAnswer(BaseModel):
    answer_id: str
    label: str
    score: int | float | None = None


class QuestionnaireQuestion(BaseModel):
    question_id: str
    text: str
    allowed_answers: list[AllowedAnswer] = Field(min_length=1)
    trait: str | None = None
    reverse_scored: bool = False


class Questionnaire(BaseModel):
    questionnaire_id: str
    name: str
    questions: list[QuestionnaireQuestion] = Field(min_length=1)
    retain_history: bool = True
    citation: str | None = None
    source_url: str | None = None


class Persona(BaseModel):
    persona_id: str
    features: dict[str, str]


class ProviderCapabilities(BaseModel):
    supports_structured_outputs: bool = False
    supports_logprobs: bool = False


class ModelSettings(BaseModel):
    model: str
    provider_base_url: str
    temperature: float
    timeout_seconds: float
    seed: int | None = None
    capabilities: ProviderCapabilities = Field(default_factory=ProviderCapabilities)


class PersonaAssumption(BaseModel):
    persona_summary: str
    assumptions: list[str] = Field(default_factory=list)


class McqStructuredAnswer(BaseModel):
    selected_answer_id: str
    explanation: str | None = None


class LlmQuestionResult(BaseModel):
    selected_answer_id: str | None = None
    raw_response: str | None = None
    structured_response: dict[str, Any] | None = None
    logprobs: dict[str, Any] | None = None
    error: str | None = None


class JsonlQuestionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    persona: Persona
    questionnaire_id: str
    question_id: str
    question_text: str
    allowed_answers: list[AllowedAnswer]
    messages: list[dict[str, str]]
    model: str
    provider_base_url: str
    temperature: float
    selected_answer_id: str | None
    raw_response: str | None
    structured_response: dict[str, Any] | None
    logprobs: dict[str, Any] | None
    error: str | None
