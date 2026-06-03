from collections.abc import Sequence
from typing import Any, Protocol, cast

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletionMessageParam

from llm_psych_scales.models import LlmQuestionResult, ModelSettings

Message = dict[str, str]


class SyncLlmClient(Protocol):
    def complete(
        self,
        messages: Sequence[Message],
        settings: ModelSettings,
        allowed_answer_ids: Sequence[str],
    ) -> LlmQuestionResult: ...


class AsyncLlmClient(Protocol):
    async def complete(
        self,
        messages: Sequence[Message],
        settings: ModelSettings,
        allowed_answer_ids: Sequence[str],
    ) -> LlmQuestionResult: ...


def _parse_choice(choice: Any) -> LlmQuestionResult:
    message = choice.message
    content = message.content or ""
    selected_answer_id = content.strip()
    return LlmQuestionResult(selected_answer_id=selected_answer_id, raw_response=content)


def _openai_messages(messages: Sequence[Message]) -> list[ChatCompletionMessageParam]:
    return cast(list[ChatCompletionMessageParam], list(messages))


def _is_selected_answer_allowed(selected_answer_id: str, allowed_answer_ids: Sequence[str]) -> bool:
    if not allowed_answer_ids:
        return True

    selected_ids = [
        answer_id.strip() for answer_id in selected_answer_id.split(",") if answer_id.strip()
    ]
    if not selected_ids:
        return False

    allowed = set(allowed_answer_ids)
    return all(answer_id in allowed for answer_id in selected_ids)


class OpenAiChatClient:
    def __init__(self, api_key: str, base_url: str) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(
        self,
        messages: Sequence[Message],
        settings: ModelSettings,
        allowed_answer_ids: Sequence[str],
    ) -> LlmQuestionResult:
        response = self._client.chat.completions.create(
            model=settings.model,
            messages=_openai_messages(messages),
            temperature=settings.temperature,
            timeout=settings.timeout_seconds,
            logprobs=settings.capabilities.supports_logprobs or None,
        )
        result = _parse_choice(response.choices[0])
        if result.selected_answer_id is not None and not _is_selected_answer_allowed(
            result.selected_answer_id, allowed_answer_ids
        ):
            result.selected_answer_id = None
        return result


class AsyncOpenAiChatClient:
    def __init__(self, api_key: str, base_url: str) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def complete(
        self,
        messages: Sequence[Message],
        settings: ModelSettings,
        allowed_answer_ids: Sequence[str],
    ) -> LlmQuestionResult:
        response = await self._client.chat.completions.create(
            model=settings.model,
            messages=_openai_messages(messages),
            temperature=settings.temperature,
            timeout=settings.timeout_seconds,
            logprobs=settings.capabilities.supports_logprobs or None,
        )
        result = _parse_choice(response.choices[0])
        if result.selected_answer_id is not None and not _is_selected_answer_allowed(
            result.selected_answer_id, allowed_answer_ids
        ):
            result.selected_answer_id = None
        return result
