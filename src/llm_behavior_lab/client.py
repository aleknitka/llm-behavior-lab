import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from email.utils import parsedate_to_datetime
from typing import Any, Protocol, cast

from loguru import logger
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    OpenAI,
)
from openai.types.chat import ChatCompletionMessageParam

from llm_behavior_lab.models import LlmQuestionResult, ModelSettings

Message = dict[str, str]
_RETRYABLE_STATUS_CODES = {408, 409, 429}


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
    logprobs = getattr(choice, "logprobs", None)
    return LlmQuestionResult(
        selected_answer_id=selected_answer_id,
        raw_response=content,
        logprobs=logprobs,
    )


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


def _is_retryable_error(exc: Exception) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return True
    return isinstance(exc, APIStatusError) and (
        exc.status_code in _RETRYABLE_STATUS_CODES or exc.status_code >= 500
    )


def _retry_after_seconds(exc: Exception) -> float | None:
    if not isinstance(exc, APIStatusError):
        return None
    value = exc.response.headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        return max(retry_at.timestamp() - time.time(), 0.0)


def _retry_delay(exc: Exception, settings: ModelSettings, attempt: int) -> float:
    retry_after = _retry_after_seconds(exc)
    if retry_after is not None:
        return min(retry_after, settings.max_backoff_seconds)
    exponential = settings.initial_backoff_seconds * (2 ** (attempt - 1))
    return min(exponential, settings.max_backoff_seconds)


class OpenAiChatClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        *,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client or OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
        )
        self._sleep = sleep

    def complete(
        self,
        messages: Sequence[Message],
        settings: ModelSettings,
        allowed_answer_ids: Sequence[str],
    ) -> LlmQuestionResult:
        openai_messages = _openai_messages(messages)
        logprobs = settings.capabilities.supports_logprobs or None
        for attempt in range(1, settings.max_attempts + 1):
            try:
                response = self._client.chat.completions.create(
                    model=settings.model,
                    messages=openai_messages,
                    temperature=settings.temperature,
                    timeout=settings.timeout_seconds,
                    seed=settings.seed,
                    logprobs=logprobs,
                )
                break
            except Exception as exc:
                if _is_retryable_error(exc):
                    if attempt >= settings.max_attempts:
                        raise
                    delay = _retry_delay(exc, settings, attempt)
                    logger.warning(
                        "Provider request failed; retrying attempt={attempt}/{max_attempts} "
                        "delay={delay}",
                        attempt=attempt + 1,
                        max_attempts=settings.max_attempts,
                        delay=delay,
                    )
                    self._sleep(delay)
                    continue
                if logprobs is not True or attempt >= settings.max_attempts:
                    raise
                logger.warning("Provider rejected logprobs; retrying without logprobs")
                logprobs = None
        else:
            raise RuntimeError("provider retry loop exhausted")
        result = _parse_choice(response.choices[0])
        if result.selected_answer_id is not None and not _is_selected_answer_allowed(
            result.selected_answer_id, allowed_answer_ids
        ):
            result.selected_answer_id = None
        return result


class AsyncOpenAiChatClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        *,
        client: Any | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._client = client or AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
        )
        self._sleep = sleep

    async def complete(
        self,
        messages: Sequence[Message],
        settings: ModelSettings,
        allowed_answer_ids: Sequence[str],
    ) -> LlmQuestionResult:
        openai_messages = _openai_messages(messages)
        logprobs = settings.capabilities.supports_logprobs or None
        for attempt in range(1, settings.max_attempts + 1):
            try:
                response = await self._client.chat.completions.create(
                    model=settings.model,
                    messages=openai_messages,
                    temperature=settings.temperature,
                    timeout=settings.timeout_seconds,
                    seed=settings.seed,
                    logprobs=logprobs,
                )
                break
            except Exception as exc:
                if _is_retryable_error(exc):
                    if attempt >= settings.max_attempts:
                        raise
                    delay = _retry_delay(exc, settings, attempt)
                    logger.warning(
                        "Provider request failed; retrying attempt={attempt}/{max_attempts} "
                        "delay={delay}",
                        attempt=attempt + 1,
                        max_attempts=settings.max_attempts,
                        delay=delay,
                    )
                    await self._sleep(delay)
                    continue
                if logprobs is not True or attempt >= settings.max_attempts:
                    raise
                logger.warning("Provider rejected logprobs; retrying without logprobs")
                logprobs = None
        else:
            raise RuntimeError("provider retry loop exhausted")
        result = _parse_choice(response.choices[0])
        if result.selected_answer_id is not None and not _is_selected_answer_allowed(
            result.selected_answer_id, allowed_answer_ids
        ):
            result.selected_answer_id = None
        return result
