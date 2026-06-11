import asyncio
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError

import llm_behavior_lab.client as client_module
from llm_behavior_lab.client import (
    AsyncOpenAiChatClient,
    OpenAiChatClient,
    _is_selected_answer_allowed,
)
from llm_behavior_lab.models import ModelSettings, ProviderCapabilities


def test_selected_answer_allows_exact_option_id() -> None:
    assert _is_selected_answer_allowed("2", ["1", "2", "3"])


def test_selected_answer_allows_empty_constraints_for_free_response() -> None:
    assert _is_selected_answer_allowed("open text", [])


def test_selected_answer_allows_comma_separated_multiple_choice_ids() -> None:
    assert _is_selected_answer_allowed("a, c", ["a", "b", "c"])


def test_selected_answer_rejects_unknown_option_id() -> None:
    assert not _is_selected_answer_allowed("z", ["a", "b", "c"])


class FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("logprobs") is True:
            raise RuntimeError("logprobs unsupported")
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="1"),
                    logprobs=None,
                )
            ]
        )


def test_openai_client_retries_without_logprobs_when_provider_rejects_them() -> None:
    completions = FakeCompletions()
    client = OpenAiChatClient(api_key="test", base_url="http://localhost:1234/v1")
    cast(Any, client)._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    settings = ModelSettings(
        model="local-model",
        provider_base_url="http://localhost:1234/v1",
        temperature=0.0,
        timeout_seconds=60.0,
        capabilities=ProviderCapabilities(supports_logprobs=True),
    )

    result = client.complete(
        messages=[{"role": "user", "content": "Question"}],
        settings=settings,
        allowed_answer_ids=["1", "2"],
    )

    assert result.selected_answer_id == "1"
    assert result.logprobs is None
    assert [call.get("logprobs") for call in completions.calls] == [True, None]


def test_openai_client_passes_seed_to_provider() -> None:
    completions = FakeCompletions()
    client = OpenAiChatClient(api_key="test", base_url="http://localhost:1234/v1")
    cast(Any, client)._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    settings = ModelSettings(
        model="local-model",
        provider_base_url="http://localhost:1234/v1",
        temperature=0.0,
        timeout_seconds=60.0,
        seed=456,
    )

    client.complete(
        messages=[{"role": "user", "content": "Question"}],
        settings=settings,
        allowed_answer_ids=["1", "2"],
    )

    assert completions.calls[0]["seed"] == 456


def test_openai_client_persists_logprobs_when_provider_returns_them() -> None:
    class ReturningCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="1"),
                        logprobs={"content": []},
                    )
                ]
            )

    client = OpenAiChatClient(api_key="test", base_url="http://localhost:1234/v1")
    cast(Any, client)._client = SimpleNamespace(
        chat=SimpleNamespace(completions=ReturningCompletions())
    )
    settings = ModelSettings(
        model="local-model",
        provider_base_url="http://localhost:1234/v1",
        temperature=0.0,
        timeout_seconds=60.0,
        capabilities=ProviderCapabilities(supports_logprobs=True),
    )

    result = client.complete(
        messages=[{"role": "user", "content": "Question"}],
        settings=settings,
        allowed_answer_ids=["1", "2"],
    )

    assert result.logprobs == {"content": []}


def _completion(answer: str = "1") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=answer),
                logprobs=None,
            )
        ]
    )


def _status_error(status_code: int, retry_after: str | None = None) -> APIStatusError:
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    response = httpx.Response(
        status_code,
        headers=headers,
        request=httpx.Request("POST", "http://localhost/v1/chat/completions"),
    )
    return APIStatusError("provider error", response=response, body=None)


class SequencedCompletions:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = outcomes
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class AsyncSequencedCompletions(SequencedCompletions):
    async def create(self, **kwargs):
        return super().create(**kwargs)


def _settings(**updates: object) -> ModelSettings:
    values: dict[str, object] = {
        "model": "local-model",
        "provider_base_url": "http://localhost:1234/v1",
        "temperature": 0.0,
        "timeout_seconds": 60.0,
        "max_attempts": 3,
        "initial_backoff_seconds": 0.5,
        "max_backoff_seconds": 2.0,
    }
    values.update(updates)
    return ModelSettings.model_validate(values)


@pytest.mark.parametrize("status_code", [408, 409, 429, 500, 503])
def test_sync_client_retries_transient_statuses(status_code: int) -> None:
    completions = SequencedCompletions([_status_error(status_code, retry_after="0"), _completion()])
    sleeps: list[float] = []
    client = OpenAiChatClient(
        api_key="test",
        base_url="http://localhost",
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        sleep=sleeps.append,
    )

    result = client.complete(
        [{"role": "user", "content": "Question"}],
        _settings(),
        ["1"],
    )

    assert result.selected_answer_id == "1"
    assert len(completions.calls) == 2
    assert sleeps == [0.0]


@pytest.mark.parametrize(
    "error",
    [
        APIConnectionError(request=httpx.Request("POST", "http://localhost/v1/chat/completions")),
        APITimeoutError(request=httpx.Request("POST", "http://localhost/v1/chat/completions")),
    ],
)
def test_sync_client_retries_transport_errors(error: Exception) -> None:
    completions = SequencedCompletions([error, _completion()])
    sleeps: list[float] = []
    client = OpenAiChatClient(
        api_key="test",
        base_url="http://localhost",
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        sleep=sleeps.append,
    )

    client.complete([{"role": "user", "content": "Question"}], _settings(), ["1"])

    assert len(completions.calls) == 2
    assert sleeps == [0.5]


def test_sync_client_keeps_logprobs_enabled_for_transient_retry() -> None:
    completions = SequencedCompletions([_status_error(503), _completion()])
    sleeps: list[float] = []
    client = OpenAiChatClient(
        api_key="test",
        base_url="http://localhost",
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        sleep=sleeps.append,
    )

    client.complete(
        [{"role": "user", "content": "Question"}],
        _settings(capabilities={"supports_logprobs": True}),
        ["1"],
    )

    assert [call["logprobs"] for call in completions.calls] == [True, True]
    assert sleeps == [0.5]


def test_sync_client_does_not_retry_other_4xx_errors() -> None:
    completions = SequencedCompletions([_status_error(400)])
    client = OpenAiChatClient(
        api_key="test",
        base_url="http://localhost",
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        sleep=lambda _: None,
    )

    with pytest.raises(APIStatusError):
        client.complete([{"role": "user", "content": "Question"}], _settings(), ["1"])

    assert len(completions.calls) == 1


def test_sync_client_does_not_treat_auth_error_as_logprobs_rejection() -> None:
    completions = SequencedCompletions([_status_error(401)])
    client = OpenAiChatClient(
        api_key="test",
        base_url="http://localhost",
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        sleep=lambda _: None,
    )

    with pytest.raises(APIStatusError):
        client.complete(
            [{"role": "user", "content": "Question"}],
            _settings(capabilities={"supports_logprobs": True}),
            ["1"],
        )

    assert len(completions.calls) == 1


def test_sync_client_does_not_treat_generic_bad_request_as_logprobs_rejection() -> None:
    completions = SequencedCompletions([_status_error(400)])
    client = OpenAiChatClient(
        api_key="test",
        base_url="http://localhost",
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        sleep=lambda _: None,
    )

    with pytest.raises(APIStatusError):
        client.complete(
            [{"role": "user", "content": "Question"}],
            _settings(capabilities={"supports_logprobs": True}),
            ["1"],
        )

    assert len(completions.calls) == 1


def test_sync_client_exhausts_max_attempts_with_capped_backoff() -> None:
    completions = SequencedCompletions([_status_error(503), _status_error(503), _status_error(503)])
    sleeps: list[float] = []
    client = OpenAiChatClient(
        api_key="test",
        base_url="http://localhost",
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        sleep=sleeps.append,
    )

    with pytest.raises(APIStatusError):
        client.complete(
            [{"role": "user", "content": "Question"}],
            _settings(initial_backoff_seconds=1.5, max_backoff_seconds=2),
            ["1"],
        )

    assert len(completions.calls) == 3
    assert sleeps == [1.5, 2.0]


def test_async_client_retries_transient_statuses() -> None:
    completions = AsyncSequencedCompletions([_status_error(429), _completion()])
    sleeps: list[float] = []

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    client = AsyncOpenAiChatClient(
        api_key="test",
        base_url="http://localhost",
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        sleep=sleep,
    )

    result = asyncio.run(
        client.complete(
            [{"role": "user", "content": "Question"}],
            _settings(),
            ["1"],
        )
    )

    assert result.selected_answer_id == "1"
    assert len(completions.calls) == 2
    assert sleeps == [0.5]


def test_sdk_clients_disable_builtin_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    sync_kwargs: dict[str, object] = {}
    async_kwargs: dict[str, object] = {}

    def fake_sync(**kwargs):
        sync_kwargs.update(kwargs)
        return SimpleNamespace()

    def fake_async(**kwargs):
        async_kwargs.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(client_module, "OpenAI", fake_sync)
    monkeypatch.setattr(client_module, "AsyncOpenAI", fake_async)

    OpenAiChatClient(api_key="test", base_url="http://localhost")
    AsyncOpenAiChatClient(api_key="test", base_url="http://localhost")

    assert sync_kwargs["max_retries"] == 0
    assert async_kwargs["max_retries"] == 0
