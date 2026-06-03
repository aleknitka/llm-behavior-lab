from types import SimpleNamespace
from typing import Any, cast

from llm_psych_scales.client import OpenAiChatClient, _is_selected_answer_allowed
from llm_psych_scales.models import ModelSettings, ProviderCapabilities


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
    cast(Any, client)._client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
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

    assert result.selected_answer_id == "1"
    assert result.logprobs is None
    assert [call.get("logprobs") for call in completions.calls] == [True, None]


def test_openai_client_passes_seed_to_provider() -> None:
    completions = FakeCompletions()
    client = OpenAiChatClient(api_key="test", base_url="http://localhost:1234/v1")
    cast(Any, client)._client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
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
