from llm_psych_scales.models import Persona
from llm_psych_scales.prompting import render_persona_intro


def test_render_persona_intro_includes_features() -> None:
    prompt = render_persona_intro(
        Persona(persona_id="p1", features={"age": "35", "country": "Poland"})
    )

    assert "Persona ID: p1" in prompt
    assert "age: 35" in prompt
    assert "country: Poland" in prompt
    assert "You are taking part in a psychological questionnaire session." in prompt
    assert "Choose the answer that best fits this persona" in prompt
    assert "Do not mention that you are an AI model" in prompt


def test_render_persona_intro_handles_missing_features() -> None:
    prompt = render_persona_intro(Persona(persona_id="p1", features={}))

    assert "No demographic features were provided." in prompt


def test_render_persona_intro_includes_optional_context() -> None:
    prompt = render_persona_intro(
        Persona(persona_id="p1", features={"age": "35"}),
        context="Read this product description before answering the questionnaire.",
    )

    assert "Additional context:" in prompt
    assert "Read this product description before answering the questionnaire." in prompt


def test_render_persona_intro_omits_context_section_when_context_is_missing() -> None:
    prompt = render_persona_intro(Persona(persona_id="p1", features={"age": "35"}))

    assert "Additional context:" not in prompt
