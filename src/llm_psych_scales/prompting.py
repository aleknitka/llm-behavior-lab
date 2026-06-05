from importlib.resources import files

from jinja2 import Environment, StrictUndefined

from llm_psych_scales.models import Persona


def render_persona_intro(persona: Persona, context: str | None = None) -> str:
    """Render the system-style persona introduction prompt.

    Args:
        persona: Persona model containing the stable persona identifier and
            demographic feature map to include in the prompt.
        context: Optional supplemental text the model should read before
            answering questionnaire items, such as an experimental vignette,
            product description, or other paragraph-level stimulus. Blank or
            whitespace-only values are treated as missing context.

    Returns:
        Rendered prompt text suitable for the first message in a questionnaire
        session.

    Examples:
        >>> persona = Persona(persona_id="p1", features={"age": "35"})
        >>> prompt = render_persona_intro(persona, context="Read this vignette first.")
        >>> "Read this vignette first." in prompt
        True
    """
    template_path = files("llm_psych_scales.prompts").joinpath("persona_intro.j2")
    template_text = template_path.read_text(encoding="utf-8")
    environment = Environment(undefined=StrictUndefined, autoescape=False)  # nosec B701
    template = environment.from_string(template_text)
    rendered_context = context.strip() if context else None
    return template.render(persona=persona, context=rendered_context)
