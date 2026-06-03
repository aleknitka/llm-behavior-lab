from importlib.resources import files

from jinja2 import Environment, StrictUndefined

from llm_psych_scales.models import Persona


def render_persona_intro(persona: Persona) -> str:
    template_path = files("llm_psych_scales.prompts").joinpath("persona_intro.j2")
    template_text = template_path.read_text(encoding="utf-8")
    environment = Environment(undefined=StrictUndefined, autoescape=False)  # nosec B701
    template = environment.from_string(template_text)
    return template.render(persona=persona)
