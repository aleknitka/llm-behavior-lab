from importlib.resources import files

from jinja2 import Environment, StrictUndefined

from llm_behavior_lab.preference_tests.models import PairwisePreferenceExperiment, PairwiseTrial


def render_pairwise_preference_prompt(
    experiment: PairwisePreferenceExperiment,
    trial: PairwiseTrial,
    displayed_stimulus_ids: tuple[str, str],
) -> str:
    first = experiment.stimulus_by_id(displayed_stimulus_ids[0])
    second = experiment.stimulus_by_id(displayed_stimulus_ids[1])
    template_path = files("llm_behavior_lab.prompts").joinpath("pairwise_preference.j2")
    template_text = template_path.read_text(encoding="utf-8")
    environment = Environment(undefined=StrictUndefined, autoescape=False)  # nosec B701
    template = environment.from_string(template_text)
    return template.render(
        instruction=experiment.instruction,
        first_text=first.text,
        second_text=second.text,
    )
