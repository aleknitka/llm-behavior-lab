import json
from pathlib import Path

from llm_behavior_lab.personas.dimensions import (
    AffluenceLevel,
    EuropeanCountry,
    Urbanicity,
)
from llm_behavior_lab.protocols import (
    UnifiedExperimentProtocol,
    expand_protocol_personas,
)

EXAMPLE_ROOT = Path(__file__).parents[2] / "examples" / "ollama-bfi10-factorial"
PROTOCOL_PATH = EXAMPLE_ROOT / "protocol.json"
ROOT_README_PATH = EXAMPLE_ROOT.parents[1] / "README.md"
EXAMPLES_README_PATH = EXAMPLE_ROOT.parent / "README.md"


def test_ollama_bfi10_factorial_protocol_expands_as_documented() -> None:
    protocol = UnifiedExperimentProtocol.model_validate(
        json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    )
    factorial = protocol.personas.factorial
    assert factorial is not None

    expansion = expand_protocol_personas(factorial, protocol.experiment_id)

    assert protocol.experiment_id == "ollama-bfi-ten"
    assert protocol.personas.count == 1
    assert factorial.base_persona_count == 1
    assert factorial.iterations == 1
    assert len({assignment.condition_id for assignment in expansion.assignments}) == 12
    assert len(expansion.base_personas.personas) == 1
    assert len(expansion.personas.personas) == 12
    assert [step.id for step in protocol.steps] == ["personality"]

    features = [persona.features for persona in expansion.personas.personas]
    assert all(feature.country == EuropeanCountry.UNITED_KINGDOM for feature in features)
    assert {feature.age for feature in features} == {25, 65}
    assert {feature.affluence_level for feature in features} == {
        AffluenceLevel.LOW,
        AffluenceLevel.MIDDLE,
        AffluenceLevel.HIGH,
    }
    assert {feature.urbanicity for feature in features} == {
        Urbanicity.URBAN,
        Urbanicity.RURAL,
    }


def test_ollama_bfi10_factorial_uses_canonical_protocol_workflow() -> None:
    readme = (EXAMPLE_ROOT / "README.md").read_text(encoding="utf-8")
    notebook = (EXAMPLE_ROOT / "notebook.py").read_text(encoding="utf-8")

    for path in (
        "protocol.json",
        "metadata.json",
        "cohorts/cohort-{uuid}/",
        "personas.json",
        "protocol-assignments.json",
        "run.json",
        "steps/personality/",
        "responses/{subject_id}.jsonl",
    ):
        assert path in readme

    assert '"protocol-create"' in notebook
    assert '"--new-run"' in notebook
    assert '"--step-id"' in notebook
    assert '"personality"' in notebook
    assert '"scale-design"' not in notebook
    assert 'main(["personas"' not in notebook
    assert 'run_root / "steps" / "personality"' in notebook
    assert 'cohort_root / "personas.json"' in notebook
    assert 'cohort_root / "protocol-assignments.json"' in notebook
    assert 'run_root / "run.json"' in notebook
    assert 'results_root / "responses.csv"' in notebook


def test_ollama_bfi10_factorial_readmes_document_twelve_personas() -> None:
    root_readme = ROOT_README_PATH.read_text(encoding="utf-8")
    examples_readme = EXAMPLES_README_PATH.read_text(encoding="utf-8")
    example_readme = (EXAMPLE_ROOT / "README.md").read_text(encoding="utf-8")

    assert "12-person" in examples_readme
    assert "12 expanded personas" in example_readme
    assert "120 questionnaire calls" in example_readme
    assert "300-person" not in examples_readme
    assert "3,000" not in example_readme
    assert "examples/ollama-bfi10-factorial/notebook.py" in root_readme
