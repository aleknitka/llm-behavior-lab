import json
from pathlib import Path

from llm_behavior_lab.personas.dimensions import (
    AffluenceLevel,
    EuropeanCountry,
    Urbanicity,
)
from llm_behavior_lab.protocols import ExperimentProtocol, expand_protocol_personas

EXAMPLE_ROOT = Path(__file__).parents[2] / "examples" / "ollama-bfi10-factorial"
PROTOCOL_PATH = EXAMPLE_ROOT / "protocol.json"


def test_ollama_bfi10_factorial_protocol_expands_as_documented() -> None:
    protocol = ExperimentProtocol.model_validate(
        json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    )

    expansion = expand_protocol_personas(protocol, "ollama-bfi-ten")

    assert protocol.base_persona_count == 5
    assert protocol.iterations == 5
    assert len({assignment.condition_id for assignment in expansion.assignments}) == 12
    assert len(expansion.base_personas.personas) == 5
    assert len(expansion.personas.personas) == 300

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
