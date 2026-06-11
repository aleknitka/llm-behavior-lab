import pytest
from pydantic import ValidationError

from llm_behavior_lab.personas.factory import RequestedDemographicField
from llm_behavior_lab.personas.value_specs import RandUniformRange
from llm_behavior_lab.protocols import (
    ExperimentProtocol,
    ProtocolFactor,
    ProtocolFactorLevel,
    expand_protocol_personas,
)


def _protocol(**updates) -> ExperimentProtocol:
    payload = {
        "version": "1.0",
        "name": "gender-affluence-factorial",
        "base_persona_count": 2,
        "seed": 123,
        "iterations": 4,
        "requested_fields": [
            "age",
            "country",
            "gender",
            "affluence_level",
        ],
        "base_persona_config": {
            "field_probabilities": {
                "country": {"PL": 1.0},
            }
        },
        "factors": [
            {
                "name": "gender",
                "field": "gender",
                "levels": [
                    {"id": "female", "value": "female"},
                    {"id": "male", "value": "male"},
                ],
            },
            {
                "name": "affluence",
                "field": "affluence_level",
                "levels": [
                    {"id": "low", "value": "low"},
                    {"id": "middle", "value": "middle"},
                    {"id": "very_high", "value": "very_high"},
                ],
            },
        ],
    }
    payload.update(updates)
    return ExperimentProtocol.model_validate(payload)


def test_protocol_loads_two_by_three_factorial_design() -> None:
    protocol = _protocol()

    assert protocol.design == "paired_factorial"
    assert protocol.base_persona_count == 2
    assert protocol.iterations == 4
    assert [factor.name for factor in protocol.factors] == ["gender", "affluence"]


def test_protocol_rejects_invalid_enum_level_value() -> None:
    with pytest.raises(ValidationError, match="gender"):
        _protocol(
            factors=[
                {
                    "name": "gender",
                    "field": "gender",
                    "levels": [{"id": "bad", "value": "not-a-gender"}],
                }
            ]
        )


def test_protocol_rejects_duplicate_factor_fields() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        _protocol(
            factors=[
                {
                    "name": "gender-a",
                    "field": "gender",
                    "levels": [{"id": "female", "value": "female"}],
                },
                {
                    "name": "gender-b",
                    "field": "gender",
                    "levels": [{"id": "male", "value": "male"}],
                },
            ]
        )


def test_single_factor_protocol_is_valid_paired_comparison() -> None:
    protocol = _protocol(
        factors=[
            {
                "name": "gender",
                "field": "gender",
                "levels": [
                    {"id": "female", "value": "female"},
                    {"id": "male", "value": "male"},
                ],
            }
        ]
    )

    assert len(protocol.factors) == 1


def test_expand_protocol_creates_factorial_cells_and_iterations() -> None:
    expansion = expand_protocol_personas(_protocol(), experiment_id="proto-study-one")

    assert len(expansion.base_personas.personas) == 2
    assert len(expansion.personas.personas) == 2 * 2 * 3 * 4
    assert len(expansion.assignments) == len(expansion.personas.personas)
    assert {assignment.condition_id for assignment in expansion.assignments} == {
        "gender-female__affluence-low",
        "gender-female__affluence-middle",
        "gender-female__affluence-very_high",
        "gender-male__affluence-low",
        "gender-male__affluence-middle",
        "gender-male__affluence-very_high",
    }


def test_expand_protocol_keeps_base_features_constant_except_factors() -> None:
    expansion = expand_protocol_personas(_protocol(base_persona_count=1), "proto-study-one")
    grouped = {}
    for persona, assignment in zip(expansion.personas.personas, expansion.assignments, strict=True):
        key = (assignment.base_subject_id, assignment.condition_id)
        grouped.setdefault(key, []).append(persona.features.model_dump(mode="json"))

    for feature_sets in grouped.values():
        assert len(feature_sets) == 4
        assert all(features == feature_sets[0] for features in feature_sets)

    first_base_id = str(expansion.base_personas.personas[0].subject_id)
    for persona, assignment in zip(expansion.personas.personas, expansion.assignments, strict=True):
        base = expansion.base_personas.personas[0].features.model_dump(mode="json")
        features = persona.features.model_dump(mode="json")
        assert assignment.base_subject_id == first_base_id
        assert features["age"] == base["age"]
        assert features["country"] == base["country"]
        assert features["gender"] == assignment.factor_values["gender"]
        assert features["affluence_level"] == assignment.factor_values["affluence_level"]


def test_protocol_models_reject_duplicate_level_ids() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        ProtocolFactor(
            name="gender",
            field=RequestedDemographicField.GENDER,
            levels=[
                ProtocolFactorLevel(id="female", value="female"),
                ProtocolFactorLevel(id="female", value="male"),
            ],
        )


def test_protocol_range_level_round_trips_as_configuration() -> None:
    protocol = _protocol(
        factors=[
            {
                "name": "age_group",
                "field": "age",
                "levels": [
                    {
                        "id": "younger",
                        "value": {
                            "type": "rand_uniform_range",
                            "left": 20,
                            "right": 35,
                        },
                    }
                ],
            }
        ]
    )

    assert protocol.factors[0].levels[0].value == RandUniformRange(20, 35)
    assert protocol.model_dump(mode="json")["factors"][0]["levels"][0]["value"] == {
        "type": "rand_uniform_range",
        "left": 20,
        "right": 35,
    }


def test_protocol_range_levels_sample_per_iteration_deterministically() -> None:
    protocol = _protocol(
        base_persona_count=1,
        iterations=5,
        requested_fields=["age", "country"],
        factors=[
            {
                "name": "age_group",
                "field": "age",
                "levels": [
                    {
                        "id": "adult",
                        "value": {
                            "type": "rand_uniform_range",
                            "left": 20,
                            "right": 120,
                        },
                    }
                ],
            }
        ],
    )

    first = expand_protocol_personas(protocol, "proto-study-one")
    second = expand_protocol_personas(protocol, "proto-study-one")
    sampled_ages = [assignment.factor_values["age"] for assignment in first.assignments]

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert len(set(sampled_ages)) > 1
    assert all(isinstance(age, int) and 20 <= age <= 120 for age in sampled_ages)
    assert [persona.features.age for persona in first.personas.personas] == sampled_ages


def test_protocol_range_samples_are_independent_of_factor_order() -> None:
    age_factor = {
        "name": "age_group",
        "field": "age",
        "levels": [
            {
                "id": "adult",
                "value": {
                    "type": "rand_uniform_range",
                    "left": 20,
                    "right": 60,
                },
            }
        ],
    }
    gender_factor = {
        "name": "gender",
        "field": "gender",
        "levels": [
            {"id": "female", "value": "female"},
            {"id": "male", "value": "male"},
        ],
    }
    common = {
        "base_persona_count": 2,
        "iterations": 3,
        "requested_fields": ["age", "country", "gender"],
    }

    first = expand_protocol_personas(
        _protocol(**common, factors=[age_factor, gender_factor]), "proto-study-one"
    )
    reordered = expand_protocol_personas(
        _protocol(**common, factors=[gender_factor, age_factor]), "proto-study-one"
    )

    def samples_by_iteration(expansion) -> dict[tuple[str, int], int]:
        return {
            (assignment.base_subject_id, assignment.iteration_index): assignment.factor_values[
                "age"
            ]
            for assignment in expansion.assignments
        }

    assert samples_by_iteration(first) == samples_by_iteration(reordered)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {
                "seed": None,
                "factors": [
                    {
                        "name": "age_group",
                        "field": "age",
                        "levels": [
                            {
                                "id": "adult",
                                "value": {
                                    "type": "rand_uniform_range",
                                    "left": 20,
                                    "right": 60,
                                },
                            }
                        ],
                    }
                ],
            },
            "seed",
        ),
        (
            {
                "factors": [
                    {
                        "name": "country",
                        "field": "country",
                        "levels": [
                            {
                                "id": "bad",
                                "value": {
                                    "type": "rand_uniform_range",
                                    "left": 1,
                                    "right": 2,
                                },
                            }
                        ],
                    }
                ],
            },
            "country",
        ),
        (
            {
                "factors": [
                    {
                        "name": "age_group",
                        "field": "age",
                        "levels": [
                            {
                                "id": "bad",
                                "value": {
                                    "type": "rand_uniform_range",
                                    "left": 121,
                                    "right": 122,
                                },
                            }
                        ],
                    }
                ],
            },
            "age",
        ),
    ],
)
def test_protocol_rejects_invalid_range_configuration(
    updates: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        _protocol(**updates)
