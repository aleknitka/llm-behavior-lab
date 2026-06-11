import json
import re
from typing import Any, cast
from uuid import UUID

import pytest
from pydantic import ValidationError

from llm_behavior_lab.personas.dimensions import AffluenceLevel, EuropeanCountry
from llm_behavior_lab.personas.factory import (
    PersonaFactory,
    PersonaFactoryRequest,
    PersonaGenerationConfig,
    RequestedDemographicField,
)
from llm_behavior_lab.personas.value_specs import RandUniformRange
from llm_behavior_lab.storage import write_persona_batch_jsonl


def test_factory_creates_requested_number_of_typed_personas() -> None:
    batch = PersonaFactory().create_demographics_batch(
        PersonaFactoryRequest(
            count=3,
            requested_fields={
                RequestedDemographicField.AGE,
                RequestedDemographicField.GENDER,
                RequestedDemographicField.COUNTRY,
            },
            seed=7,
            experiment_id="demo-study-one",
        )
    )

    assert batch.experiment_id == "demo-study-one"
    assert len(batch) == 3
    assert len(batch.personas) == 3
    for persona in batch.personas:
        UUID(str(persona.subject_id))
        assert persona.features.age is not None
        assert persona.features.gender is not None
        assert isinstance(persona.features.country, EuropeanCountry)
        assert persona.features.region is None
        assert persona.features.education_level is None


def test_seeded_generation_is_deterministic() -> None:
    request = PersonaFactoryRequest(
        count=5,
        requested_fields={
            RequestedDemographicField.AGE,
            RequestedDemographicField.GENDER,
            RequestedDemographicField.COUNTRY,
            RequestedDemographicField.REGION,
            RequestedDemographicField.EMPLOYMENT_STATUS,
        },
        seed=22,
        experiment_id="seeded-study-one",
    )

    first = PersonaFactory().create_demographics_batch(request)
    second = PersonaFactory().create_demographics_batch(request)

    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_default_experiment_id_is_seeded_friendly_name() -> None:
    request = PersonaFactoryRequest(
        count=1,
        requested_fields={RequestedDemographicField.AGE},
        seed=31,
    )

    first = PersonaFactory().create_demographics_batch(request)
    second = PersonaFactory().create_demographics_batch(request)

    assert first.experiment_id == second.experiment_id
    assert re.fullmatch(r"[a-z]+-[a-z]+-[a-z]+", first.experiment_id)


def test_invalid_requested_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        PersonaFactoryRequest(count=1, requested_fields=cast(Any, {"unknown"}), seed=1)


def test_unsafe_experiment_id_is_rejected() -> None:
    with pytest.raises(ValidationError):
        PersonaFactoryRequest(
            count=1,
            requested_fields={RequestedDemographicField.AGE},
            experiment_id="../escape",
        )


def test_strict_realism_constraints_hold_for_generated_sample() -> None:
    batch = PersonaFactory().create_demographics_batch(
        PersonaFactoryRequest(
            count=200,
            requested_fields=set(RequestedDemographicField),
            seed=123,
            experiment_id="constraints-study-one",
        )
    )

    for persona in batch.personas:
        demographics = persona.features
        assert demographics.age is not None
        assert "age_group" not in type(demographics).model_fields
        assert demographics.employment_status is not None
        assert demographics.education_level is not None
        assert demographics.region is not None
        assert demographics.household_size is not None
        assert demographics.has_children is not None
        assert demographics.number_of_dependants is not None

        if demographics.age < 18:
            assert demographics.employment_status.value not in {
                "employed_full_time",
                "self_employed",
                "retired",
            }
            assert demographics.education_level.value in {"primary", "secondary", "other"}
            assert demographics.has_children is False

        if demographics.employment_status.value == "retired":
            assert demographics.age >= 55

        if demographics.education_level.value == "doctorate":
            assert demographics.age >= 25

        if demographics.education_level.value in {"bachelor", "master"}:
            assert demographics.age >= 18

        if demographics.has_children:
            assert demographics.age >= 18

        assert demographics.household_size >= 1 + demographics.number_of_dependants
        assert demographics.has_children is (demographics.number_of_dependants > 0)


def test_region_is_random_subdivision_from_selected_country() -> None:
    batch = PersonaFactory().create_demographics_batch(
        PersonaFactoryRequest(
            count=20,
            requested_fields={RequestedDemographicField.COUNTRY, RequestedDemographicField.REGION},
            seed=44,
            experiment_id="regions-study-one",
        )
    )

    for persona in batch.personas:
        assert persona.features.country is not None
        assert persona.features.region is not None
        assert persona.features.region


def test_weighted_config_restricts_country_and_affluence_choices() -> None:
    batch = PersonaFactory().create_demographics_batch(
        PersonaFactoryRequest(
            count=20,
            requested_fields={
                RequestedDemographicField.COUNTRY,
                RequestedDemographicField.AFFLUENCE_LEVEL,
            },
            seed=55,
            experiment_id="weighted-study-one",
            generation_config=PersonaGenerationConfig(
                field_probabilities={
                    RequestedDemographicField.COUNTRY: {
                        EuropeanCountry.POLAND.value: 0.75,
                        EuropeanCountry.GERMANY.value: 0.25,
                    },
                    RequestedDemographicField.AFFLUENCE_LEVEL: {AffluenceLevel.MIDDLE.value: 1.0},
                }
            ),
        )
    )

    assert {persona.features.country for persona in batch.personas} <= {
        EuropeanCountry.POLAND,
        EuropeanCountry.GERMANY,
    }
    assert {persona.features.affluence_level for persona in batch.personas} == {
        AffluenceLevel.MIDDLE
    }


def test_weighted_config_rejects_unknown_enum_value() -> None:
    with pytest.raises(ValidationError, match="country"):
        PersonaGenerationConfig(
            field_probabilities={
                RequestedDemographicField.COUNTRY: {
                    "not-a-country": 1.0,
                }
            }
        )


def test_fixed_field_values_materialize_unchanged() -> None:
    batch = PersonaFactory().create_demographics_batch(
        PersonaFactoryRequest(
            count=2,
            requested_fields={
                RequestedDemographicField.AGE,
                RequestedDemographicField.COUNTRY,
                RequestedDemographicField.HAS_CHILDREN,
            },
            experiment_id="fixed-values-one",
            generation_config=PersonaGenerationConfig(
                field_values={
                    RequestedDemographicField.AGE: 25,
                    RequestedDemographicField.COUNTRY: "GB",
                    RequestedDemographicField.HAS_CHILDREN: False,
                }
            ),
        )
    )

    assert [
        persona.features.model_dump(mode="json", exclude_none=True) for persona in batch.personas
    ] == [
        {"age": 25, "country": "GB", "has_children": False},
        {"age": 25, "country": "GB", "has_children": False},
    ]


def test_generated_field_values_materialize_as_deterministic_integers() -> None:
    request = PersonaFactoryRequest(
        count=5,
        requested_fields={RequestedDemographicField.AGE},
        seed=17,
        experiment_id="range-values-one",
        generation_config=PersonaGenerationConfig(
            field_values={
                RequestedDemographicField.AGE: RandUniformRange(20, 25),
            }
        ),
    )

    first = PersonaFactory().create_demographics_batch(request)
    second = PersonaFactory().create_demographics_batch(request)

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert all(
        isinstance(persona.features.age, int) and 20 <= persona.features.age <= 25
        for persona in first.personas
    )


def test_sampled_age_and_dependants_drive_dependent_generation() -> None:
    batch = PersonaFactory().create_demographics_batch(
        PersonaFactoryRequest(
            count=20,
            requested_fields={
                RequestedDemographicField.AGE,
                RequestedDemographicField.EDUCATION_LEVEL,
                RequestedDemographicField.EMPLOYMENT_STATUS,
                RequestedDemographicField.NUMBER_OF_DEPENDANTS,
                RequestedDemographicField.HAS_CHILDREN,
                RequestedDemographicField.HOUSEHOLD_SIZE,
            },
            seed=29,
            experiment_id="dependent-range-one",
            generation_config=PersonaGenerationConfig(
                field_values={
                    RequestedDemographicField.AGE: RandUniformRange(15, 15),
                    RequestedDemographicField.NUMBER_OF_DEPENDANTS: RandUniformRange(0, 0),
                }
            ),
        )
    )

    for persona in batch.personas:
        features = persona.features
        assert features.age == 15
        assert features.education_level in {"primary", "secondary", "other"}
        assert features.employment_status in {"student", "unemployed", "caregiver", "other"}
        assert features.number_of_dependants == 0
        assert features.has_children is False
        assert features.household_size is not None
        assert features.household_size >= 1


def test_field_value_json_parses_range_generator() -> None:
    config = PersonaGenerationConfig.model_validate(
        {
            "field_values": {
                "age": {
                    "type": "rand_uniform_range",
                    "left": 20,
                    "right": 25,
                }
            }
        }
    )

    assert config.field_values[RequestedDemographicField.AGE] == RandUniformRange(20, 25)
    assert config.model_dump(mode="json")["field_values"]["age"] == {
        "type": "rand_uniform_range",
        "left": 20,
        "right": 25,
    }


@pytest.mark.parametrize(
    ("request_updates", "message"),
    [
        ({"seed": None}, "seed"),
        ({"requested_fields": {"country"}}, "requested"),
        (
            {
                "generation_config": {
                    "field_probabilities": {"country": {"PL": 1.0}},
                }
            },
            "requested",
        ),
        (
            {
                "generation_config": {
                    "field_values": {
                        "age": {
                            "type": "rand_uniform_range",
                            "left": 20,
                            "right": 25,
                        }
                    },
                    "field_probabilities": {"age": {"20": 1.0}},
                }
            },
            "both",
        ),
        (
            {
                "generation_config": {
                    "field_values": {
                        "country": {"type": "rand_uniform_range", "left": 1, "right": 2}
                    }
                }
            },
            "country",
        ),
        (
            {
                "generation_config": {
                    "field_values": {
                        "age": {"type": "rand_uniform_range", "left": 121, "right": 122}
                    }
                }
            },
            "age",
        ),
    ],
)
def test_field_value_configuration_rejects_invalid_inputs(
    request_updates: dict[str, Any], message: str
) -> None:
    payload: dict[str, Any] = {
        "count": 1,
        "requested_fields": {"age"},
        "seed": 11,
        "experiment_id": "invalid-range-one",
        "generation_config": {
            "field_values": {"age": {"type": "rand_uniform_range", "left": 20, "right": 25}}
        },
    }
    payload.update(request_updates)

    with pytest.raises(ValidationError, match=message):
        PersonaFactoryRequest.model_validate(payload)


@pytest.mark.parametrize(
    "field_values",
    [
        {"age": 15, "number_of_dependants": 1},
        {"age": 15, "has_children": True},
        {"number_of_dependants": 2, "has_children": False},
        {"number_of_dependants": 2, "household_size": 2},
        {"age": 20, "education_level": "doctorate"},
        {"age": 20, "employment_status": "retired"},
    ],
)
def test_fixed_field_values_reject_impossible_demographic_combinations(
    field_values: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="incompatible"):
        PersonaGenerationConfig.model_validate({"field_values": field_values})


def test_age_group_is_not_a_supported_requested_field() -> None:
    assert "age_group" not in {field.value for field in RequestedDemographicField}


def test_write_persona_batch_jsonl_creates_experiment_dump(tmp_path) -> None:
    batch = PersonaFactory().create_demographics_batch(
        PersonaFactoryRequest(
            count=2,
            requested_fields={RequestedDemographicField.AGE, RequestedDemographicField.COUNTRY},
            seed=3,
            experiment_id="friendly-run-one",
        )
    )

    path = write_persona_batch_jsonl(tmp_path, batch)

    assert path == tmp_path / "experiments" / "friendly-run-one" / "personas.jsonl"
    dump = json.loads(path.read_text(encoding="utf-8"))
    assert dump["metadata"] == {
        "experiment_id": "friendly-run-one",
        "persona_count": 2,
        "requested_fields": ["age", "country"],
        "seed": 3,
    }
    assert len(dump["personas"]) == 2
    assert dump["personas"][0]["features"]["age"] is not None
    assert dump["personas"][0]["features"]["country"] in {
        country.value for country in EuropeanCountry
    }
