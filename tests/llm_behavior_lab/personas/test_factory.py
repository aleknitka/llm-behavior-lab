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
                    RequestedDemographicField.AFFLUENCE_LEVEL: {
                        AffluenceLevel.MIDDLE.value: 1.0
                    },
                }
            ),
        )
    )

    assert {
        persona.features.country for persona in batch.personas
    } <= {EuropeanCountry.POLAND, EuropeanCountry.GERMANY}
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
