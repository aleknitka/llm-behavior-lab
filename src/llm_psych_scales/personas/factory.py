from __future__ import annotations

import random
from collections.abc import Sequence
from enum import StrEnum
from typing import Any
from uuid import UUID

import pycountry
from pydantic import BaseModel, Field, field_validator, model_validator

from llm_psych_scales.personas.dimensions import (
    AffluenceLevel,
    Demographics,
    EducationLevel,
    EmploymentStatus,
    EuropeanCountry,
    FamilyStatus,
    Gender,
    Urbanicity,
)
from llm_psych_scales.storage import generate_experiment_id, validate_experiment_id


class RequestedDemographicField(StrEnum):
    AGE = "age"
    GENDER = "gender"
    EDUCATION_LEVEL = "education_level"
    EMPLOYMENT_STATUS = "employment_status"
    AFFLUENCE_LEVEL = "affluence_level"
    COUNTRY = "country"
    REGION = "region"
    URBANICITY = "urbanicity"
    FAMILY_STATUS = "family_status"
    HOUSEHOLD_SIZE = "household_size"
    HAS_CHILDREN = "has_children"
    NUMBER_OF_DEPENDANTS = "number_of_dependants"


class GeneratedPersona(BaseModel):
    subject_id: UUID
    features: Demographics


class PersonaBatchMetadata(BaseModel):
    experiment_id: str
    persona_count: int = Field(ge=1)
    requested_fields: list[RequestedDemographicField] = Field(min_length=1)
    seed: int | None = None


class PersonaBatch(BaseModel):
    experiment_id: str
    metadata: PersonaBatchMetadata
    personas: list[GeneratedPersona] = Field(min_length=1)

    def __len__(self) -> int:
        return len(self.personas)


_WEIGHTED_FIELD_ENUMS: dict[RequestedDemographicField, type[StrEnum]] = {
    RequestedDemographicField.GENDER: Gender,
    RequestedDemographicField.EDUCATION_LEVEL: EducationLevel,
    RequestedDemographicField.EMPLOYMENT_STATUS: EmploymentStatus,
    RequestedDemographicField.AFFLUENCE_LEVEL: AffluenceLevel,
    RequestedDemographicField.COUNTRY: EuropeanCountry,
    RequestedDemographicField.URBANICITY: Urbanicity,
    RequestedDemographicField.FAMILY_STATUS: FamilyStatus,
}


class PersonaGenerationConfig(BaseModel):
    field_probabilities: dict[RequestedDemographicField, dict[str, float]] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def validate_field_probabilities(self) -> PersonaGenerationConfig:
        for field, probabilities in self.field_probabilities.items():
            enum_type = _WEIGHTED_FIELD_ENUMS.get(field)
            if enum_type is None:
                msg = f"{field.value} does not support weighted enum probabilities"
                raise ValueError(msg)
            if not probabilities:
                msg = f"{field.value} probabilities cannot be empty"
                raise ValueError(msg)
            for value, probability in probabilities.items():
                if probability <= 0:
                    msg = f"{field.value} probability for {value!r} must be positive"
                    raise ValueError(msg)
                try:
                    enum_type(value)
                except ValueError as exc:
                    msg = f"{field.value} has unsupported value {value!r}"
                    raise ValueError(msg) from exc
        return self


class PersonaFactoryRequest(BaseModel):
    count: int = Field(ge=1)
    requested_fields: set[RequestedDemographicField] = Field(min_length=1)
    seed: int | None = None
    experiment_id: str | None = None
    generation_config: PersonaGenerationConfig = Field(default_factory=PersonaGenerationConfig)

    @field_validator("experiment_id")
    @classmethod
    def validate_experiment_id(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return validate_experiment_id(value)


class PersonaFactory:
    def create_demographics_batch(self, request: PersonaFactoryRequest) -> PersonaBatch:
        rng = random.Random(request.seed)
        experiment_id = request.experiment_id or generate_experiment_id(request.seed)
        personas = [
            GeneratedPersona(
                subject_id=UUID(int=rng.getrandbits(128), version=4),
                features=self._create_demographics(
                    request.requested_fields, rng, request.generation_config
                ),
            )
            for _ in range(request.count)
        ]
        metadata = PersonaBatchMetadata(
            experiment_id=experiment_id,
            persona_count=request.count,
            requested_fields=sorted(request.requested_fields, key=lambda field: field.value),
            seed=request.seed,
        )
        return PersonaBatch(experiment_id=experiment_id, metadata=metadata, personas=personas)

    def _create_demographics(
        self,
        requested_fields: set[RequestedDemographicField],
        rng: random.Random,
        config: PersonaGenerationConfig,
    ) -> Demographics:
        age = self._choose_age(rng)
        number_of_dependants = self._choose_number_of_dependants(age, rng)
        country = self._choose_country(requested_fields, rng, config)
        values: dict[str, Any] = {}

        if RequestedDemographicField.AGE in requested_fields:
            values["age"] = age
        if RequestedDemographicField.GENDER in requested_fields:
            values["gender"] = self._choose_enum(
                RequestedDemographicField.GENDER, list(Gender), rng, config
            )
        if RequestedDemographicField.EDUCATION_LEVEL in requested_fields:
            values["education_level"] = self._choose_education_level(age, rng, config)
        if RequestedDemographicField.EMPLOYMENT_STATUS in requested_fields:
            values["employment_status"] = self._choose_employment_status(age, rng, config)
        if RequestedDemographicField.AFFLUENCE_LEVEL in requested_fields:
            values["affluence_level"] = self._choose_enum(
                RequestedDemographicField.AFFLUENCE_LEVEL,
                list(AffluenceLevel),
                rng,
                config,
            )
        if RequestedDemographicField.COUNTRY in requested_fields:
            values["country"] = country
        if RequestedDemographicField.REGION in requested_fields:
            values["region"] = self._choose_region(country, rng)
        if RequestedDemographicField.URBANICITY in requested_fields:
            values["urbanicity"] = self._choose_enum(
                RequestedDemographicField.URBANICITY, list(Urbanicity), rng, config
            )
        if RequestedDemographicField.FAMILY_STATUS in requested_fields:
            values["family_status"] = self._choose_family_status(age, rng, config)
        if RequestedDemographicField.NUMBER_OF_DEPENDANTS in requested_fields:
            values["number_of_dependants"] = number_of_dependants
        if RequestedDemographicField.HAS_CHILDREN in requested_fields:
            values["has_children"] = number_of_dependants > 0
        if RequestedDemographicField.HOUSEHOLD_SIZE in requested_fields:
            values["household_size"] = self._choose_household_size(number_of_dependants, rng)

        return Demographics(**values)

    def _choose_age(self, rng: random.Random) -> int:
        return rng.randint(13, 90)

    def _choose_country(
        self,
        requested_fields: set[RequestedDemographicField],
        rng: random.Random,
        config: PersonaGenerationConfig,
    ) -> EuropeanCountry:
        if RequestedDemographicField.REGION in requested_fields:
            return self._choose_enum(
                RequestedDemographicField.COUNTRY,
                _countries_with_subdivisions(),
                rng,
                config,
            )
        return self._choose_enum(
            RequestedDemographicField.COUNTRY, list(EuropeanCountry), rng, config
        )

    def _choose_region(self, country: EuropeanCountry, rng: random.Random) -> str:
        subdivisions = _subdivisions_for_country(country)
        return rng.choice(subdivisions)

    def _choose_education_level(
        self, age: int, rng: random.Random, config: PersonaGenerationConfig
    ) -> EducationLevel:
        if age < 18:
            return self._choose_enum(
                RequestedDemographicField.EDUCATION_LEVEL,
                [EducationLevel.PRIMARY, EducationLevel.SECONDARY, EducationLevel.OTHER],
                rng,
                config,
            )
        options = list(EducationLevel)
        if age < 25:
            options.remove(EducationLevel.DOCTORATE)
        return self._choose_enum(RequestedDemographicField.EDUCATION_LEVEL, options, rng, config)

    def _choose_employment_status(
        self, age: int, rng: random.Random, config: PersonaGenerationConfig
    ) -> EmploymentStatus:
        if age < 18:
            return self._choose_enum(
                RequestedDemographicField.EMPLOYMENT_STATUS,
                [
                    EmploymentStatus.STUDENT,
                    EmploymentStatus.UNEMPLOYED,
                    EmploymentStatus.CAREGIVER,
                    EmploymentStatus.OTHER,
                ],
                rng,
                config,
            )
        options = list(EmploymentStatus)
        if age < 55:
            options.remove(EmploymentStatus.RETIRED)
        return self._choose_enum(RequestedDemographicField.EMPLOYMENT_STATUS, options, rng, config)

    def _choose_family_status(
        self, age: int, rng: random.Random, config: PersonaGenerationConfig
    ) -> FamilyStatus:
        if age < 18:
            return self._choose_enum(
                RequestedDemographicField.FAMILY_STATUS,
                [FamilyStatus.SINGLE, FamilyStatus.OTHER],
                rng,
                config,
            )
        return self._choose_enum(
            RequestedDemographicField.FAMILY_STATUS, list(FamilyStatus), rng, config
        )

    def _choose_number_of_dependants(self, age: int, rng: random.Random) -> int:
        if age < 18:
            return 0
        return rng.randint(0, 4)

    def _choose_household_size(self, number_of_dependants: int, rng: random.Random) -> int:
        return rng.randint(1 + number_of_dependants, 1 + number_of_dependants + 4)

    def _choose_enum[T: StrEnum](
        self,
        field: RequestedDemographicField,
        options: Sequence[T],
        rng: random.Random,
        config: PersonaGenerationConfig,
    ) -> T:
        probabilities = config.field_probabilities.get(field)
        if probabilities is None:
            return rng.choice(list(options))

        by_value = {option.value: option for option in options}
        missing_values = sorted(value for value in probabilities if value not in by_value)
        if missing_values:
            msg = (
                f"{field.value} configured values are unavailable for this persona context: "
                f"{', '.join(missing_values)}"
            )
            raise ValueError(msg)

        selected_values = list(probabilities)
        selected_options = [by_value[value] for value in selected_values]
        weights = [probabilities[value] for value in selected_values]
        return rng.choices(selected_options, weights=weights, k=1)[0]


def _countries_with_subdivisions() -> list[EuropeanCountry]:
    return [
        country
        for country in EuropeanCountry
        if pycountry.subdivisions.get(country_code=country.value)
    ]


def _subdivisions_for_country(country: EuropeanCountry) -> list[str]:
    subdivisions = pycountry.subdivisions.get(country_code=country.value)
    if not subdivisions:
        raise ValueError(f"No pycountry subdivisions available for {country.value}")
    return [subdivision.name for subdivision in subdivisions]
