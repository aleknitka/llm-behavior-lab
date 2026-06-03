from __future__ import annotations

import random
from enum import StrEnum
from typing import Any
from uuid import UUID

import pycountry
from pydantic import BaseModel, Field, field_validator

from llm_psych_scales.personas.dimensions import (
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


class PersonaFactoryRequest(BaseModel):
    count: int = Field(ge=1)
    requested_fields: set[RequestedDemographicField] = Field(min_length=1)
    seed: int | None = None
    experiment_id: str | None = None

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
                features=self._create_demographics(request.requested_fields, rng),
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
        self, requested_fields: set[RequestedDemographicField], rng: random.Random
    ) -> Demographics:
        age = self._choose_age(rng)
        number_of_dependants = self._choose_number_of_dependants(age, rng)
        country = self._choose_country(requested_fields, rng)
        values: dict[str, Any] = {}

        if RequestedDemographicField.AGE in requested_fields:
            values["age"] = age
        if RequestedDemographicField.GENDER in requested_fields:
            values["gender"] = rng.choice(list(Gender))
        if RequestedDemographicField.EDUCATION_LEVEL in requested_fields:
            values["education_level"] = self._choose_education_level(age, rng)
        if RequestedDemographicField.EMPLOYMENT_STATUS in requested_fields:
            values["employment_status"] = self._choose_employment_status(age, rng)
        if RequestedDemographicField.COUNTRY in requested_fields:
            values["country"] = country
        if RequestedDemographicField.REGION in requested_fields:
            values["region"] = self._choose_region(country, rng)
        if RequestedDemographicField.URBANICITY in requested_fields:
            values["urbanicity"] = rng.choice(list(Urbanicity))
        if RequestedDemographicField.FAMILY_STATUS in requested_fields:
            values["family_status"] = self._choose_family_status(age, rng)
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
        self, requested_fields: set[RequestedDemographicField], rng: random.Random
    ) -> EuropeanCountry:
        if RequestedDemographicField.REGION in requested_fields:
            return rng.choice(_countries_with_subdivisions())
        return rng.choice(list(EuropeanCountry))

    def _choose_region(self, country: EuropeanCountry, rng: random.Random) -> str:
        subdivisions = _subdivisions_for_country(country)
        return rng.choice(subdivisions)

    def _choose_education_level(self, age: int, rng: random.Random) -> EducationLevel:
        if age < 18:
            return rng.choice(
                [EducationLevel.PRIMARY, EducationLevel.SECONDARY, EducationLevel.OTHER]
            )
        options = list(EducationLevel)
        if age < 25:
            options.remove(EducationLevel.DOCTORATE)
        return rng.choice(options)

    def _choose_employment_status(self, age: int, rng: random.Random) -> EmploymentStatus:
        if age < 18:
            return rng.choice(
                [
                    EmploymentStatus.STUDENT,
                    EmploymentStatus.UNEMPLOYED,
                    EmploymentStatus.CAREGIVER,
                    EmploymentStatus.OTHER,
                ]
            )
        options = list(EmploymentStatus)
        if age < 55:
            options.remove(EmploymentStatus.RETIRED)
        return rng.choice(options)

    def _choose_family_status(self, age: int, rng: random.Random) -> FamilyStatus:
        if age < 18:
            return rng.choice([FamilyStatus.SINGLE, FamilyStatus.OTHER])
        return rng.choice(list(FamilyStatus))

    def _choose_number_of_dependants(self, age: int, rng: random.Random) -> int:
        if age < 18:
            return 0
        return rng.randint(0, 4)

    def _choose_household_size(self, number_of_dependants: int, rng: random.Random) -> int:
        return rng.randint(1 + number_of_dependants, 1 + number_of_dependants + 4)


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
