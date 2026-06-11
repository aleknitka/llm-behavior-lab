from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from enum import StrEnum
from typing import Any
from uuid import UUID

import pycountry
from pydantic import BaseModel, Field, field_validator, model_validator

from llm_behavior_lab.personas.dimensions import (
    AffluenceLevel,
    Demographics,
    EducationLevel,
    EmploymentStatus,
    EuropeanCountry,
    FamilyStatus,
    Gender,
    Urbanicity,
)
from llm_behavior_lab.personas.value_specs import (
    PersonaFieldValue,
    RandUniformRange,
    stable_random,
)
from llm_behavior_lab.storage import generate_experiment_id, validate_experiment_id


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
    requested_fields: list[RequestedDemographicField]
    seed: int | None = None


class PersonaBatch(BaseModel):
    experiment_id: str
    metadata: PersonaBatchMetadata
    personas: list[GeneratedPersona] = Field(min_length=1)

    def __len__(self) -> int:
        return len(self.personas)


PERSONA_ENUM_FIELDS: dict[RequestedDemographicField, type[StrEnum]] = {
    RequestedDemographicField.GENDER: Gender,
    RequestedDemographicField.EDUCATION_LEVEL: EducationLevel,
    RequestedDemographicField.EMPLOYMENT_STATUS: EmploymentStatus,
    RequestedDemographicField.AFFLUENCE_LEVEL: AffluenceLevel,
    RequestedDemographicField.COUNTRY: EuropeanCountry,
    RequestedDemographicField.URBANICITY: Urbanicity,
    RequestedDemographicField.FAMILY_STATUS: FamilyStatus,
}

PERSONA_RANGE_FIELDS = {
    RequestedDemographicField.AGE,
    RequestedDemographicField.HOUSEHOLD_SIZE,
    RequestedDemographicField.NUMBER_OF_DEPENDANTS,
}


class PersonaGenerationConfig(BaseModel):
    field_values: dict[RequestedDemographicField, PersonaFieldValue] = Field(default_factory=dict)
    field_probabilities: dict[RequestedDemographicField, dict[str, float]] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def validate_configuration(self) -> PersonaGenerationConfig:
        conflicts = sorted(
            field.value for field in self.field_values.keys() & self.field_probabilities.keys()
        )
        if conflicts:
            msg = f"fields cannot define both values and probabilities: {', '.join(conflicts)}"
            raise ValueError(msg)

        fixed_values: dict[RequestedDemographicField, Any] = {}
        for field, value in self.field_values.items():
            if isinstance(value, RandUniformRange):
                if field not in PERSONA_RANGE_FIELDS:
                    msg = f"{field.value} does not support range generators"
                    raise ValueError(msg)
                _validate_demographic_value(field, value.left)
                _validate_demographic_value(field, value.right)
            else:
                fixed_values[field] = _validate_demographic_value(field, value)
        _validate_fixed_demographic_combinations(fixed_values)

        for field, probabilities in self.field_probabilities.items():
            enum_type = PERSONA_ENUM_FIELDS.get(field)
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

    @model_validator(mode="after")
    def validate_generation_config(self) -> PersonaFactoryRequest:
        configured_fields = (
            self.generation_config.field_values.keys()
            | self.generation_config.field_probabilities.keys()
        )
        missing_fields = sorted(field.value for field in configured_fields - self.requested_fields)
        if missing_fields:
            msg = f"configured fields must be requested: {', '.join(missing_fields)}"
            raise ValueError(msg)

        has_generator = any(
            isinstance(value, RandUniformRange)
            for value in self.generation_config.field_values.values()
        )
        if has_generator and self.seed is None:
            raise ValueError("seed is required when field values contain a generator")
        return self


class PersonaFactory:
    def create_demographics_batch(self, request: PersonaFactoryRequest) -> PersonaBatch:
        rng = random.Random(request.seed)  # nosec B311 - deterministic sampling only.
        experiment_id = request.experiment_id or generate_experiment_id(request.seed)
        personas = []
        for _ in range(request.count):
            subject_id = UUID(int=rng.getrandbits(128), version=4)
            personas.append(
                GeneratedPersona(
                    subject_id=subject_id,
                    features=self._create_demographics(
                        requested_fields=request.requested_fields,
                        rng=rng,
                        config=request.generation_config,
                        seed=request.seed,
                        experiment_id=experiment_id,
                        subject_id=subject_id,
                    ),
                )
            )
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
        seed: int | None,
        experiment_id: str,
        subject_id: UUID,
    ) -> Demographics:
        configured = self._resolve_field_values(
            config=config,
            seed=seed,
            experiment_id=experiment_id,
            subject_id=subject_id,
        )
        age = self._configured_value(
            configured,
            RequestedDemographicField.AGE,
            lambda: self._choose_age(rng),
        )
        number_of_dependants = self._configured_value(
            configured,
            RequestedDemographicField.NUMBER_OF_DEPENDANTS,
            lambda: self._choose_number_of_dependants(age, rng),
        )
        country = self._configured_value(
            configured,
            RequestedDemographicField.COUNTRY,
            lambda: self._choose_country(requested_fields, rng, config),
        )
        values: dict[str, Any] = {}

        if RequestedDemographicField.AGE in requested_fields:
            values["age"] = age
        if RequestedDemographicField.GENDER in requested_fields:
            values["gender"] = self._configured_value(
                configured,
                RequestedDemographicField.GENDER,
                lambda: self._choose_enum(
                    RequestedDemographicField.GENDER, list(Gender), rng, config
                ),
            )
        if RequestedDemographicField.EDUCATION_LEVEL in requested_fields:
            values["education_level"] = self._configured_value(
                configured,
                RequestedDemographicField.EDUCATION_LEVEL,
                lambda: self._choose_education_level(age, rng, config),
            )
        if RequestedDemographicField.EMPLOYMENT_STATUS in requested_fields:
            values["employment_status"] = self._configured_value(
                configured,
                RequestedDemographicField.EMPLOYMENT_STATUS,
                lambda: self._choose_employment_status(age, rng, config),
            )
        if RequestedDemographicField.AFFLUENCE_LEVEL in requested_fields:
            values["affluence_level"] = self._configured_value(
                configured,
                RequestedDemographicField.AFFLUENCE_LEVEL,
                lambda: self._choose_enum(
                    RequestedDemographicField.AFFLUENCE_LEVEL,
                    list(AffluenceLevel),
                    rng,
                    config,
                ),
            )
        if RequestedDemographicField.COUNTRY in requested_fields:
            values["country"] = country
        if RequestedDemographicField.REGION in requested_fields:
            values["region"] = self._configured_value(
                configured,
                RequestedDemographicField.REGION,
                lambda: self._choose_region(country, rng),
            )
        if RequestedDemographicField.URBANICITY in requested_fields:
            values["urbanicity"] = self._configured_value(
                configured,
                RequestedDemographicField.URBANICITY,
                lambda: self._choose_enum(
                    RequestedDemographicField.URBANICITY, list(Urbanicity), rng, config
                ),
            )
        if RequestedDemographicField.FAMILY_STATUS in requested_fields:
            values["family_status"] = self._configured_value(
                configured,
                RequestedDemographicField.FAMILY_STATUS,
                lambda: self._choose_family_status(age, rng, config),
            )
        if RequestedDemographicField.NUMBER_OF_DEPENDANTS in requested_fields:
            values["number_of_dependants"] = number_of_dependants
        if RequestedDemographicField.HAS_CHILDREN in requested_fields:
            values["has_children"] = self._configured_value(
                configured,
                RequestedDemographicField.HAS_CHILDREN,
                lambda: number_of_dependants > 0,
            )
        if RequestedDemographicField.HOUSEHOLD_SIZE in requested_fields:
            values["household_size"] = self._configured_value(
                configured,
                RequestedDemographicField.HOUSEHOLD_SIZE,
                lambda: self._choose_household_size(number_of_dependants, rng),
            )

        return Demographics(**values)

    def _resolve_field_values(
        self,
        config: PersonaGenerationConfig,
        seed: int | None,
        experiment_id: str,
        subject_id: UUID,
    ) -> dict[RequestedDemographicField, Any]:
        resolved: dict[RequestedDemographicField, Any] = {}
        for field, value in config.field_values.items():
            if isinstance(value, RandUniformRange):
                if seed is None:
                    raise ValueError("seed is required when field values contain a generator")
                value = value.sample(stable_random(seed, experiment_id, subject_id, field.value))
            resolved[field] = _validate_demographic_value(field, value)
        return resolved

    def _configured_value(
        self,
        configured: dict[RequestedDemographicField, Any],
        field: RequestedDemographicField,
        default: Callable[[], Any],
    ) -> Any:
        if field in configured:
            return configured[field]
        return default()

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


def _validate_demographic_value(field: RequestedDemographicField, value: Any) -> Any:
    try:
        demographics = Demographics.model_validate({field.value: value})
    except ValueError as exc:
        msg = f"{field.value} has unsupported value {value!r}"
        raise ValueError(msg) from exc
    return getattr(demographics, field.value)


def _validate_fixed_demographic_combinations(
    values: dict[RequestedDemographicField, Any],
) -> None:
    age = values.get(RequestedDemographicField.AGE)
    dependants = values.get(RequestedDemographicField.NUMBER_OF_DEPENDANTS)
    has_children = values.get(RequestedDemographicField.HAS_CHILDREN)
    household_size = values.get(RequestedDemographicField.HOUSEHOLD_SIZE)
    education = values.get(RequestedDemographicField.EDUCATION_LEVEL)
    employment = values.get(RequestedDemographicField.EMPLOYMENT_STATUS)

    incompatible = False
    if age is not None and age < 18:
        incompatible = (
            (dependants is not None and dependants > 0)
            or has_children is True
            or education
            not in {None, EducationLevel.PRIMARY, EducationLevel.SECONDARY, EducationLevel.OTHER}
            or employment
            not in {
                None,
                EmploymentStatus.STUDENT,
                EmploymentStatus.UNEMPLOYED,
                EmploymentStatus.CAREGIVER,
                EmploymentStatus.OTHER,
            }
        )
    if age is not None and age < 25 and education is EducationLevel.DOCTORATE:
        incompatible = True
    if age is not None and age < 55 and employment is EmploymentStatus.RETIRED:
        incompatible = True
    if dependants is not None and has_children is not None:
        incompatible = incompatible or has_children is not (dependants > 0)
    if dependants is not None and household_size is not None:
        incompatible = incompatible or household_size < 1 + dependants

    if incompatible:
        raise ValueError("configured field values are incompatible with demographic constraints")
