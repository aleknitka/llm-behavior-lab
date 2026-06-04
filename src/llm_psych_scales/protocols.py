from __future__ import annotations

import hashlib
import itertools
import random
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from llm_psych_scales.personas.dimensions import Demographics
from llm_psych_scales.personas.factory import (
    GeneratedPersona,
    PersonaBatch,
    PersonaBatchMetadata,
    PersonaFactory,
    PersonaFactoryRequest,
    PersonaGenerationConfig,
    RequestedDemographicField,
)


class ProtocolFactorLevel(BaseModel):
    id: str = Field(min_length=1)
    value: Any
    label: str | None = None


class ProtocolFactor(BaseModel):
    name: str = Field(min_length=1)
    field: RequestedDemographicField
    levels: list[ProtocolFactorLevel] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_factor(self) -> ProtocolFactor:
        level_ids = [level.id for level in self.levels]
        if len(set(level_ids)) != len(level_ids):
            msg = f"factor {self.name!r} has duplicate level ids"
            raise ValueError(msg)
        for level in self.levels:
            try:
                Demographics(**{self.field.value: level.value})
            except ValueError as exc:
                msg = f"{self.field.value} has unsupported value {level.value!r}"
                raise ValueError(msg) from exc
        return self


class ExperimentProtocol(BaseModel):
    version: str = "1.0"
    name: str = Field(min_length=1)
    design: Literal["paired_factorial"] = "paired_factorial"
    base_persona_count: int = Field(ge=1)
    seed: int | None = None
    iterations: int = Field(default=1, ge=1)
    requested_fields: set[RequestedDemographicField] = Field(min_length=1)
    base_persona_config: PersonaGenerationConfig = Field(default_factory=PersonaGenerationConfig)
    factors: list[ProtocolFactor] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_protocol(self) -> ExperimentProtocol:
        factor_names = [factor.name for factor in self.factors]
        if len(set(factor_names)) != len(factor_names):
            msg = "protocol has duplicate factor names"
            raise ValueError(msg)

        factor_fields = [factor.field for factor in self.factors]
        if len(set(factor_fields)) != len(factor_fields):
            msg = "protocol has duplicate factor fields"
            raise ValueError(msg)

        missing_fields = sorted(
            field.value for field in factor_fields if field not in self.requested_fields
        )
        if missing_fields:
            msg = f"factor fields must be requested: {', '.join(missing_fields)}"
            raise ValueError(msg)
        return self


class ProtocolAssignment(BaseModel):
    subject_id: str
    base_subject_id: str
    condition_id: str
    iteration_index: int = Field(ge=1)
    factor_values: dict[str, Any]
    factor_level_ids: dict[str, str]


class ProtocolExpansion(BaseModel):
    protocol: ExperimentProtocol
    base_personas: PersonaBatch
    personas: PersonaBatch
    assignments: list[ProtocolAssignment]


def expand_protocol_personas(protocol: ExperimentProtocol, experiment_id: str) -> ProtocolExpansion:
    factor_fields = {factor.field for factor in protocol.factors}
    base_requested_fields = protocol.requested_fields - factor_fields
    base_personas = _create_base_personas(protocol, experiment_id, base_requested_fields)

    expanded_personas: list[GeneratedPersona] = []
    assignments: list[ProtocolAssignment] = []
    for base_persona in base_personas.personas:
        base_features = base_persona.features
        for condition_id, factor_values, factor_level_ids in _condition_grid(protocol.factors):
            manipulated_features = Demographics.model_validate(
                {
                    **base_features.model_dump(mode="json", exclude_none=True),
                    **factor_values,
                }
            )
            for iteration_index in range(1, protocol.iterations + 1):
                subject_id = _expanded_subject_id(
                    experiment_id=experiment_id,
                    base_subject_id=str(base_persona.subject_id),
                    condition_id=condition_id,
                    iteration_index=iteration_index,
                )
                expanded_personas.append(
                    GeneratedPersona(subject_id=subject_id, features=manipulated_features)
                )
                assignments.append(
                    ProtocolAssignment(
                        subject_id=str(subject_id),
                        base_subject_id=str(base_persona.subject_id),
                        condition_id=condition_id,
                        iteration_index=iteration_index,
                        factor_values={
                            field: _json_value(value) for field, value in factor_values.items()
                        },
                        factor_level_ids=factor_level_ids,
                    )
                )

    expanded_batch = PersonaBatch(
        experiment_id=experiment_id,
        metadata=PersonaBatchMetadata(
            experiment_id=experiment_id,
            persona_count=len(expanded_personas),
            requested_fields=sorted(protocol.requested_fields, key=lambda field: field.value),
            seed=protocol.seed,
        ),
        personas=expanded_personas,
    )
    return ProtocolExpansion(
        protocol=protocol,
        base_personas=base_personas,
        personas=expanded_batch,
        assignments=assignments,
    )


def _create_base_personas(
    protocol: ExperimentProtocol,
    experiment_id: str,
    requested_fields: set[RequestedDemographicField],
) -> PersonaBatch:
    if requested_fields:
        return PersonaFactory().create_demographics_batch(
            PersonaFactoryRequest(
                count=protocol.base_persona_count,
                requested_fields=requested_fields,
                seed=protocol.seed,
                experiment_id=experiment_id,
                generation_config=protocol.base_persona_config,
            )
        )

    rng = random.Random(protocol.seed)  # nosec B311 - deterministic sampling only.
    personas = [
        GeneratedPersona(
            subject_id=UUID(int=rng.getrandbits(128), version=4),
            features=Demographics(),
        )
        for _ in range(protocol.base_persona_count)
    ]
    return PersonaBatch(
        experiment_id=experiment_id,
        metadata=PersonaBatchMetadata(
            experiment_id=experiment_id,
            persona_count=protocol.base_persona_count,
            requested_fields=[],
            seed=protocol.seed,
        ),
        personas=personas,
    )


def _condition_grid(
    factors: list[ProtocolFactor],
) -> list[tuple[str, dict[str, Any], dict[str, str]]]:
    conditions: list[tuple[str, dict[str, Any], dict[str, str]]] = []
    for levels in itertools.product(*(factor.levels for factor in factors)):
        condition_parts = []
        factor_values: dict[str, Any] = {}
        factor_level_ids: dict[str, str] = {}
        for factor, level in zip(factors, levels, strict=True):
            condition_parts.append(f"{factor.name}-{level.id}")
            factor_values[factor.field.value] = level.value
            factor_level_ids[factor.field.value] = level.id
        conditions.append(("__".join(condition_parts), factor_values, factor_level_ids))
    return conditions


def _expanded_subject_id(
    experiment_id: str,
    base_subject_id: str,
    condition_id: str,
    iteration_index: int,
) -> UUID:
    key = f"{experiment_id}:{base_subject_id}:{condition_id}:{iteration_index}"
    digest = hashlib.sha256(key.encode()).digest()
    return UUID(bytes=digest[:16], version=4)


def _json_value(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    return value
