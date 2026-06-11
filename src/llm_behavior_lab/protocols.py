from __future__ import annotations

import hashlib
import itertools
import json
import random
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from llm_behavior_lab.personas.dimensions import Demographics
from llm_behavior_lab.personas.factory import (
    GeneratedPersona,
    PersonaBatch,
    PersonaBatchMetadata,
    PersonaFactory,
    PersonaFactoryRequest,
    PersonaGenerationConfig,
    RequestedDemographicField,
)
from llm_behavior_lab.personas.value_specs import (
    PersonaFieldValue,
    RandUniformRange,
    stable_random,
)

_RANGE_FACTOR_FIELDS = {
    RequestedDemographicField.AGE,
    RequestedDemographicField.HOUSEHOLD_SIZE,
    RequestedDemographicField.NUMBER_OF_DEPENDANTS,
}


class ProtocolProvider(BaseModel):
    """Shared non-secret provider settings for one experiment protocol."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    base_url: str = Field(min_length=1)
    temperature: float = 0.0
    timeout_seconds: float = Field(default=60.0, gt=0)
    supports_structured_outputs: bool = False
    supports_logprobs: bool = True


class ProtocolQuestionnaireStep(BaseModel):
    """One questionnaire step in an ordered protocol."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    kind: Literal["questionnaire"] = "questionnaire"
    questionnaire_id: str = Field(min_length=1)
    questionnaire_parameters: dict[str, str] = Field(default_factory=dict)
    scoring_model_id: str | None = None
    context: str | None = None
    history: Literal["reset", "inherit"] = "reset"


class ProtocolTaskStep(BaseModel):
    """One stateful behavioral-task step in an ordered protocol."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    kind: Literal["task"] = "task"
    task_id: str = Field(min_length=1)
    task_config: dict[str, Any] = Field(default_factory=dict)
    history: Literal["reset", "inherit"] = "reset"


ProtocolStep = Annotated[
    ProtocolQuestionnaireStep | ProtocolTaskStep,
    Field(discriminator="kind"),
]


class ProtocolPersonaDesign(BaseModel):
    """Persona cohort inputs, optionally expanded through a factorial design."""

    model_config = ConfigDict(extra="forbid")

    count: int = Field(default=100, ge=1)
    requested_fields: set[RequestedDemographicField] = Field(
        default_factory=lambda: set(RequestedDemographicField),
        min_length=1,
    )
    generation_config: PersonaGenerationConfig = Field(default_factory=PersonaGenerationConfig)
    factorial: ExperimentProtocol | None = None

    @model_validator(mode="after")
    def validate_personas(self) -> ProtocolPersonaDesign:
        PersonaFactoryRequest(
            count=self.count,
            requested_fields=self.requested_fields,
            seed=0,
            generation_config=self.generation_config,
        )
        return self


class UnifiedExperimentProtocol(BaseModel):
    """Canonical immutable configuration for a multi-step experiment."""

    model_config = ConfigDict(extra="forbid")

    version: str = "1.0"
    experiment_id: str
    name: str = Field(min_length=1)
    persona_seed: int | None = None
    run_seed: int | None = None
    personas: ProtocolPersonaDesign
    provider: ProtocolProvider
    steps: list[ProtocolStep] = Field(min_length=1)

    @field_validator("experiment_id")
    @classmethod
    def validate_experiment_id(cls, value: str) -> str:
        from llm_behavior_lab.storage import validate_experiment_id

        return validate_experiment_id(value)

    @model_validator(mode="after")
    def validate_steps(self) -> UnifiedExperimentProtocol:
        step_ids = [step.id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("protocol has duplicate step ids")
        return self


class CompatibleProtocol(BaseModel):
    """Validated protocol payload plus the format used to load it."""

    source: Literal["protocol.json", "factor_protocol"]
    protocol: UnifiedExperimentProtocol | ExperimentProtocol


class ProtocolFactorLevel(BaseModel):
    id: str = Field(min_length=1)
    value: PersonaFieldValue
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
            if isinstance(level.value, RandUniformRange):
                if self.field not in _RANGE_FACTOR_FIELDS:
                    msg = f"{self.field.value} does not support range generators"
                    raise ValueError(msg)
                _validate_factor_value(self.field, level.value.left)
                _validate_factor_value(self.field, level.value.right)
            else:
                _validate_factor_value(self.field, level.value)
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

        configured_values = self.base_persona_config.field_values.values()
        factor_values = (level.value for factor in self.factors for level in factor.levels)
        if self.seed is None and any(
            isinstance(value, RandUniformRange)
            for value in itertools.chain(configured_values, factor_values)
        ):
            raise ValueError("seed is required when protocol values contain a generator")
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
        for condition_id, levels in _condition_grid(protocol.factors):
            factor_level_ids = {factor.field.value: level.id for factor, level in levels}
            for iteration_index in range(1, protocol.iterations + 1):
                factor_values = _resolve_factor_values(
                    protocol=protocol,
                    experiment_id=experiment_id,
                    base_subject_id=str(base_persona.subject_id),
                    levels=levels,
                    iteration_index=iteration_index,
                )
                manipulated_features = Demographics.model_validate(
                    {
                        **base_features.model_dump(mode="json", exclude_none=True),
                        **factor_values,
                    }
                )
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
) -> list[tuple[str, tuple[tuple[ProtocolFactor, ProtocolFactorLevel], ...]]]:
    conditions = []
    for levels in itertools.product(*(factor.levels for factor in factors)):
        condition_parts = []
        selected_levels = []
        for factor, level in zip(factors, levels, strict=True):
            condition_parts.append(f"{factor.name}-{level.id}")
            selected_levels.append((factor, level))
        conditions.append(("__".join(condition_parts), tuple(selected_levels)))
    return conditions


def _resolve_factor_values(
    protocol: ExperimentProtocol,
    experiment_id: str,
    base_subject_id: str,
    levels: tuple[tuple[ProtocolFactor, ProtocolFactorLevel], ...],
    iteration_index: int,
) -> dict[str, Any]:
    factor_values: dict[str, Any] = {}
    for factor, level in levels:
        value = level.value
        if isinstance(value, RandUniformRange):
            if protocol.seed is None:
                raise ValueError("seed is required when protocol values contain a generator")
            value = value.sample(
                stable_random(
                    protocol.seed,
                    experiment_id,
                    base_subject_id,
                    factor.name,
                    factor.field.value,
                    level.id,
                    iteration_index,
                )
            )
        factor_values[factor.field.value] = value
    return factor_values


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


def _validate_factor_value(field: RequestedDemographicField, value: Any) -> None:
    try:
        Demographics.model_validate({field.value: value})
    except ValueError as exc:
        msg = f"{field.value} has unsupported value {value!r}"
        raise ValueError(msg) from exc


def protocol_fingerprint(protocol: UnifiedExperimentProtocol) -> str:
    """Hash protocol identity while allowing run and persona seed overrides."""
    payload = protocol.model_dump(mode="json", exclude={"persona_seed", "run_seed"})
    factorial = payload.get("personas", {}).get("factorial")
    if isinstance(factorial, dict):
        factorial.pop("seed", None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_compatible_protocol(path: Path) -> CompatibleProtocol:
    """Load a canonical protocol or a legacy factor-only protocol file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "experiment_id" in payload and "steps" in payload:
        return CompatibleProtocol(
            source="protocol.json",
            protocol=UnifiedExperimentProtocol.model_validate(payload),
        )
    return CompatibleProtocol(
        source="factor_protocol",
        protocol=ExperimentProtocol.model_validate(payload),
    )


ProtocolPersonaDesign.model_rebuild()
UnifiedExperimentProtocol.model_rebuild()
CompatibleProtocol.model_rebuild()
