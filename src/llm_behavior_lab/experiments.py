from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator

from llm_behavior_lab.personas.factory import (
    PersonaBatch,
    PersonaFactory,
    PersonaFactoryRequest,
    PersonaGenerationConfig,
    RequestedDemographicField,
)
from llm_behavior_lab.protocols import ExperimentProtocol, expand_protocol_personas
from llm_behavior_lab.storage import (
    validate_experiment_id,
    write_jsonl_records,
    write_persona_batch_jsonl,
    write_persona_batch_jsonl_at_path,
)


class PersonaDesign(BaseModel):
    """Configuration for deterministic demographic persona materialization."""

    count: int = Field(default=100, ge=1)
    seed: int | None = None
    requested_fields: set[RequestedDemographicField] = Field(
        default_factory=lambda: set(RequestedDemographicField)
    )
    generation_config: PersonaGenerationConfig = Field(default_factory=PersonaGenerationConfig)


class ProviderDesign(BaseModel):
    """Non-secret provider and model settings persisted in an experiment design."""

    model: str
    base_url: str
    temperature: float = 0.0
    timeout_seconds: float = 60.0
    seed: int | None = None
    supports_structured_outputs: bool = False
    supports_logprobs: bool = True


class ScaleProcedureDesign(BaseModel):
    """Questionnaire administration and scoring inputs."""

    kind: Literal["scale"] = "scale"
    questionnaire_id: str
    questionnaire_parameters: dict[str, str] = Field(default_factory=dict)
    scoring_model_id: str | None = None
    context: str | None = None


class TaskProcedureDesign(BaseModel):
    """Stateful behavioral-task inputs."""

    kind: Literal["task"] = "task"
    task_id: str
    task_config: dict[str, Any] = Field(default_factory=dict)


ProcedureDesign = Annotated[
    ScaleProcedureDesign | TaskProcedureDesign,
    Field(discriminator="kind"),
]


class ExperimentDesign(BaseModel):
    """Immutable inputs connecting experiment preparation, execution, and scoring."""

    version: str = "2.0"
    experiment_id: str
    procedure: ProcedureDesign
    personas: PersonaDesign | None = None
    protocol: ExperimentProtocol | None = None
    provider: ProviderDesign

    @model_validator(mode="after")
    def validate_design(self) -> "ExperimentDesign":
        validate_experiment_id(self.experiment_id)
        if (self.personas is None) == (self.protocol is None):
            raise ValueError("exactly one of personas or protocol must be configured")
        return self


def create_experiment_design(project_root: Path, design: ExperimentDesign) -> Path:
    """Write a new experiment design without overwriting an existing manifest."""
    path = project_root / "experiments" / design.experiment_id / "design.json"
    if path.exists():
        raise FileExistsError(f"experiment design already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(design.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_experiment_design(project_root: Path, experiment_id: str) -> ExperimentDesign:
    """Load and validate an experiment's persisted design manifest."""
    path = project_root / "experiments" / validate_experiment_id(experiment_id) / "design.json"
    return ExperimentDesign.model_validate_json(path.read_text(encoding="utf-8"))


def materialize_personas(project_root: Path, design: ExperimentDesign) -> PersonaBatch:
    """Create the exact persona batch consumed by the later run stage."""
    experiment_root = project_root / "experiments" / design.experiment_id
    personas_path = experiment_root / "personas.jsonl"
    if personas_path.exists():
        raise FileExistsError(f"personas already exist: {personas_path}")
    if design.protocol is not None:
        expansion = expand_protocol_personas(design.protocol, design.experiment_id)
        write_persona_batch_jsonl_at_path(
            experiment_root / "base_personas.jsonl", expansion.base_personas
        )
        write_persona_batch_jsonl_at_path(personas_path, expansion.personas)
        (experiment_root / "protocol.json").write_text(
            design.protocol.model_dump_json(indent=2), encoding="utf-8"
        )
        write_jsonl_records(experiment_root / "protocol_assignments.jsonl", expansion.assignments)
        return expansion.personas

    if design.personas is None:
        raise ValueError("persona design is required when no protocol is configured")
    batch = PersonaFactory().create_demographics_batch(
        PersonaFactoryRequest(
            count=design.personas.count,
            requested_fields=design.personas.requested_fields,
            seed=design.personas.seed,
            experiment_id=design.experiment_id,
            generation_config=design.personas.generation_config,
        )
    )
    write_persona_batch_jsonl(project_root, batch)
    return batch


def load_personas(project_root: Path, experiment_id: str) -> PersonaBatch:
    """Load the persisted persona batch required by the run stage."""
    path = project_root / "experiments" / validate_experiment_id(experiment_id) / "personas.jsonl"
    return PersonaBatch.model_validate_json(path.read_text(encoding="utf-8"))
