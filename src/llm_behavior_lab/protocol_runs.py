from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from llm_behavior_lab.behavioral_tasks.batch import run_persisted_task_batch_async
from llm_behavior_lab.behavioral_tasks.catalog import resolve_behavioral_task
from llm_behavior_lab.client import Message, OpenAiChatClient
from llm_behavior_lab.experiments import (
    ExperimentDesign,
    ScaleProcedureDesign,
    TaskProcedureDesign,
)
from llm_behavior_lab.models import ModelSettings, ProviderCapabilities
from llm_behavior_lab.personas.factory import (
    PersonaBatch,
    PersonaFactory,
    PersonaFactoryRequest,
)
from llm_behavior_lab.protocols import (
    ExperimentProtocol,
    ProtocolAssignment,
    ProtocolAssignments,
    ProtocolQuestionnaireStep,
    UnifiedExperimentProtocol,
    expand_protocol_personas,
    protocol_fingerprint,
)
from llm_behavior_lab.questionnaires.catalog import resolve_questionnaire
from llm_behavior_lab.responses.base import (
    ChatMessage,
    ProviderSnapshot,
    ResponseStatus,
    RunRecord,
)
from llm_behavior_lab.runner import run_persisted_persona_batch
from llm_behavior_lab.storage import (
    load_json_document,
    normalize_prefixed_uuid,
    resolve_compatible_snapshot_path,
    slugify_model_name,
    update_experiment_metadata,
    validate_experiment_id,
    write_json_document,
    write_persona_batch_at_path,
)


class CohortMetadata(BaseModel):
    cohort_id: str
    experiment_id: str
    protocol_fingerprint: str
    persona_seed: int | None = None
    persona_count: int = Field(ge=1)
    created_at: datetime


class ProtocolStepResult(BaseModel):
    step_id: str
    kind: Literal["questionnaire", "task"]
    status: ResponseStatus
    output_path: str


@dataclass(frozen=True)
class ProtocolExperimentCreation:
    protocol_path: Path
    cohort_id: str


@dataclass(frozen=True)
class ProtocolRunResult:
    run_id: str
    run_root: Path
    cohort_id: str
    step_results: list[ProtocolStepResult]


@dataclass(frozen=True)
class LoadedProtocolExperiment:
    protocol: UnifiedExperimentProtocol
    source: Literal["protocol.json", "design.json"]


def create_protocol_experiment(
    project_root: Path,
    protocol: UnifiedExperimentProtocol,
) -> ProtocolExperimentCreation:
    """Create one immutable canonical protocol and its initial persona cohort."""
    _validate_protocol_references(protocol)
    experiment_root = _experiment_root(project_root, protocol.experiment_id)
    protocol_path = experiment_root / "protocol.json"
    if protocol_path.exists() or (experiment_root / "design.json").exists():
        raise FileExistsError(f"experiment already exists: {experiment_root}")
    experiment_root.mkdir(parents=True, exist_ok=True)
    write_json_document(protocol_path, protocol)
    cohort_id = _create_cohort(
        experiment_root,
        protocol,
        protocol.persona_seed,
    )
    return ProtocolExperimentCreation(protocol_path=protocol_path, cohort_id=cohort_id)


def create_protocol_run(
    project_root: Path,
    protocol: UnifiedExperimentProtocol,
    *,
    cohort_id: str | None = None,
    persona_seed: int | None = None,
    run_seed: int | None = None,
    api_key: str = "lm-studio",  # pragma: allowlist secret
    execute: bool = True,
) -> ProtocolRunResult:
    """Create a distinct execution of an existing immutable protocol."""
    if cohort_id is not None and persona_seed is not None:
        raise ValueError("cohort_id and persona_seed are mutually exclusive")
    loaded = load_protocol_experiment(project_root, protocol.experiment_id)
    if protocol_fingerprint(loaded.protocol) != protocol_fingerprint(protocol):
        raise ValueError("protocol configuration differs; use a new experiment_id")

    experiment_root = _experiment_root(project_root, protocol.experiment_id)
    effective_run_seed = run_seed if run_seed is not None else protocol.run_seed
    if cohort_id is not None:
        resolved_cohort_id = cohort_id
        cohort_metadata = _load_cohort_metadata(
            _cohort_root(experiment_root, resolved_cohort_id) / "metadata.json"
        )
        if cohort_metadata.protocol_fingerprint != protocol_fingerprint(protocol):
            raise ValueError("cohort protocol fingerprint does not match experiment")
        effective_persona_seed = cohort_metadata.persona_seed
    else:
        effective_persona_seed = (
            persona_seed if persona_seed is not None else protocol.persona_seed
        )
        resolved_cohort_id = _find_cohort(
            experiment_root,
            effective_persona_seed,
        )
        if resolved_cohort_id is None:
            resolved_cohort_id = _create_cohort(
                experiment_root,
                protocol,
                effective_persona_seed,
            )
    cohort_root = _cohort_root(experiment_root, resolved_cohort_id)
    personas = _load_personas(cohort_root / "personas.json")
    assignments = _load_assignments(cohort_root / "protocol-assignments.json")

    started_at = datetime.now(UTC)
    run_id = _next_protocol_run_id(experiment_root, protocol.provider.model, started_at)
    run_root = experiment_root / run_id
    run_root.mkdir(parents=True)
    settings = ModelSettings(
        model=protocol.provider.model,
        provider_base_url=protocol.provider.base_url,
        temperature=protocol.provider.temperature,
        timeout_seconds=protocol.provider.timeout_seconds,
        seed=effective_run_seed,
        capabilities=ProviderCapabilities(
            supports_structured_outputs=protocol.provider.supports_structured_outputs,
            supports_logprobs=protocol.provider.supports_logprobs,
        ),
    )
    step_results: list[ProtocolStepResult] = []
    status = ResponseStatus.SKIPPED
    histories: dict[str, list[Message]] = {}
    if execute:
        client = OpenAiChatClient(api_key=api_key, base_url=protocol.provider.base_url)
        metadata = _response_metadata(
            protocol,
            resolved_cohort_id,
            effective_persona_seed,
            effective_run_seed,
            personas,
            assignments,
        )
        status = ResponseStatus.COMPLETED
        for step in protocol.steps:
            step_root = run_root / "steps" / step.id
            inherited = histories if step.history == "inherit" else None
            if isinstance(step, ProtocolQuestionnaireStep):
                result = run_persisted_persona_batch(
                    personas=personas,
                    questionnaire=resolve_questionnaire(
                        step.questionnaire_id,
                        step.questionnaire_parameters,
                    ),
                    settings=settings,
                    client=client,
                    project_root=project_root,
                    context=step.context,
                    response_metadata_by_subject={
                        subject_id: {**values, "step_id": step.id}
                        for subject_id, values in metadata.items()
                    },
                    initial_histories=inherited,
                    run_id=run_id,
                    run_root_override=step_root,
                )
                histories = result.histories or {}
                step_status = result.runs[0].status
            else:
                task = resolve_behavioral_task(step.task_id, step.task_config)
                task_run = asyncio.run(
                    run_persisted_task_batch_async(
                        personas=personas,
                        task=task,
                        settings=settings,
                        client_factory=lambda: client,
                        project_root=project_root,
                        run_id=run_id,
                        response_metadata_by_subject={
                            subject_id: {**values, "step_id": step.id}
                            for subject_id, values in metadata.items()
                        },
                        initial_histories=inherited,
                        run_root_override=step_root,
                    )
                )
                histories = _load_conversations(step_root / "conversations")
                step_status = task_run.status
            _write_conversations(run_root / "conversations", histories)
            step_results.append(
                ProtocolStepResult(
                    step_id=step.id,
                    kind=step.kind,
                    status=step_status,
                    output_path=str(step_root),
                )
            )
            status = _combined_status(status, step_status)

    completed_at = datetime.now(UTC)
    run_record = RunRecord(
        experiment_id=protocol.experiment_id,
        session_id=normalize_prefixed_uuid("session-"),
        run_id=run_id,
        subject_ids=[str(persona.subject_id) for persona in personas.personas],
        persona_count=len(personas.personas),
        procedure_kind="protocol",
        procedure_id=protocol.name,
        procedure_version=protocol.version,
        model_slug=slugify_model_name(protocol.provider.model),
        provider=ProviderSnapshot(
            provider_base_url=protocol.provider.base_url,
            model=protocol.provider.model,
            temperature=protocol.provider.temperature,
            timeout_seconds=protocol.provider.timeout_seconds,
            supports_structured_outputs=protocol.provider.supports_structured_outputs,
            supports_logprobs=protocol.provider.supports_logprobs,
        ),
        started_at=started_at,
        completed_at=completed_at,
        status=status,
        error_count=sum(
            result.status in {ResponseStatus.FAILED, ResponseStatus.INVALID}
            for result in step_results
        ),
        item_count=len(step_results),
        output_paths={
            "run": str(run_root / "run.json"),
            "steps": str(run_root / "steps"),
            "conversations": str(run_root / "conversations"),
        },
        metadata={
            "protocol_fingerprint": protocol_fingerprint(protocol),
            "cohort_id": resolved_cohort_id,
            "persona_seed": effective_persona_seed,
            "run_seed": effective_run_seed,
            "step_results": [
                result.model_dump(mode="json") for result in step_results
            ],
        },
    )
    write_json_document(run_root / "run.json", run_record)
    update_experiment_metadata(experiment_root / "metadata.json", run_record)
    return ProtocolRunResult(
        run_id=run_id,
        run_root=run_root,
        cohort_id=resolved_cohort_id,
        step_results=step_results,
    )


def load_protocol_experiment(
    project_root: Path,
    experiment_id: str,
) -> LoadedProtocolExperiment:
    """Load a canonical protocol, converting legacy design.json when needed."""
    experiment_root = _experiment_root(project_root, experiment_id)
    protocol_path = experiment_root / "protocol.json"
    if protocol_path.exists():
        payload = json.loads(protocol_path.read_text(encoding="utf-8"))
        if "experiment_id" in payload and "steps" in payload:
            return LoadedProtocolExperiment(
                protocol=UnifiedExperimentProtocol.model_validate(payload),
                source="protocol.json",
            )
    design_path = experiment_root / "design.json"
    if not design_path.exists():
        raise FileNotFoundError(f"experiment protocol not found: {experiment_root}")
    design = ExperimentDesign.model_validate_json(
        design_path.read_text(encoding="utf-8")
    )
    return LoadedProtocolExperiment(
        protocol=_protocol_from_legacy_design(design),
        source="design.json",
    )


def _protocol_from_legacy_design(design: ExperimentDesign) -> UnifiedExperimentProtocol:
    if isinstance(design.procedure, ScaleProcedureDesign):
        step: dict[str, object] = {
            "id": "questionnaire",
            "kind": "questionnaire",
            "questionnaire_id": design.procedure.questionnaire_id,
            "questionnaire_parameters": design.procedure.questionnaire_parameters,
            "scoring_model_id": design.procedure.scoring_model_id,
            "context": design.procedure.context,
        }
    elif isinstance(design.procedure, TaskProcedureDesign):
        step = {
            "id": "task",
            "kind": "task",
            "task_id": design.procedure.task_id,
            "task_config": design.procedure.task_config,
        }
    else:
        raise TypeError("unsupported legacy procedure")

    if design.protocol is not None:
        personas: dict[str, object] = {
            "count": design.protocol.base_persona_count,
            "requested_fields": design.protocol.requested_fields,
            "generation_config": design.protocol.base_persona_config,
            "factorial": design.protocol,
        }
        persona_seed = design.protocol.seed
    else:
        if design.personas is None:
            raise ValueError("legacy design has no persona configuration")
        personas = design.personas.model_dump(
            mode="json",
            exclude={"seed"},
        )
        persona_seed = design.personas.seed
    return UnifiedExperimentProtocol.model_validate(
        {
            "experiment_id": design.experiment_id,
            "name": design.experiment_id,
            "persona_seed": persona_seed,
            "run_seed": design.provider.seed,
            "personas": personas,
            "provider": {
                "model": design.provider.model,
                "base_url": design.provider.base_url,
                "temperature": design.provider.temperature,
                "timeout_seconds": design.provider.timeout_seconds,
                "supports_structured_outputs": design.provider.supports_structured_outputs,
                "supports_logprobs": design.provider.supports_logprobs,
            },
            "steps": [step],
        }
    )


def _validate_protocol_references(protocol: UnifiedExperimentProtocol) -> None:
    for step in protocol.steps:
        if isinstance(step, ProtocolQuestionnaireStep):
            questionnaire = resolve_questionnaire(
                step.questionnaire_id,
                step.questionnaire_parameters,
            )
            if step.scoring_model_id is not None and all(
                model.id != step.scoring_model_id
                for model in questionnaire.scoring_models
            ):
                raise ValueError(
                    f"unknown scoring model {step.scoring_model_id!r} "
                    f"for {questionnaire.id}"
                )
        else:
            resolve_behavioral_task(step.task_id, step.task_config)


def _create_cohort(
    experiment_root: Path,
    protocol: UnifiedExperimentProtocol,
    persona_seed: int | None,
) -> str:
    cohort_id = f"cohort-{uuid4()}"
    cohort_root = _cohort_root(experiment_root, cohort_id)
    cohort_root.mkdir(parents=True)
    assignments: list[ProtocolAssignment] = []
    if protocol.personas.factorial is not None:
        factor_protocol: ExperimentProtocol = protocol.personas.factorial.model_copy(
            update={"seed": persona_seed}
        )
        expansion = expand_protocol_personas(
            factor_protocol,
            protocol.experiment_id,
        )
        personas = expansion.personas
        assignments = expansion.assignments
    else:
        personas = PersonaFactory().create_demographics_batch(
            PersonaFactoryRequest(
                count=protocol.personas.count,
                requested_fields=protocol.personas.requested_fields,
                seed=persona_seed,
                experiment_id=protocol.experiment_id,
                generation_config=protocol.personas.generation_config,
            )
        )
    write_persona_batch_at_path(cohort_root / "personas.json", personas)
    write_json_document(
        cohort_root / "protocol-assignments.json",
        ProtocolAssignments(assignments=assignments),
    )
    metadata = CohortMetadata(
        cohort_id=cohort_id,
        experiment_id=protocol.experiment_id,
        protocol_fingerprint=protocol_fingerprint(protocol),
        persona_seed=persona_seed,
        persona_count=len(personas.personas),
        created_at=datetime.now(UTC),
    )
    write_json_document(cohort_root / "metadata.json", metadata)
    return cohort_id


def _find_cohort(experiment_root: Path, persona_seed: int | None) -> str | None:
    cohorts_root = experiment_root / "cohorts"
    if not cohorts_root.exists():
        return None
    for path in sorted(cohorts_root.glob("cohort-*")):
        metadata = _load_cohort_metadata(path / "metadata.json")
        if metadata.persona_seed == persona_seed:
            return metadata.cohort_id
    return None


def _load_cohort_metadata(path: Path) -> CohortMetadata:
    return CohortMetadata.model_validate_json(path.read_text(encoding="utf-8"))


def _load_personas(path: Path) -> PersonaBatch:
    resolved = resolve_compatible_snapshot_path(path, path.with_suffix(".jsonl"))
    return load_json_document(resolved, PersonaBatch)


def _load_assignments(path: Path) -> dict[str, ProtocolAssignment]:
    legacy_path = path.with_suffix(".jsonl")
    if not path.exists() and not legacy_path.exists():
        return {}
    legacy_assignments = (
        [
            ProtocolAssignment.model_validate_json(line)
            for line in legacy_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if legacy_path.exists()
        else None
    )
    normalized_assignments = (
        load_json_document(path, ProtocolAssignments).assignments
        if path.exists()
        else None
    )
    if (
        normalized_assignments is not None
        and legacy_assignments is not None
        and normalized_assignments != legacy_assignments
    ):
        raise ValueError(
            "conflicting canonical snapshot files: "
            f"{path.name} and {legacy_path.name}"
        )
    assignments = normalized_assignments or legacy_assignments or []
    return {assignment.subject_id: assignment for assignment in assignments}


def _response_metadata(
    protocol: UnifiedExperimentProtocol,
    cohort_id: str,
    persona_seed: int | None,
    run_seed: int | None,
    personas: PersonaBatch,
    assignments: dict[str, ProtocolAssignment],
) -> dict[str, dict[str, object]]:
    common: dict[str, object] = {
        "protocol_name": protocol.name,
        "protocol_fingerprint": protocol_fingerprint(protocol),
        "cohort_id": cohort_id,
        "persona_seed": persona_seed,
        "run_seed": run_seed,
    }
    output: dict[str, dict[str, object]] = {
        str(persona.subject_id): dict(common) for persona in personas.personas
    }
    for assignment in assignments.values():
        output[assignment.subject_id] = {
            **common,
            "base_subject_id": assignment.base_subject_id,
            "condition_id": assignment.condition_id,
            "iteration_index": assignment.iteration_index,
            "factor_values": assignment.factor_values,
            "factor_level_ids": assignment.factor_level_ids,
        }
    return output


def _load_conversations(path: Path) -> dict[str, list[Message]]:
    output: dict[str, list[Message]] = {}
    for conversation_path in path.glob("*.jsonl"):
        output[conversation_path.stem] = [
            ChatMessage.model_validate_json(line).model_dump()
            for line in conversation_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return output


def _write_conversations(
    path: Path,
    histories: dict[str, list[Message]],
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for subject_id, history in histories.items():
        (path / f"{subject_id}.jsonl").write_text(
            "".join(ChatMessage(**message).model_dump_json() + "\n" for message in history),
            encoding="utf-8",
        )


def _combined_status(
    current: ResponseStatus,
    new: ResponseStatus,
) -> ResponseStatus:
    order = {
        ResponseStatus.COMPLETED: 0,
        ResponseStatus.SKIPPED: 1,
        ResponseStatus.INVALID: 2,
        ResponseStatus.FAILED: 3,
    }
    return new if order[new] > order[current] else current


def _next_protocol_run_id(
    experiment_root: Path,
    model: str,
    started_at: datetime,
) -> str:
    candidate = started_at
    while True:
        run_id = (
            f"run-protocol-{slugify_model_name(model)}-"
            f"{candidate.strftime('%Y%m%d%H%M%S')}"
        )
        if not (experiment_root / run_id).exists():
            return run_id
        candidate += timedelta(seconds=1)


def _experiment_root(project_root: Path, experiment_id: str) -> Path:
    return project_root / "experiments" / validate_experiment_id(experiment_id)


def _cohort_root(experiment_root: Path, cohort_id: str) -> Path:
    normalized = normalize_prefixed_uuid("cohort-", cohort_id)
    path = experiment_root / "cohorts" / normalized
    if cohort_id != normalized:
        raise ValueError(f"invalid cohort_id: {cohort_id}")
    return path
