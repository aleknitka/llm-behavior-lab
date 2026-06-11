from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from llm_behavior_lab.behavioral_tasks.iowa_gambling import IowaGamblingTask
from llm_behavior_lab.behavioral_tasks.runner import run_behavioral_task
from llm_behavior_lab.client import Message, SyncLlmClient
from llm_behavior_lab.models import ModelSettings, Persona
from llm_behavior_lab.personas.factory import GeneratedPersona, PersonaBatch
from llm_behavior_lab.responses.base import ProviderSnapshot, ResponseStatus, RunRecord
from llm_behavior_lab.storage import (
    load_json_document,
    normalize_prefixed_uuid,
    resolve_experiment_paths,
    slugify_model_name,
    update_experiment_metadata,
    write_json_document,
)


async def run_persisted_task_batch_async(
    *,
    personas: PersonaBatch,
    task: IowaGamblingTask,
    settings: ModelSettings,
    client_factory: Callable[[], SyncLlmClient],
    project_root: Path,
    run_id: str | None = None,
    concurrency: int = 4,
    resume: bool = True,
    retry_failed: bool = False,
    response_metadata_by_subject: dict[str, dict[str, object]] | None = None,
    initial_histories: dict[str, list[Message]] | None = None,
    run_root_override: Path | None = None,
) -> RunRecord:
    """Run subjects concurrently while preserving sequential trials per subject."""
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    experiment_id = personas.experiment_id
    started_at = datetime.now(UTC)
    resolved_run_id = run_id or _next_task_run_id(
        project_root, experiment_id, task, settings, started_at
    )
    paths = resolve_experiment_paths(
        project_root,
        experiment_id,
        resolved_run_id,
        run_root_override=run_root_override,
    )
    session_id = _existing_session_id(paths.run_path) or normalize_prefixed_uuid(
        "session-"
    )
    paths.run_root.mkdir(parents=True, exist_ok=True)
    task_path = paths.run_root / "task.json"
    if not task_path.exists():
        write_json_document(task_path, task.config)
    elif task.config != type(task.config).model_validate_json(
        task_path.read_text(encoding="utf-8")
    ):
        raise ValueError("persisted task configuration does not match experiment")

    semaphore = asyncio.Semaphore(concurrency)
    metadata = response_metadata_by_subject or {}

    async def run_subject(generated: GeneratedPersona):
        subject_id = str(generated.subject_id)
        response_path = paths.response_path_for_subject(subject_id)
        if response_path.exists() and not resume:
            raise FileExistsError(f"task response ledger already exists: {response_path}")
        async with semaphore:
            return await asyncio.to_thread(
                run_behavioral_task,
                persona=_runtime_persona(generated),
                task=task,
                settings=settings,
                client=client_factory(),
                project_root=project_root,
                experiment_id=experiment_id,
                session_id=session_id,
                run_id=resolved_run_id,
                resolved_schedule=task.resolve_schedule(
                    seed=settings.seed,
                    subject_id=subject_id,
                ),
                resume=response_path.exists(),
                retry_failed=retry_failed,
                response_metadata=metadata.get(subject_id),
                initial_history=(initial_histories or {}).get(subject_id),
                run_root_override=run_root_override,
            )

    results = await asyncio.gather(
        *(run_subject(persona) for persona in personas.personas)
    )
    completed_at = datetime.now(UTC)
    statuses = [result.status for result in results]
    status = ResponseStatus.COMPLETED
    if ResponseStatus.FAILED in statuses:
        status = ResponseStatus.FAILED
    elif ResponseStatus.INVALID in statuses:
        status = ResponseStatus.INVALID
    elif ResponseStatus.SKIPPED in statuses:
        status = ResponseStatus.SKIPPED
    records = [record for result in results for record in result.records]
    run_record = RunRecord(
        experiment_id=experiment_id,
        session_id=session_id,
        run_id=resolved_run_id,
        subject_ids=[str(persona.subject_id) for persona in personas.personas],
        persona_count=len(personas.personas),
        procedure_kind="task",
        procedure_id=task.id,
        procedure_version=task.version,
        model_slug=slugify_model_name(settings.model),
        provider=ProviderSnapshot(
            provider_base_url=settings.provider_base_url,
            model=settings.model,
            temperature=settings.temperature,
            timeout_seconds=settings.timeout_seconds,
            max_attempts=settings.max_attempts,
            initial_backoff_seconds=settings.initial_backoff_seconds,
            max_backoff_seconds=settings.max_backoff_seconds,
            max_concurrency=settings.max_concurrency,
            supports_structured_outputs=settings.capabilities.supports_structured_outputs,
            supports_logprobs=settings.capabilities.supports_logprobs,
        ),
        started_at=started_at,
        completed_at=completed_at,
        status=status,
        error_count=sum(
            record.status in {ResponseStatus.FAILED, ResponseStatus.INVALID}
            for record in records
        ),
        item_count=sum(
            record.status == ResponseStatus.COMPLETED for record in records
        ),
        output_paths={
            "run": str(paths.run_path),
            "responses": str(paths.responses_root),
            "task": str(task_path),
            "schedules": str(paths.run_root / "schedules"),
            "conversations": str(paths.run_root / "conversations"),
        },
        metadata={
            "base_seed": settings.seed,
            "concurrency": concurrency,
        },
    )
    write_json_document(paths.run_path, run_record)
    update_experiment_metadata(paths.metadata_path, run_record)
    return run_record


def _runtime_persona(persona: GeneratedPersona) -> Persona:
    dumped = persona.features.model_dump(mode="json", exclude_none=True)
    return Persona(
        persona_id=str(persona.subject_id),
        features={key: str(value) for key, value in dumped.items()},
    )


def _existing_session_id(run_path: Path) -> str | None:
    if not run_path.exists():
        return None
    return load_json_document(run_path, RunRecord).session_id


def _next_task_run_id(
    project_root: Path,
    experiment_id: str,
    task: IowaGamblingTask,
    settings: ModelSettings,
    started_at: datetime,
) -> str:
    candidate = started_at
    while True:
        timestamp = candidate.strftime("%Y%m%d%H%M%S")
        run_id = (
            f"run-task-{slugify_model_name(task.id)}-"
            f"{slugify_model_name(settings.model)}-{timestamp}"
        )
        if not resolve_experiment_paths(
            project_root, experiment_id, run_id
        ).run_root.exists():
            return run_id
        candidate += timedelta(seconds=1)
