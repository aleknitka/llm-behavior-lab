from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger

from llm_behavior_lab.client import SyncLlmClient
from llm_behavior_lab.models import LlmQuestionResult, ModelSettings, Persona
from llm_behavior_lab.personas.factory import (
    PersonaBatch,
    PersonaFactory,
    PersonaFactoryRequest,
    PersonaGenerationConfig,
    RequestedDemographicField,
)
from llm_behavior_lab.preference_tests.models import (
    PairwisePreferenceExperiment,
    PairwisePreferenceRecord,
    PairwiseTrial,
)
from llm_behavior_lab.preference_tests.prompting import render_pairwise_preference_prompt
from llm_behavior_lab.prompting import render_persona_intro
from llm_behavior_lab.responses.base import (
    ChatMessage,
    ProviderSnapshot,
    ResponseStatus,
    RunRecord,
)
from llm_behavior_lab.storage import (
    append_jsonl_record,
    generate_experiment_id,
    normalize_prefixed_uuid,
    resolve_experiment_paths,
    slugify_model_name,
    update_experiment_metadata,
    validate_experiment_id,
    write_json_document,
    write_persona_batch,
)

Message = dict[str, str]


@dataclass(frozen=True)
class PreferenceBatchRunResult:
    experiment_id: str
    session_id: str
    personas: PersonaBatch
    runs: list[RunRecord]


def run_pairwise_preference_test(
    persona: Persona,
    experiment: PairwisePreferenceExperiment,
    settings: ModelSettings,
    client: SyncLlmClient,
    project_root: Path,
    experiment_id: str | None = None,
    session_id: str | None = None,
    run_id: str | None = None,
    response_metadata: dict[str, object] | None = None,
) -> list[PairwisePreferenceRecord]:
    resolved_experiment_id = (
        validate_experiment_id(experiment_id) if experiment_id else generate_experiment_id()
    )
    resolved_session_id = normalize_prefixed_uuid("session-", session_id)
    started_at = datetime.now(UTC)
    resolved_run_id = run_id or _next_preference_run_id(
        project_root=project_root,
        experiment_id=resolved_experiment_id,
        preference_experiment=experiment,
        settings=settings,
        started_at=started_at,
    )
    paths = resolve_experiment_paths(
        project_root=project_root,
        experiment_id=resolved_experiment_id,
        run_id=resolved_run_id,
    )
    logger.info(
        "Starting pairwise preference run {run_id} persona={persona_id} "
        "preference_experiment={preference_experiment_id} model={model}",
        run_id=resolved_run_id,
        persona_id=persona.persona_id,
        preference_experiment_id=experiment.id,
        model=settings.model,
    )
    experiment_path = paths.run_root / "experiment.json"
    _write_experiment_copy(experiment_path, experiment)
    records = _run_pairwise_preference_for_persona(
        persona=persona,
        experiment=experiment,
        settings=settings,
        client=client,
        experiment_id=resolved_experiment_id,
        session_id=resolved_session_id,
        run_id=resolved_run_id,
        response_path=paths.response_path_for_subject(persona.persona_id),
        response_metadata=response_metadata,
    )

    completed_at = datetime.now(UTC)
    run_record = _run_record(
        experiment_id=resolved_experiment_id,
        session_id=resolved_session_id,
        run_id=resolved_run_id,
        subject_ids=[persona.persona_id],
        persona_count=1,
        preference_experiment=experiment,
        settings=settings,
        started_at=started_at,
        completed_at=completed_at,
        records=records,
        output_paths={
            "run": str(paths.run_path),
            "responses": str(paths.responses_root),
            "experiment": str(experiment_path),
        },
    )
    write_json_document(paths.run_path, run_record)
    update_experiment_metadata(paths.metadata_path, run_record)
    return records


def run_pairwise_preference_batch(
    experiment: PairwisePreferenceExperiment,
    settings: ModelSettings,
    client: SyncLlmClient,
    project_root: Path,
    experiment_id: str | None = None,
    persona_count: int = 100,
    seed: int | None = None,
    persona_config: PersonaGenerationConfig | None = None,
) -> PreferenceBatchRunResult:
    resolved_experiment_id = (
        validate_experiment_id(experiment_id) if experiment_id else generate_experiment_id(seed)
    )
    session_id = normalize_prefixed_uuid("session-")
    personas = PersonaFactory().create_demographics_batch(
        PersonaFactoryRequest(
            count=persona_count,
            requested_fields=set(RequestedDemographicField),
            seed=seed,
            experiment_id=resolved_experiment_id,
            generation_config=persona_config or PersonaGenerationConfig(),
        )
    )
    write_persona_batch(project_root, personas)
    run_settings = settings.model_copy(
        update={"seed": settings.seed if settings.seed is not None else seed}
    )
    started_at = datetime.now(UTC)
    run_id = _next_preference_run_id(
        project_root=project_root,
        experiment_id=resolved_experiment_id,
        preference_experiment=experiment,
        settings=run_settings,
        started_at=started_at,
    )
    paths = resolve_experiment_paths(
        project_root=project_root,
        experiment_id=resolved_experiment_id,
        run_id=run_id,
    )
    experiment_path = paths.run_root / "experiment.json"
    _write_experiment_copy(experiment_path, experiment)

    all_records: list[PairwisePreferenceRecord] = []
    for generated_persona in personas.personas:
        runtime_persona = _runtime_persona_from_generated(generated_persona)
        records = _run_pairwise_preference_for_persona(
            persona=runtime_persona,
            experiment=experiment,
            settings=run_settings,
            client=client,
            experiment_id=resolved_experiment_id,
            session_id=session_id,
            run_id=run_id,
            response_path=paths.response_path_for_subject(runtime_persona.persona_id),
            response_metadata=None,
        )
        all_records.extend(records)

    completed_at = datetime.now(UTC)
    run_record = _run_record(
        experiment_id=resolved_experiment_id,
        session_id=session_id,
        run_id=run_id,
        subject_ids=[str(persona.subject_id) for persona in personas.personas],
        persona_count=len(personas.personas),
        preference_experiment=experiment,
        settings=run_settings,
        started_at=started_at,
        completed_at=completed_at,
        records=all_records,
        output_paths={
            "run": str(paths.run_path),
            "responses": str(paths.responses_root),
            "experiment": str(experiment_path),
        },
    )
    write_json_document(paths.run_path, run_record)
    update_experiment_metadata(paths.metadata_path, run_record)
    return PreferenceBatchRunResult(
        experiment_id=resolved_experiment_id,
        session_id=session_id,
        personas=personas,
        runs=[run_record],
    )


def _run_pairwise_preference_for_persona(
    persona: Persona,
    experiment: PairwisePreferenceExperiment,
    settings: ModelSettings,
    client: SyncLlmClient,
    experiment_id: str,
    session_id: str,
    run_id: str,
    response_path: Path,
    response_metadata: dict[str, object] | None,
) -> list[PairwisePreferenceRecord]:
    system_message = {"role": "system", "content": render_persona_intro(persona)}
    records: list[PairwisePreferenceRecord] = []

    for trial in experiment.trials:
        displayed_stimulus_ids = _displayed_stimulus_ids(settings.seed, persona.persona_id, trial)
        prompt = render_pairwise_preference_prompt(
            experiment=experiment,
            trial=trial,
            displayed_stimulus_ids=displayed_stimulus_ids,
        )
        messages = [system_message, {"role": "user", "content": prompt}]
        call_seed = _trial_call_seed(settings.seed, persona.persona_id, trial.id)
        call_settings = settings.model_copy(update={"seed": call_seed})
        try:
            result = client.complete(messages, call_settings, ["A", "B"])
        except Exception as exc:
            logger.exception(
                "Provider call failed preference_trial={trial_id} run_id={run_id}",
                trial_id=trial.id,
                run_id=run_id,
            )
            result = LlmQuestionResult(error=str(exc))

        record = _record_from_result(
            persona=persona,
            experiment=experiment,
            trial=trial,
            displayed_stimulus_ids=displayed_stimulus_ids,
            experiment_id=experiment_id,
            session_id=session_id,
            run_id=run_id,
            messages=messages,
            result=result,
            seed=call_seed,
            metadata_extra=response_metadata,
        )
        append_jsonl_record(response_path, record)
        records.append(record)
    return records


def _record_from_result(
    persona: Persona,
    experiment: PairwisePreferenceExperiment,
    trial: PairwiseTrial,
    displayed_stimulus_ids: tuple[str, str],
    experiment_id: str,
    session_id: str,
    run_id: str,
    messages: list[Message],
    result: LlmQuestionResult,
    seed: int | None,
    metadata_extra: dict[str, object] | None,
) -> PairwisePreferenceRecord:
    selected_label = result.selected_answer_id.strip() if result.selected_answer_id else None
    selected_stimulus_id: str | None = None
    rejected_stimulus_id: str | None = None
    status = ResponseStatus.FAILED if result.error else ResponseStatus.COMPLETED
    if result.error is None:
        if selected_label == "A":
            selected_stimulus_id = displayed_stimulus_ids[0]
            rejected_stimulus_id = displayed_stimulus_ids[1]
        elif selected_label == "B":
            selected_stimulus_id = displayed_stimulus_ids[1]
            rejected_stimulus_id = displayed_stimulus_ids[0]
        else:
            selected_label = None
            status = ResponseStatus.INVALID

    metadata: dict[str, Any] = {"experiment_id": experiment_id}
    if metadata_extra:
        metadata.update(metadata_extra)
    if seed is not None:
        metadata["seed"] = seed

    return PairwisePreferenceRecord(
        subject_id=persona.persona_id,
        session_id=session_id,
        run_id=run_id,
        preference_experiment_id=experiment.id,
        preference_experiment_version=experiment.version,
        trial_id=trial.id,
        trial_order=trial.order,
        stimulus_ids=trial.stimulus_ids,
        displayed_stimulus_ids=displayed_stimulus_ids,
        selected_label=selected_label,
        selected_stimulus_id=selected_stimulus_id,
        rejected_stimulus_id=rejected_stimulus_id,
        messages=[ChatMessage(**message) for message in messages if message["role"] != "system"],
        raw_response=result.raw_response,
        structured_response=result.structured_response,
        logprobs=result.logprobs,
        status=status,
        error=result.error,
        metadata=metadata,
    )


def _next_preference_run_id(
    project_root: Path,
    experiment_id: str,
    preference_experiment: PairwisePreferenceExperiment,
    settings: ModelSettings,
    started_at: datetime,
) -> str:
    candidate_time = started_at
    while True:
        timestamp = candidate_time.strftime("%Y%m%d%H%M%S")
        run_id = (
            f"run-pref-{slugify_model_name(preference_experiment.id)}-"
            f"{slugify_model_name(settings.model)}-{timestamp}"
        )
        paths = resolve_experiment_paths(project_root, experiment_id, run_id)
        if not paths.run_root.exists():
            return run_id
        candidate_time += timedelta(seconds=1)


def _write_experiment_copy(path: Path, experiment: PairwisePreferenceExperiment) -> None:
    write_json_document(path, experiment)


def _runtime_persona_from_generated(persona: Any) -> Persona:
    dumped = persona.features.model_dump(mode="json", exclude_none=True)
    return Persona(
        persona_id=str(persona.subject_id),
        features={key: str(value) for key, value in dumped.items()},
    )


def _displayed_stimulus_ids(
    base_seed: int | None,
    subject_id: str,
    trial: PairwiseTrial,
) -> tuple[str, str]:
    digest = hashlib.sha256(
        f"display:{base_seed}:{subject_id}:{trial.id}".encode()
    ).hexdigest()
    if int(digest[:2], 16) % 2 == 0:
        return trial.stimulus_ids
    return (trial.stimulus_ids[1], trial.stimulus_ids[0])


def _trial_call_seed(base_seed: int | None, subject_id: str, trial_id: str) -> int | None:
    if base_seed is None:
        return None
    digest = hashlib.sha256(f"{base_seed}:{subject_id}:{trial_id}".encode()).hexdigest()
    return int(digest[:8], 16)


def _provider_snapshot(settings: ModelSettings) -> ProviderSnapshot:
    return ProviderSnapshot(
        provider_base_url=settings.provider_base_url,
        model=settings.model,
        temperature=settings.temperature,
        timeout_seconds=settings.timeout_seconds,
        supports_structured_outputs=settings.capabilities.supports_structured_outputs,
        supports_logprobs=settings.capabilities.supports_logprobs,
    )


def _response_status(records: list[PairwisePreferenceRecord]) -> ResponseStatus:
    if any(record.status == ResponseStatus.FAILED for record in records):
        return ResponseStatus.FAILED
    if any(record.status == ResponseStatus.INVALID for record in records):
        return ResponseStatus.INVALID
    return ResponseStatus.COMPLETED


def _run_record(
    experiment_id: str,
    session_id: str,
    run_id: str,
    subject_ids: list[str],
    persona_count: int,
    preference_experiment: PairwisePreferenceExperiment,
    settings: ModelSettings,
    started_at: datetime,
    completed_at: datetime,
    records: list[PairwisePreferenceRecord],
    output_paths: dict[str, str],
) -> RunRecord:
    return RunRecord(
        experiment_id=experiment_id,
        session_id=session_id,
        run_id=run_id,
        subject_ids=subject_ids,
        persona_count=persona_count,
        procedure_kind="pairwise",
        procedure_id=preference_experiment.id,
        procedure_version=preference_experiment.version,
        questionnaire_id=preference_experiment.id,
        questionnaire_shorthand="pref",
        questionnaire_version=preference_experiment.version,
        model_slug=slugify_model_name(settings.model),
        provider=_provider_snapshot(settings),
        started_at=started_at,
        completed_at=completed_at,
        status=_response_status(records),
        error_count=sum(1 for record in records if record.status == ResponseStatus.FAILED),
        item_count=len(records),
        output_paths=output_paths,
        metadata={
            "experiment_kind": "pairwise_preference",
            "preference_experiment_id": preference_experiment.id,
            "base_seed": settings.seed,
        },
    )
