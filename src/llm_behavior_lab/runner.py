import asyncio
import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from loguru import logger

from llm_behavior_lab.client import AsyncLlmClient, SyncLlmClient
from llm_behavior_lab.models import (
    LlmQuestionResult,
    ModelSettings,
    Persona,
)
from llm_behavior_lab.personas.factory import (
    GeneratedPersona,
    PersonaBatch,
    PersonaFactory,
    PersonaFactoryRequest,
    PersonaGenerationConfig,
    RequestedDemographicField,
)
from llm_behavior_lab.prompting import render_persona_intro
from llm_behavior_lab.protocols import (
    ExperimentProtocol,
    ProtocolAssignment,
    ProtocolAssignments,
    expand_protocol_personas,
)
from llm_behavior_lab.questionnaires.base.response_formats import (
    LikertFormat,
    MultipleChoiceFormat,
    NumericFormat,
    ResponseFormat,
    SingleChoiceFormat,
    TextFormat,
)
from llm_behavior_lab.questionnaires.base.scale import Item, Questionnaire
from llm_behavior_lab.responses.base import (
    AnswerValue,
    ChatMessage,
    ItemResponseRecord,
    LikertAnswerValue,
    MultipleChoiceAnswerValue,
    NumericAnswerValue,
    ProviderSnapshot,
    ResponseStatus,
    RunRecord,
    SingleChoiceAnswerValue,
    TextAnswerValue,
)
from llm_behavior_lab.responses.item_ledgers import (
    latest_item_attempts,
    load_item_ledger,
    pending_item_ids,
    validate_item_ledger,
)
from llm_behavior_lab.storage import (
    ExperimentPaths,
    append_jsonl_record,
    build_run_directory_name,
    generate_experiment_id,
    load_json_document,
    normalize_prefixed_uuid,
    resolve_experiment_paths,
    slugify_model_name,
    update_experiment_metadata,
    validate_experiment_id,
    write_json_document,
    write_persona_batch,
    write_persona_batch_at_path,
)

Message = dict[str, str]


@dataclass(frozen=True)
class BatchRunResult:
    experiment_id: str
    session_id: str
    personas: PersonaBatch
    runs: list[RunRecord]
    histories: dict[str, list[Message]] | None = None


def _allowed_answer_ids(response_format: ResponseFormat) -> list[str]:
    if isinstance(response_format, LikertFormat):
        return [
            str(value) for value in range(response_format.min_value, response_format.max_value + 1)
        ]
    if isinstance(response_format, SingleChoiceFormat | MultipleChoiceFormat):
        return [option.id for option in response_format.options]
    return []


def _response_options(response_format: ResponseFormat) -> str:
    if isinstance(response_format, LikertFormat):
        return "\n".join(
            f"{value}. {response_format.labels.get(value, str(value))}"
            for value in range(response_format.min_value, response_format.max_value + 1)
        )
    if isinstance(response_format, SingleChoiceFormat | MultipleChoiceFormat):
        return "\n".join(f"{option.id}. {option.label}" for option in response_format.options)
    if isinstance(response_format, NumericFormat):
        bounds = []
        if response_format.min_value is not None:
            bounds.append(f"minimum {response_format.min_value}")
        if response_format.max_value is not None:
            bounds.append(f"maximum {response_format.max_value}")
        unit = f" in {response_format.unit}" if response_format.unit else ""
        bounds_text = f" ({', '.join(bounds)})" if bounds else ""
        return f"Reply with a numeric value{unit}{bounds_text}."
    if isinstance(response_format, TextFormat):
        max_length = (
            f" no longer than {response_format.max_length} characters"
            if response_format.max_length is not None
            else ""
        )
        return f"Reply with text{max_length}."
    return ""


def _question_prompt(item: Item) -> str:
    answers = _response_options(item.response_format)
    return (
        f"{item.text}\n\nAllowed answers:\n{answers}\n\nReply with exactly one allowed answer id."
    )


def _answer_from_result(item: Item, result: LlmQuestionResult) -> AnswerValue | None:
    selected = result.selected_answer_id
    if selected is None:
        return None

    response_format = item.response_format
    if isinstance(response_format, LikertFormat):
        try:
            value = int(selected)
        except ValueError:
            return None
        if value < response_format.min_value or value > response_format.max_value:
            return None
        return LikertAnswerValue(value=value, label=response_format.labels.get(value))

    if isinstance(response_format, SingleChoiceFormat):
        option = next((option for option in response_format.options if option.id == selected), None)
        if option is None:
            return None
        return SingleChoiceAnswerValue(
            option_id=option.id,
            label=option.label,
            value=option.value,
        )

    if isinstance(response_format, MultipleChoiceFormat):
        selected_ids = [answer_id.strip() for answer_id in selected.split(",") if answer_id.strip()]
        options = [option for option in response_format.options if option.id in selected_ids]
        if len(options) != len(selected_ids):
            return None
        return MultipleChoiceAnswerValue(
            option_ids=[option.id for option in options],
            labels=[option.label for option in options],
            values=[option.value for option in options],
        )

    if isinstance(response_format, NumericFormat):
        try:
            value = float(selected)
        except ValueError:
            return None
        if response_format.min_value is not None and value < response_format.min_value:
            return None
        if response_format.max_value is not None and value > response_format.max_value:
            return None
        return NumericAnswerValue(value=value, unit=response_format.unit)

    if isinstance(response_format, TextFormat):
        if response_format.max_length is not None and len(selected) > response_format.max_length:
            return None
        return TextAnswerValue(text=selected)

    return None


def _record_from_result(
    experiment_id: str,
    subject_id: str,
    session_id: str,
    run_id: str,
    questionnaire: Questionnaire,
    item: Item,
    messages: Sequence[Message],
    result: LlmQuestionResult,
    seed: int | None = None,
    metadata_extra: dict[str, object] | None = None,
) -> ItemResponseRecord:
    answer = _answer_from_result(item, result)
    status = ResponseStatus.FAILED if result.error else ResponseStatus.COMPLETED
    if result.error is None and answer is None:
        status = ResponseStatus.INVALID
    metadata: dict[str, object] = {"experiment_id": experiment_id}
    if metadata_extra:
        metadata.update(metadata_extra)
    if seed is not None:
        metadata["seed"] = seed

    return ItemResponseRecord(
        subject_id=subject_id,
        session_id=session_id,
        run_id=run_id,
        questionnaire_id=questionnaire.id,
        questionnaire_version=questionnaire.version,
        item_id=item.id,
        item_order=item.order,
        item_text=item.text,
        response_format_type=str(item.response_format.type),
        messages=[ChatMessage(**message) for message in messages if message["role"] != "system"],
        answer=answer,
        raw_response=result.raw_response,
        structured_response=result.structured_response,
        logprobs=result.logprobs,
        status=status,
        error=result.error,
        metadata=metadata,
    )


def _provider_snapshot(settings: ModelSettings) -> ProviderSnapshot:
    return ProviderSnapshot(
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
    )


def _response_status(
    records: Sequence[ItemResponseRecord],
    *,
    expected_item_count: int,
    cancelled: bool = False,
) -> ResponseStatus:
    if cancelled:
        return ResponseStatus.CANCELLED
    if len(records) < expected_item_count:
        return ResponseStatus.PARTIAL
    if any(record.status == ResponseStatus.FAILED for record in records):
        return ResponseStatus.FAILED
    if any(record.status == ResponseStatus.INVALID for record in records):
        return ResponseStatus.INVALID
    return ResponseStatus.COMPLETED


def _item_call_seed(base_seed: int | None, subject_id: str, item_id: str) -> int | None:
    if base_seed is None:
        return None
    digest = hashlib.sha256(f"{base_seed}:{subject_id}:{item_id}".encode()).hexdigest()
    return int(digest[:8], 16)


def _run_record(
    experiment_id: str,
    session_id: str,
    run_id: str,
    subject_ids: Sequence[str],
    persona_count: int,
    questionnaire: Questionnaire,
    settings: ModelSettings,
    started_at: datetime,
    completed_at: datetime | None,
    records: Sequence[ItemResponseRecord],
    output_paths: dict[str, str],
    *,
    cancelled: bool = False,
) -> RunRecord:
    effective_records = latest_item_attempts(records)
    expected_item_count = len(questionnaire.items) * persona_count
    return RunRecord(
        experiment_id=experiment_id,
        session_id=session_id,
        run_id=run_id,
        subject_ids=list(subject_ids),
        persona_count=persona_count,
        procedure_kind="scale",
        procedure_id=questionnaire.id,
        procedure_version=questionnaire.version,
        questionnaire_id=questionnaire.id,
        questionnaire_shorthand=questionnaire.shorthand,
        questionnaire_version=questionnaire.version,
        model_slug=slugify_model_name(settings.model),
        provider=_provider_snapshot(settings),
        started_at=started_at,
        completed_at=completed_at,
        status=_response_status(
            effective_records,
            expected_item_count=expected_item_count,
            cancelled=cancelled,
        ),
        error_count=sum(
            1 for record in effective_records if record.status == ResponseStatus.FAILED
        ),
        item_count=len(effective_records),
        output_paths=output_paths,
        metadata={"scoring_model_id": None, "base_seed": settings.seed},
    )


def _write_scale_copy(path: Path, questionnaire: Questionnaire) -> None:
    write_json_document(path, questionnaire)


def validate_questionnaire_resume(
    paths: ExperimentPaths,
    questionnaire: Questionnaire,
    settings: ModelSettings,
    subject_ids: Sequence[str],
    run_id: str,
) -> RunRecord | None:
    if not paths.run_root.exists():
        return None
    if not paths.scale_path.exists():
        raise ValueError("existing run is missing its questionnaire snapshot")
    persisted_questionnaire = load_json_document(paths.scale_path, Questionnaire)
    if persisted_questionnaire != questionnaire:
        raise ValueError("run questionnaire configuration does not match resume request")

    existing = load_json_document(paths.run_path, RunRecord) if paths.run_path.exists() else None
    existing_response_paths = list(paths.responses_root.glob("*.jsonl"))
    if existing is None and existing_response_paths:
        raise ValueError(
            "existing response ledgers are missing their run manifest; "
            "provider and session identity cannot be verified"
        )
    if existing is not None:
        if existing.run_id != run_id:
            raise ValueError("run manifest ID does not match resume request")
        if existing.provider != _provider_snapshot(settings):
            raise ValueError("run provider configuration does not match resume request")
        if existing.subject_ids != list(subject_ids):
            raise ValueError("run persona cohort does not match resume request")

    expected_subjects = set(subject_ids)
    existing_subjects = {response_path.stem for response_path in existing_response_paths}
    unexpected = existing_subjects - expected_subjects
    if unexpected:
        raise ValueError(
            "run persona cohort contains unexpected response ledgers: "
            + ", ".join(sorted(unexpected))
        )
    for subject_id in existing_subjects:
        records = load_item_ledger(paths.response_path_for_subject(subject_id))
        validate_item_ledger(questionnaire, subject_id, run_id, records)
    return existing


def _initialize_questionnaire_run(
    *,
    paths: ExperimentPaths,
    experiment_id: str,
    session_id: str,
    run_id: str,
    subject_ids: Sequence[str],
    questionnaire: Questionnaire,
    settings: ModelSettings,
    started_at: datetime,
    update_metadata: bool,
) -> RunRecord:
    _write_scale_copy(paths.scale_path, questionnaire)
    run_record = _run_record(
        experiment_id,
        session_id,
        run_id,
        subject_ids,
        len(subject_ids),
        questionnaire,
        settings,
        started_at,
        None,
        [],
        output_paths={
            "run": str(paths.run_path),
            "responses": str(paths.responses_root),
            "scale": str(paths.scale_path),
        },
    )
    write_json_document(paths.run_path, run_record)
    if update_metadata:
        update_experiment_metadata(paths.metadata_path, run_record)
    return run_record


def _next_run_id(
    project_root: Path,
    experiment_id: str,
    questionnaire: Questionnaire,
    settings: ModelSettings,
    started_at: datetime,
) -> str:
    candidate_time = started_at
    while True:
        run_id = build_run_directory_name(
            questionnaire_shorthand=questionnaire.shorthand,
            model=settings.model,
            started_at=candidate_time,
        )
        paths = resolve_experiment_paths(
            project_root=project_root,
            experiment_id=experiment_id,
            run_id=run_id,
        )
        if not paths.run_root.exists():
            return run_id
        candidate_time += timedelta(seconds=1)


def run_questionnaire(
    persona: Persona,
    questionnaire: Questionnaire,
    settings: ModelSettings,
    client: SyncLlmClient,
    project_root: Path,
    experiment_id: str | None = None,
    session_id: str | None = None,
    run_id: str | None = None,
    write_session_record: bool = True,
    response_metadata: dict[str, object] | None = None,
    context: str | None = None,
    initial_history: Sequence[Message] | None = None,
    run_root_override: Path | None = None,
    retry_failed: bool = False,
) -> list[ItemResponseRecord]:
    """Run one questionnaire synchronously for a single persona.

    Args:
        persona: Persona that the model should assume for the whole session.
        questionnaire: Validated questionnaire definition whose items are asked in
            order.
        settings: Provider and model settings used for each item-level call.
        client: Synchronous OpenAI-compatible client wrapper.
        project_root: Repository or experiment root under which JSONL output is
            written.
        experiment_id: Optional stable experiment identifier. When omitted, a
            generated three-part identifier is used.
        session_id: Optional ``session-[uuid]`` value. When omitted, a new session
            identifier is generated.
        run_id: Optional run directory name. When omitted, a timestamped
            ``run-{questionnaire}-{model}-{timestamp}`` identifier is generated.
        write_session_record: Whether to append run-level metadata records.
        response_metadata: Optional metadata copied into each item response record.
        context: Optional supplemental text inserted into the initial persona
            prompt, such as a vignette or paragraph the model should read before
            answering questionnaire items.

    Returns:
        Item-level response records in questionnaire order. Each record is also
        appended to the subject JSONL file under the resolved run directory.
    """
    resolved_experiment_id = (
        validate_experiment_id(experiment_id) if experiment_id else generate_experiment_id()
    )
    resolved_session_id = normalize_prefixed_uuid("session-", session_id)
    started_at = datetime.now(UTC)
    resolved_run_id = run_id or _next_run_id(
        project_root=project_root,
        experiment_id=resolved_experiment_id,
        questionnaire=questionnaire,
        settings=settings,
        started_at=started_at,
    )
    paths = resolve_experiment_paths(
        project_root=project_root,
        experiment_id=resolved_experiment_id,
        run_id=resolved_run_id,
        run_root_override=run_root_override,
    )
    logger.info(
        "Starting questionnaire run {run_id} for persona {persona_id} "
        "questionnaire={questionnaire_id} model={model}",
        run_id=resolved_run_id,
        persona_id=persona.persona_id,
        questionnaire_id=questionnaire.id,
        model=settings.model,
    )
    response_path = paths.response_path_for_subject(persona.persona_id)
    persisted = load_item_ledger(response_path)
    if persisted:
        validate_item_ledger(
            questionnaire,
            persona.persona_id,
            resolved_run_id,
            persisted,
        )
    pending = set(
        pending_item_ids(
            questionnaire,
            persisted,
            retry_failed=retry_failed,
        )
    )
    effective = latest_item_attempts(persisted)

    for item in questionnaire.items:
        if item.id not in pending:
            continue
        preceding = (
            [record for record in effective if record.item_order < item.order]
            if bool(questionnaire.metadata.get("retain_history", True))
            else []
        )
        history = _history_after_questionnaire(
            persona,
            context,
            initial_history,
            preceding,
        )
        logger.debug(
            "Asking item {item_order}/{item_count} item_id={item_id} run_id={run_id}",
            item_order=item.order,
            item_count=len(questionnaire.items),
            item_id=item.id,
            run_id=resolved_run_id,
        )
        messages = [*history, {"role": "user", "content": _question_prompt(item)}]
        allowed_answer_ids = _allowed_answer_ids(item.response_format)
        call_seed = _item_call_seed(settings.seed, persona.persona_id, item.id)
        call_settings = settings.model_copy(update={"seed": call_seed})

        try:
            result = client.complete(messages, call_settings, allowed_answer_ids)
        except Exception as exc:
            logger.warning(
                "Provider call failed item_id={item_id} run_id={run_id} "
                "error_type={error_type} error={error}",
                item_id=item.id,
                run_id=resolved_run_id,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            result = LlmQuestionResult(error=str(exc))

        record = _record_from_result(
            resolved_experiment_id,
            persona.persona_id,
            resolved_session_id,
            resolved_run_id,
            questionnaire,
            item,
            messages,
            result,
            call_seed,
            response_metadata,
        )
        append_jsonl_record(response_path, record)
        effective = latest_item_attempts([*effective, record])
        logger.debug(
            "Recorded item response item_id={item_id} run_id={run_id} status={status}",
            item_id=item.id,
            run_id=resolved_run_id,
            status=record.status,
        )

    records = latest_item_attempts(load_item_ledger(response_path))

    completed_at = datetime.now(UTC)
    run_record = _run_record(
        resolved_experiment_id,
        resolved_session_id,
        resolved_run_id,
        [persona.persona_id],
        1,
        questionnaire,
        settings,
        started_at,
        completed_at,
        records,
        output_paths={
            "run": str(paths.run_path),
            "responses": str(paths.responses_root),
            "scale": str(paths.scale_path),
        },
    )
    if write_session_record:
        write_json_document(paths.run_path, run_record)
        update_experiment_metadata(paths.metadata_path, run_record)
        _write_scale_copy(paths.scale_path, questionnaire)
    logger.info(
        "Completed questionnaire run {run_id} status={status} errors={error_count} "
        "responses_path={responses_path}",
        run_id=resolved_run_id,
        status=run_record.status,
        error_count=run_record.error_count,
        responses_path=paths.response_path_for_subject(persona.persona_id),
    )
    return records


def _runtime_persona_from_generated(persona: GeneratedPersona) -> Persona:
    dumped = persona.features.model_dump(mode="json", exclude_none=True)
    return Persona(
        persona_id=str(persona.subject_id),
        features={key: str(value) for key, value in dumped.items()},
    )


def run_persona_questionnaire_batch(
    questionnaire: Questionnaire,
    settings: ModelSettings,
    client: SyncLlmClient,
    project_root: Path,
    experiment_id: str | None = None,
    persona_count: int = 100,
    seed: int | None = None,
    persona_config: PersonaGenerationConfig | None = None,
    context: str | None = None,
) -> BatchRunResult:
    """Generate personas and run the same questionnaire for each one.

    Args:
        questionnaire: Validated questionnaire definition to administer.
        settings: Provider and model settings used for item-level calls.
        client: Synchronous OpenAI-compatible client wrapper.
        project_root: Repository or experiment root under which experiment files
            are written.
        experiment_id: Optional stable experiment identifier. When omitted, a
            generated three-part identifier is used.
        persona_count: Number of generated personas to create and run.
        seed: Optional seed used for reproducible persona generation and run ID
            generation.
        persona_config: Optional demographic generation configuration.
        context: Optional supplemental text inserted into every persona's initial
            prompt before questionnaire items are asked.

    Returns:
        Batch run metadata containing the experiment ID, shared session ID,
        generated personas, and the single aggregate run record.
    """
    resolved_experiment_id = (
        validate_experiment_id(experiment_id) if experiment_id else generate_experiment_id(seed)
    )
    session_id = normalize_prefixed_uuid("session-")
    logger.info(
        "Starting persona questionnaire batch experiment_id={experiment_id} "
        "session_id={session_id} persona_count={persona_count} questionnaire={questionnaire_id} "
        "model={model}",
        experiment_id=resolved_experiment_id,
        session_id=session_id,
        persona_count=persona_count,
        questionnaire_id=questionnaire.id,
        model=settings.model,
    )
    personas = PersonaFactory().create_demographics_batch(
        PersonaFactoryRequest(
            count=persona_count,
            requested_fields=set(RequestedDemographicField),
            seed=seed,
            experiment_id=resolved_experiment_id,
            generation_config=persona_config or PersonaGenerationConfig(),
        )
    )
    personas_path = write_persona_batch(project_root, personas)
    logger.info(
        "Wrote persona batch experiment_id={experiment_id} personas_path={personas_path}",
        experiment_id=resolved_experiment_id,
        personas_path=personas_path,
    )

    started_at = datetime.now(UTC)
    run_id = _next_run_id(
        project_root=project_root,
        experiment_id=resolved_experiment_id,
        questionnaire=questionnaire,
        settings=settings,
        started_at=started_at,
    )
    paths = resolve_experiment_paths(
        project_root=project_root,
        experiment_id=resolved_experiment_id,
        run_id=run_id,
    )
    all_records: list[ItemResponseRecord] = []
    statuses: list[ResponseStatus] = []
    for index, generated_persona in enumerate(personas.personas, start=1):
        logger.info(
            "Starting persona run {index}/{persona_count} persona_id={persona_id}",
            index=index,
            persona_count=persona_count,
            persona_id=generated_persona.subject_id,
        )
        runtime_persona = _runtime_persona_from_generated(generated_persona)
        records = run_questionnaire(
            persona=runtime_persona,
            questionnaire=questionnaire,
            settings=settings,
            client=client,
            project_root=project_root,
            experiment_id=resolved_experiment_id,
            session_id=session_id,
            run_id=run_id,
            write_session_record=False,
            context=context,
        )
        all_records.extend(records)
        run_status = _response_status(
            records,
            expected_item_count=len(questionnaire.items),
        )
        statuses.append(run_status)
        logger.info(
            "Finished persona run {index}/{persona_count} run_id={run_id} "
            "status={status} errors={error_count}",
            index=index,
            persona_count=persona_count,
            run_id=run_id,
            status=run_status,
            error_count=sum(1 for record in records if record.status == ResponseStatus.FAILED),
        )

    batch_status = ResponseStatus.COMPLETED
    if ResponseStatus.FAILED in statuses:
        batch_status = ResponseStatus.FAILED
    elif ResponseStatus.INVALID in statuses:
        batch_status = ResponseStatus.INVALID

    completed_at = datetime.now(UTC)
    run_record = _run_record(
        resolved_experiment_id,
        session_id,
        run_id,
        [str(persona.subject_id) for persona in personas.personas],
        len(personas.personas),
        questionnaire,
        settings,
        started_at,
        completed_at,
        all_records,
        output_paths={
            "run": str(paths.run_path),
            "responses": str(paths.responses_root),
            "scale": str(paths.scale_path),
        },
    )
    write_json_document(paths.run_path, run_record)
    update_experiment_metadata(paths.metadata_path, run_record)
    _write_scale_copy(paths.scale_path, questionnaire)
    logger.info(
        "Completed persona questionnaire batch experiment_id={experiment_id} "
        "session_id={session_id} status={status} runs={run_count}",
        experiment_id=resolved_experiment_id,
        session_id=session_id,
        status=batch_status,
        run_count=1,
    )
    return BatchRunResult(
        experiment_id=resolved_experiment_id,
        session_id=session_id,
        personas=personas,
        runs=[run_record],
    )


def run_persisted_persona_batch(
    *,
    personas: PersonaBatch,
    questionnaire: Questionnaire,
    settings: ModelSettings,
    client: SyncLlmClient,
    project_root: Path,
    context: str | None = None,
    response_metadata_by_subject: dict[str, dict[str, object]] | None = None,
    initial_histories: dict[str, list[Message]] | None = None,
    run_id: str | None = None,
    run_root_override: Path | None = None,
    retry_failed: bool = False,
    update_metadata: bool = True,
) -> BatchRunResult:
    """Run a previously materialized persona batch without regenerating subjects."""
    experiment_id = validate_experiment_id(personas.experiment_id)
    session_id = normalize_prefixed_uuid("session-")
    started_at = datetime.now(UTC)
    resolved_run_id = run_id or _next_run_id(
        project_root=project_root,
        experiment_id=experiment_id,
        questionnaire=questionnaire,
        settings=settings,
        started_at=started_at,
    )
    paths = resolve_experiment_paths(
        project_root,
        experiment_id,
        resolved_run_id,
        run_root_override=run_root_override,
    )
    subject_ids = [str(persona.subject_id) for persona in personas.personas]
    existing_run = validate_questionnaire_resume(
        paths,
        questionnaire,
        settings,
        subject_ids,
        resolved_run_id,
    )
    if existing_run is not None:
        session_id = existing_run.session_id
        started_at = existing_run.started_at
    else:
        _initialize_questionnaire_run(
            paths=paths,
            experiment_id=experiment_id,
            session_id=session_id,
            run_id=resolved_run_id,
            subject_ids=subject_ids,
            questionnaire=questionnaire,
            settings=settings,
            started_at=started_at,
            update_metadata=update_metadata,
        )
    all_records: list[ItemResponseRecord] = []
    histories: dict[str, list[Message]] = {}
    for generated_persona in personas.personas:
        subject_id = str(generated_persona.subject_id)
        persona = _runtime_persona_from_generated(generated_persona)
        records = run_questionnaire(
            persona=persona,
            questionnaire=questionnaire,
            settings=settings,
            client=client,
            project_root=project_root,
            experiment_id=experiment_id,
            session_id=session_id,
            run_id=resolved_run_id,
            write_session_record=False,
            response_metadata=(response_metadata_by_subject or {}).get(subject_id),
            context=context,
            initial_history=(initial_histories or {}).get(subject_id),
            run_root_override=run_root_override,
            retry_failed=retry_failed,
        )
        all_records.extend(records)
        histories[subject_id] = _history_after_questionnaire(
            persona,
            context,
            (initial_histories or {}).get(subject_id),
            records,
        )

    completed_at = datetime.now(UTC)
    run_record = _run_record(
        experiment_id,
        session_id,
        resolved_run_id,
        subject_ids,
        len(personas.personas),
        questionnaire,
        settings,
        started_at,
        completed_at,
        all_records,
        output_paths={
            "run": str(paths.run_path),
            "responses": str(paths.responses_root),
            "scale": str(paths.scale_path),
        },
    )
    write_json_document(paths.run_path, run_record)
    if update_metadata:
        update_experiment_metadata(paths.metadata_path, run_record)
    _write_scale_copy(paths.scale_path, questionnaire)
    return BatchRunResult(
        experiment_id=experiment_id,
        session_id=session_id,
        personas=personas,
        runs=[run_record],
        histories=histories,
    )


async def run_persisted_persona_batch_async(
    *,
    personas: PersonaBatch,
    questionnaire: Questionnaire,
    settings: ModelSettings,
    client_factory: Callable[[], AsyncLlmClient],
    project_root: Path,
    context: str | None = None,
    response_metadata_by_subject: dict[str, dict[str, object]] | None = None,
    initial_histories: dict[str, list[Message]] | None = None,
    run_id: str | None = None,
    run_root_override: Path | None = None,
    retry_failed: bool = False,
    cancel_event: asyncio.Event | None = None,
    update_metadata: bool = True,
) -> BatchRunResult:
    """Run persisted questionnaire subjects concurrently within a fixed bound."""
    experiment_id = validate_experiment_id(personas.experiment_id)
    session_id = normalize_prefixed_uuid("session-")
    started_at = datetime.now(UTC)
    resolved_run_id = run_id or _next_run_id(
        project_root=project_root,
        experiment_id=experiment_id,
        questionnaire=questionnaire,
        settings=settings,
        started_at=started_at,
    )
    paths = resolve_experiment_paths(
        project_root,
        experiment_id,
        resolved_run_id,
        run_root_override=run_root_override,
    )
    subject_ids = [str(persona.subject_id) for persona in personas.personas]
    existing_run = validate_questionnaire_resume(
        paths,
        questionnaire,
        settings,
        subject_ids,
        resolved_run_id,
    )
    if existing_run is not None:
        session_id = existing_run.session_id
        started_at = existing_run.started_at
    else:
        _initialize_questionnaire_run(
            paths=paths,
            experiment_id=experiment_id,
            session_id=session_id,
            run_id=resolved_run_id,
            subject_ids=subject_ids,
            questionnaire=questionnaire,
            settings=settings,
            started_at=started_at,
            update_metadata=update_metadata,
        )
    cancellation = cancel_event or asyncio.Event()
    semaphore = asyncio.Semaphore(settings.max_concurrency)
    records_by_subject: dict[str, list[ItemResponseRecord]] = {}
    histories: dict[str, list[Message]] = {}

    async def run_subject(generated_persona: GeneratedPersona) -> None:
        async with semaphore:
            if cancellation.is_set():
                return
            subject_id = str(generated_persona.subject_id)
            persona = _runtime_persona_from_generated(generated_persona)
            records = await run_questionnaire_async(
                persona=persona,
                questionnaire=questionnaire,
                settings=settings,
                client=client_factory(),
                project_root=project_root,
                experiment_id=experiment_id,
                session_id=session_id,
                run_id=resolved_run_id,
                write_session_record=False,
                response_metadata=(response_metadata_by_subject or {}).get(subject_id),
                context=context,
                initial_history=(initial_histories or {}).get(subject_id),
                run_root_override=run_root_override,
                retry_failed=retry_failed,
            )
            records_by_subject[subject_id] = records
            histories[subject_id] = _history_after_questionnaire(
                persona,
                context,
                (initial_histories or {}).get(subject_id),
                records,
            )

    await asyncio.gather(*(run_subject(persona) for persona in personas.personas))
    all_records = [
        record
        for persona in personas.personas
        for record in records_by_subject.get(str(persona.subject_id), [])
    ]
    completed_at = datetime.now(UTC)
    run_record = _run_record(
        experiment_id,
        session_id,
        resolved_run_id,
        subject_ids,
        len(personas.personas),
        questionnaire,
        settings,
        started_at,
        completed_at,
        all_records,
        output_paths={
            "run": str(paths.run_path),
            "responses": str(paths.responses_root),
            "scale": str(paths.scale_path),
        },
        cancelled=cancellation.is_set(),
    )
    write_json_document(paths.run_path, run_record)
    if update_metadata:
        update_experiment_metadata(paths.metadata_path, run_record)
    _write_scale_copy(paths.scale_path, questionnaire)
    return BatchRunResult(
        experiment_id=experiment_id,
        session_id=session_id,
        personas=personas,
        runs=[run_record],
        histories=histories,
    )


def _history_after_questionnaire(
    persona: Persona,
    context: str | None,
    initial_history: Sequence[Message] | None,
    records: Sequence[ItemResponseRecord],
) -> list[Message]:
    history = (
        list(initial_history)
        if initial_history is not None
        else [{"role": "system", "content": render_persona_intro(persona, context=context)}]
    )
    for record in records:
        if record.status != ResponseStatus.COMPLETED:
            continue
        user_message = next(
            (message for message in reversed(record.messages) if message.role == "user"),
            None,
        )
        if user_message is None:
            continue
        history.extend(
            [
                {"role": "user", "content": user_message.content},
                {"role": "assistant", "content": record.raw_response or ""},
            ]
        )
    return history


def run_protocol_experiment(
    protocol: ExperimentProtocol,
    questionnaire: Questionnaire,
    settings: ModelSettings,
    client: SyncLlmClient,
    project_root: Path,
    experiment_id: str | None = None,
    context: str | None = None,
) -> BatchRunResult:
    """Expand a protocol design and run its questionnaire conditions.

    Args:
        protocol: Factorial or paired protocol defining base personas, factors,
            levels, and iterations.
        questionnaire: Validated questionnaire definition to administer to each
            expanded protocol subject.
        settings: Provider and model settings used for item-level calls.
        client: Synchronous OpenAI-compatible client wrapper.
        project_root: Repository or experiment root under which experiment files
            are written.
        experiment_id: Optional stable experiment identifier. When omitted, a
            generated three-part identifier is derived from the protocol seed.
        context: Optional supplemental text inserted into every expanded persona's
            initial prompt before questionnaire items are asked.

    Returns:
        Batch run metadata containing the experiment ID, shared session ID,
        expanded personas, and the single aggregate run record.
    """
    resolved_experiment_id = (
        validate_experiment_id(experiment_id)
        if experiment_id
        else generate_experiment_id(protocol.seed)
    )
    session_id = normalize_prefixed_uuid("session-")
    run_settings = settings.model_copy(
        update={"seed": settings.seed if settings.seed is not None else protocol.seed}
    )
    logger.info(
        "Starting protocol experiment experiment_id={experiment_id} "
        "protocol={protocol_name} base_personas={base_count} iterations={iterations}",
        experiment_id=resolved_experiment_id,
        protocol_name=protocol.name,
        base_count=protocol.base_persona_count,
        iterations=protocol.iterations,
    )
    expansion = expand_protocol_personas(protocol, resolved_experiment_id)

    started_at = datetime.now(UTC)
    run_id = _next_run_id(
        project_root=project_root,
        experiment_id=resolved_experiment_id,
        questionnaire=questionnaire,
        settings=run_settings,
        started_at=started_at,
    )
    paths = resolve_experiment_paths(
        project_root=project_root,
        experiment_id=resolved_experiment_id,
        run_id=run_id,
    )
    paths.protocol_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_document(paths.protocol_path, protocol)
    write_persona_batch_at_path(paths.base_personas_path, expansion.base_personas)
    write_persona_batch_at_path(paths.personas_path, expansion.personas)
    write_json_document(
        paths.protocol_assignments_path,
        ProtocolAssignments(assignments=expansion.assignments),
    )

    assignments_by_subject = {
        assignment.subject_id: assignment for assignment in expansion.assignments
    }
    all_records: list[ItemResponseRecord] = []
    statuses: list[ResponseStatus] = []
    for index, generated_persona in enumerate(expansion.personas.personas, start=1):
        assignment = assignments_by_subject[str(generated_persona.subject_id)]
        logger.info(
            "Starting protocol subject {index}/{subject_count} subject_id={subject_id} "
            "condition={condition_id} iteration={iteration_index}",
            index=index,
            subject_count=len(expansion.personas.personas),
            subject_id=generated_persona.subject_id,
            condition_id=assignment.condition_id,
            iteration_index=assignment.iteration_index,
        )
        records = run_questionnaire(
            persona=_runtime_persona_from_generated(generated_persona),
            questionnaire=questionnaire,
            settings=run_settings,
            client=client,
            project_root=project_root,
            experiment_id=resolved_experiment_id,
            session_id=session_id,
            run_id=run_id,
            write_session_record=False,
            response_metadata=_assignment_metadata(protocol, assignment),
            context=context,
        )
        all_records.extend(records)
        statuses.append(
            _response_status(
                records,
                expected_item_count=len(questionnaire.items),
            )
        )

    batch_status = ResponseStatus.COMPLETED
    if ResponseStatus.FAILED in statuses:
        batch_status = ResponseStatus.FAILED
    elif ResponseStatus.INVALID in statuses:
        batch_status = ResponseStatus.INVALID

    completed_at = datetime.now(UTC)
    run_record = _run_record(
        resolved_experiment_id,
        session_id,
        run_id,
        [str(persona.subject_id) for persona in expansion.personas.personas],
        len(expansion.personas.personas),
        questionnaire,
        run_settings,
        started_at,
        completed_at,
        all_records,
        output_paths={
            "run": str(paths.run_path),
            "responses": str(paths.responses_root),
            "scale": str(paths.scale_path),
            "protocol": str(paths.protocol_path),
            "base_personas": str(paths.base_personas_path),
            "assignments": str(paths.protocol_assignments_path),
        },
    )
    write_json_document(paths.run_path, run_record)
    update_experiment_metadata(paths.metadata_path, run_record)
    _write_scale_copy(paths.scale_path, questionnaire)
    logger.info(
        "Completed protocol experiment experiment_id={experiment_id} "
        "status={status} subjects={subject_count}",
        experiment_id=resolved_experiment_id,
        status=batch_status,
        subject_count=len(expansion.personas.personas),
    )
    return BatchRunResult(
        experiment_id=resolved_experiment_id,
        session_id=session_id,
        personas=expansion.personas,
        runs=[run_record],
    )


def _assignment_metadata(
    protocol: ExperimentProtocol,
    assignment: ProtocolAssignment,
) -> dict[str, object]:
    return {
        "protocol_name": protocol.name,
        "base_subject_id": assignment.base_subject_id,
        "condition_id": assignment.condition_id,
        "iteration_index": assignment.iteration_index,
        "factor_values": assignment.factor_values,
        "factor_level_ids": assignment.factor_level_ids,
    }


async def run_questionnaire_async(
    persona: Persona,
    questionnaire: Questionnaire,
    settings: ModelSettings,
    client: AsyncLlmClient,
    project_root: Path,
    experiment_id: str | None = None,
    session_id: str | None = None,
    run_id: str | None = None,
    write_session_record: bool = True,
    response_metadata: dict[str, object] | None = None,
    context: str | None = None,
    initial_history: Sequence[Message] | None = None,
    run_root_override: Path | None = None,
    retry_failed: bool = False,
) -> list[ItemResponseRecord]:
    """Run one questionnaire asynchronously for a single persona.

    Args:
        persona: Persona that the model should assume for the whole session.
        questionnaire: Validated questionnaire definition whose items are asked in
            order.
        settings: Provider and model settings used for each item-level call.
        client: Async OpenAI-compatible client wrapper.
        project_root: Repository or experiment root under which JSONL output is
            written.
        experiment_id: Optional stable experiment identifier. When omitted, a
            generated three-part identifier is used.
        session_id: Optional ``session-[uuid]`` value. When omitted, a new session
            identifier is generated.
        run_id: Optional run directory name. When omitted, a timestamped
            ``run-{questionnaire}-{model}-{timestamp}`` identifier is generated.
        write_session_record: Whether to append run-level metadata records.
        response_metadata: Optional metadata copied into each item response record.
        context: Optional supplemental text inserted into the initial persona
            prompt, such as a vignette or paragraph the model should read before
            answering questionnaire items.

    Returns:
        Item-level response records in questionnaire order. Each record is also
        appended to the subject JSONL file under the resolved run directory.
    """
    resolved_experiment_id = (
        validate_experiment_id(experiment_id) if experiment_id else generate_experiment_id()
    )
    resolved_session_id = normalize_prefixed_uuid("session-", session_id)
    started_at = datetime.now(UTC)
    resolved_run_id = run_id or _next_run_id(
        project_root=project_root,
        experiment_id=resolved_experiment_id,
        questionnaire=questionnaire,
        settings=settings,
        started_at=started_at,
    )
    paths = resolve_experiment_paths(
        project_root=project_root,
        experiment_id=resolved_experiment_id,
        run_id=resolved_run_id,
        run_root_override=run_root_override,
    )
    logger.info(
        "Starting async questionnaire run {run_id} for persona {persona_id} "
        "questionnaire={questionnaire_id} model={model}",
        run_id=resolved_run_id,
        persona_id=persona.persona_id,
        questionnaire_id=questionnaire.id,
        model=settings.model,
    )
    response_path = paths.response_path_for_subject(persona.persona_id)
    persisted = load_item_ledger(response_path)
    if persisted:
        validate_item_ledger(
            questionnaire,
            persona.persona_id,
            resolved_run_id,
            persisted,
        )
    pending = set(
        pending_item_ids(
            questionnaire,
            persisted,
            retry_failed=retry_failed,
        )
    )
    effective = latest_item_attempts(persisted)

    for item in questionnaire.items:
        if item.id not in pending:
            continue
        preceding = (
            [record for record in effective if record.item_order < item.order]
            if bool(questionnaire.metadata.get("retain_history", True))
            else []
        )
        history = _history_after_questionnaire(
            persona,
            context,
            initial_history,
            preceding,
        )
        logger.debug(
            "Asking async item {item_order}/{item_count} item_id={item_id} run_id={run_id}",
            item_order=item.order,
            item_count=len(questionnaire.items),
            item_id=item.id,
            run_id=resolved_run_id,
        )
        messages = [*history, {"role": "user", "content": _question_prompt(item)}]
        allowed_answer_ids = _allowed_answer_ids(item.response_format)
        call_seed = _item_call_seed(settings.seed, persona.persona_id, item.id)
        call_settings = settings.model_copy(update={"seed": call_seed})

        try:
            result = await client.complete(messages, call_settings, allowed_answer_ids)
        except Exception as exc:
            logger.warning(
                "Async provider call failed item_id={item_id} run_id={run_id} "
                "error_type={error_type} error={error}",
                item_id=item.id,
                run_id=resolved_run_id,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            result = LlmQuestionResult(error=str(exc))

        record = _record_from_result(
            resolved_experiment_id,
            persona.persona_id,
            resolved_session_id,
            resolved_run_id,
            questionnaire,
            item,
            messages,
            result,
            call_seed,
            response_metadata,
        )
        append_jsonl_record(response_path, record)
        effective = latest_item_attempts([*effective, record])
        logger.debug(
            "Recorded async item response item_id={item_id} run_id={run_id} status={status}",
            item_id=item.id,
            run_id=resolved_run_id,
            status=record.status,
        )

    records = latest_item_attempts(load_item_ledger(response_path))

    completed_at = datetime.now(UTC)
    run_record = _run_record(
        resolved_experiment_id,
        resolved_session_id,
        resolved_run_id,
        [persona.persona_id],
        1,
        questionnaire,
        settings,
        started_at,
        completed_at,
        records,
        output_paths={
            "run": str(paths.run_path),
            "responses": str(paths.responses_root),
            "scale": str(paths.scale_path),
        },
    )
    if write_session_record:
        write_json_document(paths.run_path, run_record)
        update_experiment_metadata(paths.metadata_path, run_record)
        _write_scale_copy(paths.scale_path, questionnaire)
    logger.info(
        "Completed async questionnaire run {run_id} status={status} errors={error_count} "
        "responses_path={responses_path}",
        run_id=resolved_run_id,
        status=run_record.status,
        error_count=run_record.error_count,
        responses_path=paths.response_path_for_subject(persona.persona_id),
    )
    return records
