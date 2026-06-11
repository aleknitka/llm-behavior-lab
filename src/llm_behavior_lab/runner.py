import hashlib
from collections.abc import Sequence
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
from llm_behavior_lab.storage import (
    append_jsonl_record,
    build_run_directory_name,
    generate_experiment_id,
    normalize_prefixed_uuid,
    resolve_experiment_paths,
    slugify_model_name,
    validate_experiment_id,
    write_jsonl_records,
    write_persona_batch_jsonl,
    write_persona_batch_jsonl_at_path,
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
        supports_structured_outputs=settings.capabilities.supports_structured_outputs,
        supports_logprobs=settings.capabilities.supports_logprobs,
    )


def _response_status(records: Sequence[ItemResponseRecord]) -> ResponseStatus:
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
    completed_at: datetime,
    records: Sequence[ItemResponseRecord],
    output_paths: dict[str, str],
) -> RunRecord:
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
        status=_response_status(records),
        error_count=sum(1 for record in records if record.status == ResponseStatus.FAILED),
        item_count=len(records),
        output_paths=output_paths,
        metadata={"scoring_model_id": None, "base_seed": settings.seed},
    )


def _write_scale_copy(path: Path, questionnaire: Questionnaire) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(questionnaire.model_dump_json(indent=2), encoding="utf-8")


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
    history: list[Message] = (
        list(initial_history)
        if initial_history is not None
        else [{"role": "system", "content": render_persona_intro(persona, context=context)}]
    )
    records: list[ItemResponseRecord] = []

    for item in questionnaire.items:
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
        append_jsonl_record(paths.response_path_for_subject(persona.persona_id), record)
        records.append(record)
        logger.debug(
            "Recorded item response item_id={item_id} run_id={run_id} status={status}",
            item_id=item.id,
            run_id=resolved_run_id,
            status=record.status,
        )

        if result.error is None and bool(questionnaire.metadata.get("retain_history", True)):
            history = [
                *messages,
                {"role": "assistant", "content": result.raw_response or ""},
            ]

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
        append_jsonl_record(paths.run_path, run_record)
        append_jsonl_record(paths.metadata_path, run_record)
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
    personas_path = write_persona_batch_jsonl(project_root, personas)
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
        run_status = _response_status(records)
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
    append_jsonl_record(paths.run_path, run_record)
    append_jsonl_record(paths.metadata_path, run_record)
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
    append_jsonl_record(paths.run_path, run_record)
    append_jsonl_record(paths.metadata_path, run_record)
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
            (
                message
                for message in reversed(record.messages)
                if message.role == "user"
            ),
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
    paths.protocol_path.write_text(
        protocol.model_dump_json(indent=2),
        encoding="utf-8",
    )
    write_persona_batch_jsonl_at_path(paths.base_personas_path, expansion.base_personas)
    write_persona_batch_jsonl_at_path(paths.personas_path, expansion.personas)
    write_jsonl_records(paths.protocol_assignments_path, expansion.assignments)

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
        statuses.append(_response_status(records))

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
    append_jsonl_record(paths.run_path, run_record)
    append_jsonl_record(paths.metadata_path, run_record)
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
    )
    logger.info(
        "Starting async questionnaire run {run_id} for persona {persona_id} "
        "questionnaire={questionnaire_id} model={model}",
        run_id=resolved_run_id,
        persona_id=persona.persona_id,
        questionnaire_id=questionnaire.id,
        model=settings.model,
    )
    history: list[Message] = [
        {"role": "system", "content": render_persona_intro(persona, context=context)}
    ]
    records: list[ItemResponseRecord] = []

    for item in questionnaire.items:
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
        append_jsonl_record(paths.response_path_for_subject(persona.persona_id), record)
        records.append(record)
        logger.debug(
            "Recorded async item response item_id={item_id} run_id={run_id} status={status}",
            item_id=item.id,
            run_id=resolved_run_id,
            status=record.status,
        )

        if result.error is None and bool(questionnaire.metadata.get("retain_history", True)):
            history = [
                *messages,
                {"role": "assistant", "content": result.raw_response or ""},
            ]

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
        append_jsonl_record(paths.run_path, run_record)
        append_jsonl_record(paths.metadata_path, run_record)
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
