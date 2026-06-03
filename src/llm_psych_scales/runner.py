import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from loguru import logger

from llm_psych_scales.client import AsyncLlmClient, SyncLlmClient
from llm_psych_scales.models import (
    LlmQuestionResult,
    ModelSettings,
    Persona,
)
from llm_psych_scales.personas.factory import (
    GeneratedPersona,
    PersonaBatch,
    PersonaFactory,
    PersonaFactoryRequest,
    RequestedDemographicField,
)
from llm_psych_scales.prompting import render_persona_intro
from llm_psych_scales.questionnaires.base.response_formats import (
    LikertFormat,
    MultipleChoiceFormat,
    NumericFormat,
    ResponseFormat,
    SingleChoiceFormat,
    TextFormat,
)
from llm_psych_scales.questionnaires.base.scale import Item, Questionnaire
from llm_psych_scales.responses.base import (
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
from llm_psych_scales.storage import (
    append_jsonl_record,
    build_run_directory_name,
    generate_experiment_id,
    normalize_prefixed_uuid,
    resolve_experiment_paths,
    slugify_model_name,
    validate_experiment_id,
    write_persona_batch_jsonl,
)

Message = dict[str, str]


@dataclass(frozen=True)
class BatchRunResult:
    experiment_id: str
    session_id: str
    personas: PersonaBatch
    runs: list[RunRecord]


def _allowed_answer_ids(response_format: ResponseFormat) -> list[str]:
    if isinstance(response_format, LikertFormat):
        return [
            str(value)
            for value in range(response_format.min_value, response_format.max_value + 1)
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
        f"{item.text}\n\n"
        f"Allowed answers:\n{answers}\n\n"
        "Reply with exactly one allowed answer id."
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
) -> ItemResponseRecord:
    answer = _answer_from_result(item, result)
    status = ResponseStatus.FAILED if result.error else ResponseStatus.COMPLETED
    if result.error is None and answer is None:
        status = ResponseStatus.INVALID
    metadata: dict[str, object] = {"experiment_id": experiment_id}
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
) -> list[ItemResponseRecord]:
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
        "Starting questionnaire run {run_id} for persona {persona_id} "
        "questionnaire={questionnaire_id} model={model}",
        run_id=resolved_run_id,
        persona_id=persona.persona_id,
        questionnaire_id=questionnaire.id,
        model=settings.model,
    )
    history: list[Message] = [{"role": "system", "content": render_persona_intro(persona)}]
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
            logger.exception(
                "Provider call failed item_id={item_id} run_id={run_id}",
                item_id=item.id,
                run_id=resolved_run_id,
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
        )
        append_jsonl_record(paths.response_path_for_subject(persona.persona_id), record)
        records.append(record)
        logger.debug(
            "Recorded item response item_id={item_id} run_id={run_id} status={status}",
            item_id=item.id,
            run_id=resolved_run_id,
            status=record.status,
        )

        if bool(questionnaire.metadata.get("retain_history", True)):
            history = [
                *messages,
                {"role": "assistant", "content": result.raw_response or result.error or ""},
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
) -> BatchRunResult:
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
) -> list[ItemResponseRecord]:
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
    history: list[Message] = [{"role": "system", "content": render_persona_intro(persona)}]
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
            logger.exception(
                "Async provider call failed item_id={item_id} run_id={run_id}",
                item_id=item.id,
                run_id=resolved_run_id,
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
        )
        append_jsonl_record(paths.response_path_for_subject(persona.persona_id), record)
        records.append(record)
        logger.debug(
            "Recorded async item response item_id={item_id} run_id={run_id} status={status}",
            item_id=item.id,
            run_id=resolved_run_id,
            status=record.status,
        )

        if bool(questionnaire.metadata.get("retain_history", True)):
            history = [
                *messages,
                {"role": "assistant", "content": result.raw_response or result.error or ""},
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
