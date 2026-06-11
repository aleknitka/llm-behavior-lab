import argparse
import asyncio
import json
import os
import signal
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from llm_behavior_lab.behavioral_tasks.analysis import (
    analyze_task_run,
    export_task_results,
)
from llm_behavior_lab.behavioral_tasks.batch import run_persisted_task_batch_async
from llm_behavior_lab.behavioral_tasks.catalog import (
    load_task_config,
    resolve_behavioral_task,
)
from llm_behavior_lab.client import AsyncOpenAiChatClient, OpenAiChatClient
from llm_behavior_lab.config import (
    DEFAULT_API_BASE_URL,
    DEFAULT_API_KEY,
    DEFAULT_INITIAL_BACKOFF_SECONDS,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MAX_BACKOFF_SECONDS,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MODEL,
    DEFAULT_PROJECT_ROOT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_SECONDS,
)
from llm_behavior_lab.experiments import (
    ExperimentDesign,
    PersonaDesign,
    ProviderDesign,
    ScaleProcedureDesign,
    TaskProcedureDesign,
    create_experiment_design,
    list_persona_fields,
    load_experiment_design,
    load_personas,
    materialize_personas,
    preview_persona_creation,
)
from llm_behavior_lab.models import ModelSettings, ProviderCapabilities
from llm_behavior_lab.personas.factory import (
    PersonaGenerationConfig,
    RequestedDemographicField,
)
from llm_behavior_lab.protocol_runs import (
    create_protocol_experiment,
    create_protocol_run,
    load_protocol_experiment,
)
from llm_behavior_lab.protocols import (
    ExperimentProtocol,
    ProtocolAssignment,
    ProtocolAssignments,
    ProtocolQuestionnaireStep,
    UnifiedExperimentProtocol,
)
from llm_behavior_lab.questionnaires.catalog import (
    QuestionnaireDescriptor,
    describe_questionnaire,
    list_questionnaires,
    resolve_questionnaire,
)
from llm_behavior_lab.runner import run_persisted_persona_batch_async
from llm_behavior_lab.scoring import export_results, score_run
from llm_behavior_lab.storage import load_json_document

LOG_LEVELS = ("TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL")


@dataclass(frozen=True)
class ProviderRuntimeConfig:
    base_url: str
    api_key: str


def parse_features(values: list[str]) -> dict[str, str]:
    features: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected KEY=VALUE feature, got {value!r}")
        key, feature_value = value.split("=", maxsplit=1)
        if not key.strip():
            raise ValueError(f"Expected non-empty feature key, got {value!r}")
        features[key.strip()] = feature_value.strip()
    return features


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--log-level", choices=LOG_LEVELS, default="INFO")


def _persona_design_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--persona-count", type=int, default=100)
    parser.add_argument("--persona-config", type=Path)
    parser.add_argument(
        "--persona-field",
        dest="persona_fields",
        action="append",
        choices=[field.value for field in RequestedDemographicField],
    )
    parser.add_argument("--seed", type=int)


def _provider_design_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_API_BASE_URL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument(
        "--initial-backoff",
        type=float,
        default=DEFAULT_INITIAL_BACKOFF_SECONDS,
    )
    parser.add_argument(
        "--max-backoff",
        type=float,
        default=DEFAULT_MAX_BACKOFF_SECONDS,
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_MAX_CONCURRENCY,
    )
    parser.add_argument(
        "--logprobs",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--structured-outputs", action="store_true", default=False)


async def _run_with_cancellation[ResultT](
    operation: Callable[[asyncio.Event], Awaitable[ResultT]],
) -> ResultT:
    cancel_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    installed = False
    try:
        loop.add_signal_handler(signal.SIGINT, cancel_event.set)
        installed = True
    except (NotImplementedError, RuntimeError):
        pass
    try:
        return await operation(cancel_event)
    finally:
        if installed:
            loop.remove_signal_handler(signal.SIGINT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-behavior-lab")
    commands = parser.add_subparsers(dest="command", required=True)

    questionnaire_list = commands.add_parser(
        "questionnaire-list", help="List coded questionnaires without contacting a provider."
    )
    questionnaire_list.add_argument("--json", action="store_true")

    questionnaire_describe = commands.add_parser(
        "questionnaire-describe",
        help="Describe one coded questionnaire without contacting a provider.",
    )
    questionnaire_describe.add_argument("questionnaire_id")
    questionnaire_describe.add_argument("--json", action="store_true")

    persona_fields = commands.add_parser(
        "persona-fields",
        help="List supported persona fields without creating files.",
    )
    persona_fields.add_argument("--json", action="store_true")

    persona_preview = commands.add_parser(
        "persona-preview",
        help="Preview validated persona settings without creating files.",
    )
    _common_parser(persona_preview)
    persona_preview.add_argument("--experiment-id", required=True)
    _persona_design_parser(persona_preview)
    persona_preview.add_argument("--json", action="store_true")

    protocol_create = commands.add_parser(
        "protocol-create",
        help="Create a canonical experiment protocol or start another matching run.",
    )
    _common_parser(protocol_create)
    protocol_create.add_argument("--file", type=Path, required=True)
    protocol_run = protocol_create.add_mutually_exclusive_group()
    protocol_run.add_argument("--new-run", action="store_true")
    protocol_run.add_argument("--run-id")
    cohort = protocol_create.add_mutually_exclusive_group()
    cohort.add_argument("--cohort-id")
    cohort.add_argument("--persona-seed", type=int)
    protocol_create.add_argument("--run-seed", type=int)
    protocol_create.add_argument("--retry-failed", action="store_true", default=False)
    protocol_create.add_argument("--api-key")

    design = commands.add_parser(
        "scale-design", help="Create a validated questionnaire experiment manifest."
    )
    _common_parser(design)
    design.add_argument("--experiment-id", required=True)
    design.add_argument("--questionnaire", required=True)
    design.add_argument("--questionnaire-param", action="append", default=[])
    _persona_design_parser(design)
    design.add_argument("--protocol", type=Path)
    _provider_design_parser(design)
    design.add_argument("--scoring-model-id")
    design.add_argument("--context")

    personas = commands.add_parser("personas", help="Materialize personas from a design.")
    _common_parser(personas)
    personas.add_argument("--experiment-id", required=True)
    personas.add_argument("--replace", action="store_true")

    task_design = commands.add_parser(
        "task-design", help="Create a validated behavioral-task experiment manifest."
    )
    _common_parser(task_design)
    task_design.add_argument("--experiment-id", required=True)
    task_design.add_argument("--task", required=True)
    task_design.add_argument("--task-config", type=Path)
    _persona_design_parser(task_design)
    task_design.add_argument("--protocol", type=Path)
    _provider_design_parser(task_design)

    run = commands.add_parser("scale-run", help="Run persisted personas through a scale.")
    _common_parser(run)
    run.add_argument("--experiment-id", required=True)
    run.add_argument("--api-key")
    run.add_argument("--run-id")
    run.add_argument("--retry-failed", action="store_true", default=False)

    task_run = commands.add_parser(
        "task-run", help="Run persisted personas through a behavioral task."
    )
    _common_parser(task_run)
    task_run.add_argument("--experiment-id", required=True)
    task_run.add_argument("--api-key")
    task_run.add_argument("--run-id")
    task_run.add_argument("--retry-failed", action="store_true", default=False)

    score = commands.add_parser("scale-score", help="Score a completed scale run.")
    _common_parser(score)
    score.add_argument("--experiment-id", required=True)
    score.add_argument("--run-id")
    score.add_argument("--step-id")
    score.add_argument("--scoring-model-id")

    results = commands.add_parser("scale-results", help="Export scored scale results.")
    _common_parser(results)
    results.add_argument("--experiment-id", required=True)
    results.add_argument("--run-id")
    results.add_argument("--step-id")
    results.add_argument("--scoring-directory")

    task_analyze = commands.add_parser(
        "task-analyze", help="Analyze a completed behavioral-task run."
    )
    _common_parser(task_analyze)
    task_analyze.add_argument("--experiment-id", required=True)
    task_analyze.add_argument("--run-id")
    task_analyze.add_argument("--step-id")
    task_analyze.add_argument("--block-size", type=int, default=20)

    task_results = commands.add_parser(
        "task-results", help="Export behavioral-task results."
    )
    _common_parser(task_results)
    task_results.add_argument("--experiment-id", required=True)
    task_results.add_argument("--run-id")
    task_results.add_argument("--step-id")
    task_results.add_argument("--analysis-directory")
    return parser


def _parameter_summary(descriptor: QuestionnaireDescriptor) -> str:
    if not descriptor.parameters:
        return "none"
    return ", ".join(
        f"{parameter.name} ({'required' if parameter.required else 'optional'})"
        for parameter in descriptor.parameters
    )


def _print_questionnaire_list(descriptors: list[QuestionnaireDescriptor], *, as_json: bool) -> None:
    if as_json:
        print(
            json.dumps(
                [descriptor.model_dump(mode="json") for descriptor in descriptors],
                indent=2,
            )
        )
        return
    for descriptor in descriptors:
        print(
            f"{descriptor.id}: {descriptor.name} "
            f"(version {descriptor.version}, {descriptor.item_count} items; "
            f"parameters: {_parameter_summary(descriptor)})"
        )


def _print_questionnaire_description(descriptor: QuestionnaireDescriptor, *, as_json: bool) -> None:
    if as_json:
        print(descriptor.model_dump_json(indent=2))
        return
    print(f"{descriptor.name} ({descriptor.id})")
    print(f"Version: {descriptor.version}")
    print(f"Language: {descriptor.language or 'unspecified'}")
    print(f"Items: {descriptor.item_count}")
    print(f"Scales: {', '.join(descriptor.scale_ids) or 'none'}")
    print(f"Scoring models: {', '.join(descriptor.scoring_model_ids) or 'none'}")
    print(f"Response formats: {', '.join(descriptor.response_format_types)}")
    print(f"Parameters: {_parameter_summary(descriptor)}")
    for parameter in descriptor.parameters:
        print(f"  {parameter.name}: {parameter.description}")
        if parameter.example is not None:
            print(f"  Example: {parameter.example}")
    print(f"Reference: {descriptor.reference}")
    print(f"Licence: {descriptor.licence}")
    if descriptor.source_url is not None:
        print(f"Source: {descriptor.source_url}")


def configure_logging(level: str) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}",
    )


def load_env_file(project_root: Path) -> dict[str, str]:
    path = project_root / ".env"
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", maxsplit=1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def load_persona_config(path: Path | None) -> PersonaGenerationConfig:
    if path is None:
        return PersonaGenerationConfig()
    return PersonaGenerationConfig.model_validate_json(path.read_text(encoding="utf-8"))


def load_protocol(path: Path) -> ExperimentProtocol:
    return ExperimentProtocol.model_validate_json(path.read_text(encoding="utf-8"))


def load_unified_protocol(path: Path) -> UnifiedExperimentProtocol:
    return UnifiedExperimentProtocol.model_validate_json(path.read_text(encoding="utf-8"))


def _persona_design_from_args(args: argparse.Namespace) -> PersonaDesign:
    requested_fields = (
        set(args.persona_fields)
        if args.persona_fields is not None
        else set(RequestedDemographicField)
    )
    return PersonaDesign(
        count=args.persona_count,
        seed=args.seed,
        requested_fields=requested_fields,
        generation_config=load_persona_config(args.persona_config),
    )


def _print_persona_fields(*, as_json: bool) -> None:
    descriptors = list_persona_fields()
    if as_json:
        print(
            json.dumps(
                [descriptor.model_dump(mode="json") for descriptor in descriptors],
                indent=2,
            )
        )
        return
    for descriptor in descriptors:
        capabilities = ["fixed"]
        if descriptor.supports_range:
            capabilities.append("range")
        if descriptor.supports_probabilities:
            capabilities.append("probabilities")
        allowed = (
            f"; values: {', '.join(descriptor.allowed_values)}" if descriptor.allowed_values else ""
        )
        print(
            f"{descriptor.id.value}: {descriptor.value_type} ({', '.join(capabilities)}{allowed})"
        )


def resolve_provider_config(
    cli_base_url: str | None,
    cli_api_key: str | None,
    env_values: dict[str, str],
) -> ProviderRuntimeConfig:
    return ProviderRuntimeConfig(
        base_url=(
            cli_base_url
            or os.getenv("OPENAI_BASE_URL")
            or env_values.get("OPENAI_BASE_URL")
            or DEFAULT_API_BASE_URL
        ),
        api_key=(
            cli_api_key
            or os.getenv("OPENAI_API_KEY")
            or env_values.get("OPENAI_API_KEY")
            or DEFAULT_API_KEY
        ),
    )


def _run_root(project_root: Path, experiment_id: str, run_id: str | None) -> Path:
    experiment_root = project_root / "experiments" / experiment_id
    runs = sorted(path for path in experiment_root.glob("run-*") if path.is_dir())
    if run_id is not None:
        selected = experiment_root / run_id
        if selected not in runs:
            raise ValueError(f"run not found: {run_id}")
        return selected
    if len(runs) != 1:
        raise ValueError("--run-id is required unless the experiment has exactly one run")
    return runs[0]


def _procedure_root(
    project_root: Path,
    experiment_id: str,
    run_id: str | None,
    step_id: str | None,
) -> Path:
    run_root = _run_root(project_root, experiment_id, run_id)
    if step_id is None:
        return run_root
    step_root = run_root / "steps" / step_id
    if not step_root.is_dir():
        raise ValueError(f"protocol step not found: {step_id}")
    return step_root


def _assignment_metadata(project_root: Path, experiment_id: str):
    experiment_root = project_root / "experiments" / experiment_id
    normalized = experiment_root / "protocol_assignments.json"
    legacy = experiment_root / "protocol_assignments.jsonl"
    if not normalized.exists() and not legacy.exists():
        return {}
    if normalized.exists() and legacy.exists():
        normalized_assignments = load_json_document(
            normalized, ProtocolAssignments
        ).assignments
        legacy_assignments = [
            ProtocolAssignment.model_validate_json(line)
            for line in legacy.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if normalized_assignments != legacy_assignments:
            raise ValueError(
                "conflicting canonical snapshot files: "
                "protocol_assignments.json and protocol_assignments.jsonl"
            )
        assignments = normalized_assignments
    elif normalized.exists():
        assignments = load_json_document(normalized, ProtocolAssignments).assignments
    else:
        assignments = [
            ProtocolAssignment.model_validate_json(line)
            for line in legacy.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    output = {}
    for assignment in assignments:
        output[assignment.subject_id] = {
            "base_subject_id": assignment.base_subject_id,
            "condition_id": assignment.condition_id,
            "iteration_index": assignment.iteration_index,
            "factor_values": assignment.factor_values,
            "factor_level_ids": assignment.factor_level_ids,
        }
    return output


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(getattr(args, "log_level", "INFO"))
    if args.command == "questionnaire-list":
        _print_questionnaire_list(list_questionnaires(), as_json=args.json)
        return 0
    if args.command == "questionnaire-describe":
        _print_questionnaire_description(
            describe_questionnaire(args.questionnaire_id),
            as_json=args.json,
        )
        return 0
    if args.command == "persona-fields":
        _print_persona_fields(as_json=args.json)
        return 0
    if args.command == "persona-preview":
        preview = preview_persona_creation(
            args.experiment_id,
            _persona_design_from_args(args),
        )
        if args.json:
            payload = preview.model_dump(mode="json")
            payload["requested_fields"] = sorted(payload["requested_fields"])
            print(json.dumps(payload, indent=2))
        else:
            print(f"Experiment: {preview.experiment_id}")
            print(f"Persona count: {preview.count}")
            print(f"Seed: {preview.seed}")
            print(
                "Requested fields: "
                + ", ".join(sorted(field.value for field in preview.requested_fields))
            )
            print(preview.generation_config.model_dump_json(indent=2))
        return 0
    if args.command == "protocol-create":
        protocol = load_unified_protocol(args.file)
        experiment_root = (
            args.project_root / "experiments" / protocol.experiment_id
        )
        if not experiment_root.exists():
            if args.run_id is not None:
                raise ValueError("cannot resume a protocol experiment that does not exist")
            created = create_protocol_experiment(args.project_root, protocol)
            print(created.protocol_path)
            return 0
        load_protocol_experiment(args.project_root, protocol.experiment_id)
        if args.run_id is not None and (
            args.cohort_id is not None or args.persona_seed is not None
        ):
            raise ValueError("--run-id cannot be combined with cohort or persona seed")
        if args.run_id is None and not args.new_run:
            if not sys.stdin.isatty():
                raise ValueError(
                    "experiment already exists; non-interactive callers must pass --new-run"
                )
            answer = input("Protocol matches an existing experiment. Start another run? [y/N] ")
            if answer.strip().lower() not in {"y", "yes"}:
                return 0
        cohort_id = args.cohort_id
        persona_seed = args.persona_seed
        if (
            args.run_id is None
            and sys.stdin.isatty()
            and cohort_id is None
            and persona_seed is None
        ):
            answer = input("Reuse an existing persona cohort? [Y/n] ")
            if answer.strip().lower() in {"n", "no"}:
                seed_text = input(
                    f"Persona seed [{protocol.persona_seed}]: "
                ).strip()
                persona_seed = (
                    int(seed_text) if seed_text else protocol.persona_seed
                )
        run_seed = args.run_seed
        if args.run_id is None and sys.stdin.isatty() and run_seed is None:
            seed_text = input(f"Run seed [{protocol.run_seed}]: ").strip()
            run_seed = int(seed_text) if seed_text else protocol.run_seed
        runtime = resolve_provider_config(
            protocol.provider.base_url,
            args.api_key,
            load_env_file(args.project_root),
        )
        result = create_protocol_run(
            args.project_root,
            protocol,
            cohort_id=cohort_id,
            persona_seed=persona_seed,
            run_seed=run_seed,
            run_id=args.run_id,
            retry_failed=args.retry_failed,
            api_key=runtime.api_key,
        )
        print(result.run_id)
        return 0
    if args.command in {"scale-design", "task-design"}:
        protocol = load_protocol(args.protocol) if args.protocol else None
        personas = None
        if protocol is None:
            personas = _persona_design_from_args(args)
        if args.command == "scale-design":
            procedure = ScaleProcedureDesign(
                questionnaire_id=args.questionnaire,
                questionnaire_parameters=parse_features(args.questionnaire_param),
                scoring_model_id=args.scoring_model_id,
                context=args.context,
            )
            questionnaire = resolve_questionnaire(
                procedure.questionnaire_id, procedure.questionnaire_parameters
            )
            if procedure.scoring_model_id is not None and all(
                model.id != procedure.scoring_model_id
                for model in questionnaire.scoring_models
            ):
                raise ValueError(
                    f"unknown scoring model {procedure.scoring_model_id!r} "
                    f"for {questionnaire.id}"
                )
        else:
            task_config = load_task_config(args.task_config)
            procedure = TaskProcedureDesign(
                task_id=args.task,
                task_config=task_config.model_dump(mode="json"),
            )
            resolve_behavioral_task(procedure.task_id, procedure.task_config)
        design = ExperimentDesign(
            experiment_id=args.experiment_id,
            procedure=procedure,
            personas=personas,
            protocol=protocol,
            provider=ProviderDesign(
                model=args.model,
                base_url=args.base_url,
                temperature=args.temperature,
                timeout_seconds=args.timeout,
                max_attempts=args.max_attempts,
                initial_backoff_seconds=args.initial_backoff,
                max_backoff_seconds=args.max_backoff,
                max_concurrency=args.max_concurrency,
                seed=args.seed,
                supports_structured_outputs=args.structured_outputs,
                supports_logprobs=args.logprobs,
            ),
        )
        path = create_experiment_design(args.project_root, design)
        print(path)
        return 0
    if args.command == "personas":
        design = load_experiment_design(args.project_root, args.experiment_id)
        batch = materialize_personas(
            args.project_root,
            design,
            replace=args.replace,
        )
        path = args.project_root / "experiments" / args.experiment_id / "personas.json"
        print(f"Created {len(batch.personas)} personas at {path}")
        return 0
    if args.command in {"scale-run", "task-run"}:
        design = load_experiment_design(args.project_root, args.experiment_id)
        runtime = resolve_provider_config(
            design.provider.base_url,
            args.api_key,
            load_env_file(args.project_root),
        )
        settings = ModelSettings(
            model=design.provider.model,
            provider_base_url=runtime.base_url,
            temperature=design.provider.temperature,
            timeout_seconds=design.provider.timeout_seconds,
            max_attempts=design.provider.max_attempts,
            initial_backoff_seconds=design.provider.initial_backoff_seconds,
            max_backoff_seconds=design.provider.max_backoff_seconds,
            max_concurrency=design.provider.max_concurrency,
            seed=design.provider.seed,
            capabilities=ProviderCapabilities(
                supports_structured_outputs=design.provider.supports_structured_outputs,
                supports_logprobs=design.provider.supports_logprobs,
            ),
        )
        if args.command == "scale-run":
            procedure = design.procedure
            if not isinstance(procedure, ScaleProcedureDesign):
                raise ValueError("experiment design is not a scale procedure")

            async def run_scale(cancel_event: asyncio.Event):
                return await run_persisted_persona_batch_async(
                    personas=load_personas(args.project_root, args.experiment_id),
                    questionnaire=resolve_questionnaire(
                        procedure.questionnaire_id,
                        procedure.questionnaire_parameters,
                    ),
                    settings=settings,
                    client_factory=lambda: AsyncOpenAiChatClient(
                        api_key=runtime.api_key,
                        base_url=runtime.base_url,
                    ),
                    project_root=args.project_root,
                    context=procedure.context,
                    response_metadata_by_subject=_assignment_metadata(
                        args.project_root,
                        args.experiment_id,
                    ),
                    run_id=args.run_id,
                    retry_failed=args.retry_failed,
                    cancel_event=cancel_event,
                )

            result = asyncio.run(_run_with_cancellation(run_scale))
            print(result.runs[0].run_id)
            return 0
        if not isinstance(design.procedure, TaskProcedureDesign):
            raise ValueError("experiment design is not a task procedure")
        task = resolve_behavioral_task(
            design.procedure.task_id, design.procedure.task_config
        )
        result = asyncio.run(
            run_persisted_task_batch_async(
                personas=load_personas(args.project_root, args.experiment_id),
                task=task,
                settings=settings,
                client_factory=lambda: OpenAiChatClient(
                    api_key=runtime.api_key, base_url=runtime.base_url
                ),
                project_root=args.project_root,
                run_id=args.run_id,
                concurrency=settings.max_concurrency,
                resume=True,
                retry_failed=args.retry_failed,
                response_metadata_by_subject=_assignment_metadata(
                    args.project_root, args.experiment_id
                ),
            )
        )
        print(result.run_id)
        return 0
    if args.command == "scale-score":
        if args.step_id is None:
            design = load_experiment_design(args.project_root, args.experiment_id)
            if not isinstance(design.procedure, ScaleProcedureDesign):
                raise ValueError("experiment design is not a scale procedure")
            default_scoring_model_id = design.procedure.scoring_model_id
        else:
            protocol = load_protocol_experiment(
                args.project_root, args.experiment_id
            ).protocol
            step = next(
                (step for step in protocol.steps if step.id == args.step_id),
                None,
            )
            if not isinstance(step, ProtocolQuestionnaireStep):
                raise ValueError("protocol step is not a questionnaire")
            default_scoring_model_id = step.scoring_model_id
        run_root = _procedure_root(
            args.project_root, args.experiment_id, args.run_id, args.step_id
        )
        scoring_model_id = args.scoring_model_id or default_scoring_model_id
        result = score_run(run_root, scoring_model_id)
        print(result.output_root)
        return 0
    if args.command == "scale-results":
        run_root = _procedure_root(
            args.project_root, args.experiment_id, args.run_id, args.step_id
        )
        result = export_results(run_root, args.scoring_directory)
        print(result.output_root)
        return 0
    if args.command == "task-analyze":
        run_root = _procedure_root(
            args.project_root, args.experiment_id, args.run_id, args.step_id
        )
        result = analyze_task_run(run_root, args.block_size)
        print(result.output_root)
        return 0
    run_root = _procedure_root(
        args.project_root, args.experiment_id, args.run_id, args.step_id
    )
    result = export_task_results(run_root, args.analysis_directory)
    print(result.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
