import argparse
import asyncio
import os
import sys
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
from llm_behavior_lab.client import OpenAiChatClient
from llm_behavior_lab.config import (
    DEFAULT_API_BASE_URL,
    DEFAULT_API_KEY,
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
    load_experiment_design,
    load_personas,
    materialize_personas,
)
from llm_behavior_lab.models import ModelSettings, ProviderCapabilities
from llm_behavior_lab.personas.factory import PersonaGenerationConfig
from llm_behavior_lab.protocols import ExperimentProtocol, ProtocolAssignment
from llm_behavior_lab.questionnaires.catalog import resolve_questionnaire
from llm_behavior_lab.runner import run_persisted_persona_batch
from llm_behavior_lab.scoring import export_results, score_run

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-behavior-lab")
    commands = parser.add_subparsers(dest="command", required=True)

    design = commands.add_parser(
        "scale-design", help="Create a validated questionnaire experiment manifest."
    )
    _common_parser(design)
    design.add_argument("--experiment-id", required=True)
    design.add_argument("--questionnaire", required=True)
    design.add_argument("--questionnaire-param", action="append", default=[])
    design.add_argument("--persona-count", type=int, default=100)
    design.add_argument("--persona-config", type=Path)
    design.add_argument("--protocol", type=Path)
    design.add_argument("--model", default=DEFAULT_MODEL)
    design.add_argument("--base-url", default=DEFAULT_API_BASE_URL)
    design.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    design.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    design.add_argument("--seed", type=int)
    design.add_argument("--scoring-model-id", default="default")
    design.add_argument("--context")
    design.add_argument("--logprobs", action=argparse.BooleanOptionalAction, default=True)
    design.add_argument("--structured-outputs", action="store_true", default=False)

    personas = commands.add_parser("personas", help="Materialize personas from a design.")
    _common_parser(personas)
    personas.add_argument("--experiment-id", required=True)

    task_design = commands.add_parser(
        "task-design", help="Create a validated behavioral-task experiment manifest."
    )
    _common_parser(task_design)
    task_design.add_argument("--experiment-id", required=True)
    task_design.add_argument("--task", required=True)
    task_design.add_argument("--task-config", type=Path)
    task_design.add_argument("--persona-count", type=int, default=100)
    task_design.add_argument("--persona-config", type=Path)
    task_design.add_argument("--protocol", type=Path)
    task_design.add_argument("--model", default=DEFAULT_MODEL)
    task_design.add_argument("--base-url", default=DEFAULT_API_BASE_URL)
    task_design.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    task_design.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    task_design.add_argument("--seed", type=int)
    task_design.add_argument("--logprobs", action=argparse.BooleanOptionalAction, default=True)
    task_design.add_argument("--structured-outputs", action="store_true", default=False)

    run = commands.add_parser("scale-run", help="Run persisted personas through a scale.")
    _common_parser(run)
    run.add_argument("--experiment-id", required=True)
    run.add_argument("--api-key")

    task_run = commands.add_parser(
        "task-run", help="Run persisted personas through a behavioral task."
    )
    _common_parser(task_run)
    task_run.add_argument("--experiment-id", required=True)
    task_run.add_argument("--api-key")
    task_run.add_argument("--run-id")
    task_run.add_argument("--concurrency", type=int, default=4)
    task_run.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    task_run.add_argument("--retry-failed", action="store_true", default=False)

    score = commands.add_parser("scale-score", help="Score a completed scale run.")
    _common_parser(score)
    score.add_argument("--experiment-id", required=True)
    score.add_argument("--run-id")
    score.add_argument("--scoring-model-id")

    results = commands.add_parser("scale-results", help="Export scored scale results.")
    _common_parser(results)
    results.add_argument("--experiment-id", required=True)
    results.add_argument("--run-id")
    results.add_argument("--scoring-directory")

    task_analyze = commands.add_parser(
        "task-analyze", help="Analyze a completed behavioral-task run."
    )
    _common_parser(task_analyze)
    task_analyze.add_argument("--experiment-id", required=True)
    task_analyze.add_argument("--run-id")
    task_analyze.add_argument("--block-size", type=int, default=20)

    task_results = commands.add_parser(
        "task-results", help="Export behavioral-task results."
    )
    _common_parser(task_results)
    task_results.add_argument("--experiment-id", required=True)
    task_results.add_argument("--run-id")
    task_results.add_argument("--analysis-directory")
    return parser


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


def _assignment_metadata(project_root: Path, experiment_id: str):
    path = project_root / "experiments" / experiment_id / "protocol_assignments.jsonl"
    if not path.exists():
        return {}
    output = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        assignment = ProtocolAssignment.model_validate_json(line)
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
    configure_logging(args.log_level)
    if args.command in {"scale-design", "task-design"}:
        protocol = load_protocol(args.protocol) if args.protocol else None
        personas = None
        if protocol is None:
            personas = PersonaDesign(
                count=args.persona_count,
                seed=args.seed,
                generation_config=load_persona_config(args.persona_config),
            )
        if args.command == "scale-design":
            procedure = ScaleProcedureDesign(
                questionnaire_id=args.questionnaire,
                questionnaire_parameters=parse_features(args.questionnaire_param),
                scoring_model_id=args.scoring_model_id,
                context=args.context,
            )
            resolve_questionnaire(
                procedure.questionnaire_id, procedure.questionnaire_parameters
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
        batch = materialize_personas(args.project_root, design)
        print(f"Created {len(batch.personas)} personas")
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
            seed=design.provider.seed,
            capabilities=ProviderCapabilities(
                supports_structured_outputs=design.provider.supports_structured_outputs,
                supports_logprobs=design.provider.supports_logprobs,
            ),
        )
        if args.command == "scale-run":
            if not isinstance(design.procedure, ScaleProcedureDesign):
                raise ValueError("experiment design is not a scale procedure")
            result = run_persisted_persona_batch(
                personas=load_personas(args.project_root, args.experiment_id),
                questionnaire=resolve_questionnaire(
                    design.procedure.questionnaire_id,
                    design.procedure.questionnaire_parameters,
                ),
                settings=settings,
                client=OpenAiChatClient(
                    api_key=runtime.api_key, base_url=runtime.base_url
                ),
                project_root=args.project_root,
                context=design.procedure.context,
                response_metadata_by_subject=_assignment_metadata(
                    args.project_root, args.experiment_id
                ),
            )
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
                concurrency=args.concurrency,
                resume=args.resume,
                retry_failed=args.retry_failed,
                response_metadata_by_subject=_assignment_metadata(
                    args.project_root, args.experiment_id
                ),
            )
        )
        print(result.run_id)
        return 0
    if args.command == "scale-score":
        run_root = _run_root(args.project_root, args.experiment_id, args.run_id)
        result = score_run(run_root, args.scoring_model_id)
        print(result.output_root)
        return 0
    if args.command == "scale-results":
        run_root = _run_root(args.project_root, args.experiment_id, args.run_id)
        result = export_results(run_root, args.scoring_directory)
        print(result.output_root)
        return 0
    if args.command == "task-analyze":
        run_root = _run_root(args.project_root, args.experiment_id, args.run_id)
        result = analyze_task_run(run_root, args.block_size)
        print(result.output_root)
        return 0
    run_root = _run_root(args.project_root, args.experiment_id, args.run_id)
    result = export_task_results(run_root, args.analysis_directory)
    print(result.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
