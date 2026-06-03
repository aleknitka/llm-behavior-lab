import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from llm_psych_scales.client import OpenAiChatClient
from llm_psych_scales.config import (
    DEFAULT_API_BASE_URL,
    DEFAULT_API_KEY,
    DEFAULT_MODEL,
    DEFAULT_PROJECT_ROOT,
    DEFAULT_SUPPORTS_LOGPROBS,
    DEFAULT_SUPPORTS_STRUCTURED_OUTPUTS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_SECONDS,
)
from llm_psych_scales.models import ModelSettings, ProviderCapabilities
from llm_psych_scales.questionnaires.bfi10 import BFI_10
from llm_psych_scales.runner import run_persona_questionnaire_batch

LOG_LEVELS = ("TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL")


@dataclass(frozen=True)
class ProviderRuntimeConfig:
    base_url: str
    api_key: str


def parse_features(values: list[str]) -> dict[str, str]:
    features: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            msg = f"Expected KEY=VALUE feature, got {value!r}"
            raise ValueError(msg)
        key, feature_value = value.split("=", maxsplit=1)
        key = key.strip()
        if not key:
            msg = f"Expected non-empty feature key, got {value!r}"
            raise ValueError(msg)
        features[key] = feature_value.strip()
    return features


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-psych-scales",
        description="Run the BFI-10 questionnaire against an OpenAI-compatible model.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--persona-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-level", choices=LOG_LEVELS, default="INFO")
    parser.add_argument(
        "--logprobs",
        action="store_true",
        default=DEFAULT_SUPPORTS_LOGPROBS,
        help="Request token logprobs when the provider supports them.",
    )
    parser.add_argument(
        "--structured-outputs",
        action="store_true",
        default=DEFAULT_SUPPORTS_STRUCTURED_OUTPUTS,
        help="Mark the provider as supporting structured outputs.",
    )
    return parser


def configure_logging(level: str) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}",
    )


def load_env_file(project_root: Path) -> dict[str, str]:
    env_path = project_root / ".env"
    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", maxsplit=1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            values[key] = value
    return values


def resolve_provider_config(
    cli_base_url: str | None,
    cli_api_key: str | None,
    env_values: dict[str, str],
) -> ProviderRuntimeConfig:
    base_url = (
        cli_base_url
        or os.getenv("OPENAI_BASE_URL")
        or env_values.get("OPENAI_BASE_URL")
        or DEFAULT_API_BASE_URL
    )
    api_key = (
        cli_api_key
        or os.getenv("OPENAI_API_KEY")
        or env_values.get("OPENAI_API_KEY")
        or DEFAULT_API_KEY
    )
    return ProviderRuntimeConfig(base_url=base_url, api_key=api_key)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    env_values = load_env_file(args.project_root)
    provider_config = resolve_provider_config(
        cli_base_url=args.base_url,
        cli_api_key=args.api_key,
        env_values=env_values,
    )
    logger.info(
        "Configured run model={model} base_url={base_url} persona_count={persona_count}",
        model=args.model,
        base_url=provider_config.base_url,
        persona_count=args.persona_count,
    )

    settings = ModelSettings(
        model=args.model,
        provider_base_url=provider_config.base_url,
        temperature=args.temperature,
        timeout_seconds=args.timeout,
        capabilities=ProviderCapabilities(
            supports_structured_outputs=args.structured_outputs,
            supports_logprobs=args.logprobs,
        ),
    )
    client = OpenAiChatClient(api_key=provider_config.api_key, base_url=provider_config.base_url)
    result = run_persona_questionnaire_batch(
        questionnaire=BFI_10,
        settings=settings,
        client=client,
        project_root=args.project_root,
        experiment_id=args.experiment_id,
        persona_count=args.persona_count,
        seed=args.seed,
    )

    failed = sum(run.error_count for run in result.runs)
    print(
        f"Saved {len(result.runs)} runs for {len(result.personas.personas)} personas "
        f"under experiments/{result.experiment_id}"
    )
    if failed:
        print(f"{failed} records contain provider errors")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
