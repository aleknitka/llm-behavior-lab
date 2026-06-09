from pathlib import Path

import pytest

from llm_behavior_lab.main import (
    build_parser,
    load_env_file,
    load_persona_config,
    load_protocol,
    parse_features,
    resolve_provider_config,
)
from llm_behavior_lab.personas.factory import RequestedDemographicField


def test_parse_features_converts_key_value_pairs() -> None:
    features = parse_features(["age=35", "country=Poland"])

    assert features == {"age": "35", "country": "Poland"}


def test_parse_features_rejects_missing_separator() -> None:
    with pytest.raises(ValueError, match="Expected KEY=VALUE"):
        parse_features(["age"])


def test_parser_defaults_to_bfi10_and_local_model() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_parser_accepts_staged_design_command() -> None:
    args = build_parser().parse_args(
        [
            "scale-design",
            "--experiment-id",
            "pilot-study-one",
            "--questionnaire",
            "bfi_10",
            "--persona-count",
            "10",
        ]
    )

    assert args.command == "scale-design"
    assert args.questionnaire == "bfi_10"
    assert args.project_root == Path(".")
    assert args.experiment_id == "pilot-study-one"
    assert args.persona_count == 10


def test_parser_accepts_log_level() -> None:
    args = build_parser().parse_args(
        ["personas", "--experiment-id", "pilot-study-one", "--log-level", "DEBUG"]
    )

    assert args.log_level == "DEBUG"


def test_parser_rejects_old_output_argument() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--output", "runs/questionnaire_responses.jsonl"])


def test_load_env_file_reads_simple_key_value_pairs(tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "# local provider\n"
        "OPENAI_BASE_URL=http://localhost:9999/v1\n"
        "OPENAI_API_KEY='secret'\n",  # pragma: allowlist secret
        encoding="utf-8",
    )

    values = load_env_file(tmp_path)

    assert values == {
        "OPENAI_BASE_URL": "http://localhost:9999/v1",
        "OPENAI_API_KEY": "secret",  # pragma: allowlist secret
    }


def test_load_persona_config_reads_weighted_field_probabilities(tmp_path) -> None:
    config_path = tmp_path / "persona-config.json"
    config_path.write_text(
        '{"field_probabilities": {"country": {"PL": 0.8, "DE": 0.2}, '
        '"affluence_level": {"middle": 1.0}}}',
        encoding="utf-8",
    )

    config = load_persona_config(config_path)

    assert config.field_probabilities[RequestedDemographicField.COUNTRY] == {
        "PL": 0.8,
        "DE": 0.2,
    }
    assert config.field_probabilities[RequestedDemographicField.AFFLUENCE_LEVEL] == {
        "middle": 1.0
    }


def test_parser_accepts_protocol_file() -> None:
    args = build_parser().parse_args(
        [
            "scale-design",
            "--experiment-id",
            "pilot-study-one",
            "--questionnaire",
            "bfi_10",
            "--protocol",
            "protocol.json",
        ]
    )

    assert args.protocol == Path("protocol.json")


def test_parser_accepts_task_workflow_commands() -> None:
    design = build_parser().parse_args(
        [
            "task-design",
            "--experiment-id",
            "card-task-one",
            "--task",
            "four-deck-card-task",
            "--task-config",
            "task.json",
        ]
    )
    run = build_parser().parse_args(
        ["task-run", "--experiment-id", "card-task-one", "--concurrency", "4"]
    )
    analyze = build_parser().parse_args(
        ["task-analyze", "--experiment-id", "card-task-one"]
    )

    assert design.task == "four-deck-card-task"
    assert design.task_config == Path("task.json")
    assert run.concurrency == 4
    assert analyze.command == "task-analyze"


def test_load_protocol_reads_valid_protocol_json(tmp_path) -> None:
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(
        """
        {
          "version": "1.0",
          "name": "gender-pair",
          "base_persona_count": 1,
          "seed": 123,
          "iterations": 1,
          "requested_fields": ["age", "gender"],
          "factors": [
            {
              "name": "gender",
              "field": "gender",
              "levels": [
                {"id": "female", "value": "female"},
                {"id": "male", "value": "male"}
              ]
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    protocol = load_protocol(protocol_path)

    assert protocol.name == "gender-pair"
    assert protocol.factors[0].field == RequestedDemographicField.GENDER


def test_resolve_provider_config_prefers_cli_then_environment_then_dotenv(monkeypatch) -> None:
    env_values = {
        "OPENAI_BASE_URL": "http://dotenv:1234/v1",
        "OPENAI_API_KEY": "dotenv-key",  # pragma: allowlist secret
    }
    monkeypatch.setenv("OPENAI_BASE_URL", "http://env:1234/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")  # pragma: allowlist secret

    config = resolve_provider_config(
        cli_base_url="http://cli:1234/v1",
        cli_api_key="cli-key",  # pragma: allowlist secret
        env_values=env_values,
    )

    assert config.base_url == "http://cli:1234/v1"
    assert config.api_key == "cli-key"  # pragma: allowlist secret


def test_resolve_provider_config_uses_dotenv_when_cli_and_environment_absent(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    config = resolve_provider_config(
        cli_base_url=None,
        cli_api_key=None,
        env_values={
            "OPENAI_BASE_URL": "http://dotenv:1234/v1",
            "OPENAI_API_KEY": "dotenv-key",  # pragma: allowlist secret
        },
    )

    assert config.base_url == "http://dotenv:1234/v1"
    assert config.api_key == "dotenv-key"  # pragma: allowlist secret
