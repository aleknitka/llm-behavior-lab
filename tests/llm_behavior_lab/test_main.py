import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_behavior_lab.experiments import (
    ExperimentDesign,
    PersonaDesign,
    ProviderDesign,
    ScaleProcedureDesign,
    create_experiment_design,
    create_personas,
    load_personas,
)
from llm_behavior_lab.main import (
    _run_with_cancellation,
    build_parser,
    load_env_file,
    load_persona_config,
    load_protocol,
    main,
    parse_features,
    resolve_provider_config,
)
from llm_behavior_lab.models import ModelSettings
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
    assert args.scoring_model_id is None


@pytest.mark.parametrize("command", ["scale-design", "task-design"])
def test_design_parser_accepts_provider_execution_policy(command: str) -> None:
    procedure_args = (
        ["--questionnaire", "bfi_10"]
        if command == "scale-design"
        else ["--task", "four-deck-card-task"]
    )
    args = build_parser().parse_args(
        [
            command,
            "--experiment-id",
            "pilot-study-one",
            *procedure_args,
            "--max-attempts",
            "5",
            "--initial-backoff",
            "0.5",
            "--max-backoff",
            "8",
            "--max-concurrency",
            "6",
        ]
    )

    assert args.max_attempts == 5
    assert args.initial_backoff == 0.5
    assert args.max_backoff == 8
    assert args.max_concurrency == 6


def test_scale_run_parser_accepts_explicit_resume_controls() -> None:
    args = build_parser().parse_args(
        [
            "scale-run",
            "--experiment-id",
            "pilot-study-one",
            "--run-id",
            "run-bfi10-test-20260603142709",
            "--retry-failed",
        ]
    )

    assert args.run_id == "run-bfi10-test-20260603142709"
    assert args.retry_failed is True


def test_parser_accepts_questionnaire_discovery_commands() -> None:
    list_args = build_parser().parse_args(["questionnaire-list", "--json"])
    describe_args = build_parser().parse_args(["questionnaire-describe", "bfi_10", "--json"])

    assert list_args.command == "questionnaire-list"
    assert list_args.json is True
    assert describe_args.command == "questionnaire-describe"
    assert describe_args.questionnaire_id == "bfi_10"
    assert describe_args.json is True


def test_parser_accepts_persona_discovery_and_preview_commands() -> None:
    fields = build_parser().parse_args(["persona-fields", "--json"])
    preview = build_parser().parse_args(
        [
            "persona-preview",
            "--experiment-id",
            "pilot-study-one",
            "--persona-count",
            "5",
            "--seed",
            "7",
            "--persona-field",
            "age",
            "--persona-field",
            "country",
            "--json",
        ]
    )

    assert fields.command == "persona-fields"
    assert fields.json is True
    assert preview.command == "persona-preview"
    assert preview.persona_count == 5
    assert preview.persona_fields == ["age", "country"]
    assert preview.json is True


def test_persona_fields_prints_json_without_creating_artifacts(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.chdir(tmp_path)

    result = main(["persona-fields", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert {entry["id"] for entry in payload} == set(RequestedDemographicField)
    assert next(entry for entry in payload if entry["id"] == "age")["supports_range"] is True
    assert list(tmp_path.iterdir()) == []


def test_persona_preview_prints_resolved_settings_without_writing(tmp_path: Path, capsys) -> None:
    result = main(
        [
            "persona-preview",
            "--project-root",
            str(tmp_path),
            "--experiment-id",
            "preview-study-one",
            "--persona-count",
            "3",
            "--seed",
            "11",
            "--persona-field",
            "age",
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["count"] == 3
    assert payload["seed"] == 11
    assert payload["requested_fields"] == ["age"]
    assert list(tmp_path.iterdir()) == []


def test_questionnaire_list_prints_human_readable_catalog(capsys) -> None:
    result = main(["questionnaire-list"])

    output = capsys.readouterr().out
    assert result == 0
    assert "bfi_10" in output
    assert "consumer_involvement" in output
    assert "purchase_decision_making_inventory" in output
    assert "target (required)" in output


def test_questionnaire_list_prints_json_without_creating_artifacts(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.chdir(tmp_path)

    result = main(["questionnaire-list", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert [entry["id"] for entry in payload] == [
        "bfi_10",
        "consumer_involvement",
        "purchase_decision_making_inventory",
    ]
    assert list(tmp_path.iterdir()) == []


def test_questionnaire_describe_prints_detailed_human_output(capsys) -> None:
    result = main(["questionnaire-describe", "consumer_involvement"])

    output = capsys.readouterr().out
    assert result == 0
    assert "Consumer Involvement Scale" in output
    assert "Items: 12" in output
    assert "Response formats: likert" in output
    assert "target (required)" in output
    assert "meal delivery services" in output


def test_questionnaire_describe_prints_json(capsys) -> None:
    result = main(["questionnaire-describe", "bfi_10", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["id"] == "bfi_10"
    assert payload["item_count"] == 10
    assert payload["scoring_model_ids"] == ["default"]
    assert payload["parameters"] == []


def test_questionnaire_describe_rejects_shorthand_alias() -> None:
    with pytest.raises(ValueError, match="unknown questionnaire_id: bfi10"):
        main(["questionnaire-describe", "bfi10"])


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


def test_scale_design_accepts_requested_persona_fields() -> None:
    args = build_parser().parse_args(
        [
            "scale-design",
            "--experiment-id",
            "pilot-study-one",
            "--questionnaire",
            "bfi_10",
            "--persona-field",
            "age",
            "--persona-field",
            "country",
        ]
    )

    assert args.persona_fields == ["age", "country"]


def test_cli_and_python_persona_creation_are_equivalent(tmp_path: Path, capsys) -> None:
    cli_root = tmp_path / "cli"
    python_root = tmp_path / "python"
    config_path = tmp_path / "persona-config.json"
    config_path.write_text(
        '{"field_values": {"country": "PL"}}',
        encoding="utf-8",
    )
    common = [
        "--experiment-id",
        "parity-study-one",
        "--persona-count",
        "3",
        "--seed",
        "7",
        "--persona-field",
        "age",
        "--persona-field",
        "country",
        "--persona-config",
        str(config_path),
    ]

    assert (
        main(
            [
                "scale-design",
                "--project-root",
                str(cli_root),
                "--questionnaire",
                "bfi_10",
                *common,
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert (
        main(
            [
                "personas",
                "--project-root",
                str(cli_root),
                "--experiment-id",
                "parity-study-one",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out

    python_batch = create_personas(
        python_root,
        "parity-study-one",
        PersonaDesign.model_validate(
            {
                "count": 3,
                "seed": 7,
                "requested_fields": ["age", "country"],
                "generation_config": {"field_values": {"country": "PL"}},
            }
        ),
    )

    assert load_personas(cli_root, "parity-study-one") == python_batch
    assert "Created 3 personas at" in output
    assert "personas.json" in output


def test_personas_cli_requires_replace_to_overwrite(tmp_path: Path, capsys) -> None:
    common = [
        "--project-root",
        str(tmp_path),
        "--experiment-id",
        "replace-study-one",
    ]
    assert main(["scale-design", *common, "--questionnaire", "bfi_10", "--seed", "7"]) == 0
    capsys.readouterr()
    assert main(["personas", *common]) == 0

    with pytest.raises(FileExistsError, match="personas already exist"):
        main(["personas", *common])

    assert main(["personas", *common, "--replace"]) == 0


def test_scale_design_rejects_invalid_persona_settings_before_writing(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "persona-config.json"
    config_path.write_text(
        '{"field_values": {"country": "PL"}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="configured fields must be requested"):
        main(
            [
                "scale-design",
                "--project-root",
                str(tmp_path),
                "--experiment-id",
                "invalid-study-one",
                "--questionnaire",
                "bfi_10",
                "--persona-field",
                "age",
                "--persona-config",
                str(config_path),
            ]
        )

    assert not (tmp_path / "experiments").exists()


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
    run = build_parser().parse_args(["task-run", "--experiment-id", "card-task-one"])
    analyze = build_parser().parse_args(
        ["task-analyze", "--experiment-id", "card-task-one"]
    )

    assert design.task == "four-deck-card-task"
    assert design.task_config == Path("task.json")
    assert not hasattr(run, "concurrency")
    assert analyze.command == "task-analyze"


@pytest.mark.anyio
async def test_run_with_cancellation_sets_event_from_sigint_callback(monkeypatch) -> None:
    callbacks: dict[int, Callable[[], None]] = {}

    class FakeLoop:
        def add_signal_handler(self, signal_number, callback) -> None:
            callbacks[signal_number] = callback

        def remove_signal_handler(self, signal_number) -> bool:
            callbacks.pop(signal_number)
            return True

    monkeypatch.setattr(asyncio, "get_running_loop", lambda: FakeLoop())

    async def operation(cancel_event: asyncio.Event) -> bool:
        next(iter(callbacks.values()))()
        return cancel_event.is_set()

    assert await _run_with_cancellation(operation) is True
    assert callbacks == {}


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


def test_scale_score_uses_scoring_model_from_design_by_default(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    design = ExperimentDesign(
        experiment_id="pilot-study-one",
        procedure=ScaleProcedureDesign(
            questionnaire_id="bfi_10",
            scoring_model_id="construct_high",
        ),
        personas=PersonaDesign(count=1),
        provider=ProviderDesign(model="test", base_url="http://localhost"),
    )
    create_experiment_design(tmp_path, design)
    run_root = tmp_path / "experiments" / "pilot-study-one" / "run-bfi10-test-1"
    run_root.mkdir()
    captured: dict[str, object] = {}

    def fake_score_run(path: Path, scoring_model_id: str | None):
        captured["path"] = path
        captured["scoring_model_id"] = scoring_model_id
        return SimpleNamespace(output_root=path / "scoring" / "construct_high-1.0")

    monkeypatch.setattr("llm_behavior_lab.main.score_run", fake_score_run)

    result = main(
        [
            "scale-score",
            "--project-root",
            str(tmp_path),
            "--experiment-id",
            "pilot-study-one",
        ]
    )

    assert result == 0
    assert captured == {
        "path": run_root,
        "scoring_model_id": "construct_high",
    }
    assert "construct_high-1.0" in capsys.readouterr().out


def test_scale_run_dispatches_async_with_persisted_execution_policy(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    persona_design = PersonaDesign(count=1, seed=7)
    design = ExperimentDesign(
        experiment_id="pilot-study-one",
        procedure=ScaleProcedureDesign(questionnaire_id="bfi_10"),
        personas=persona_design,
        provider=ProviderDesign(
            model="test",
            base_url="http://localhost",
            max_attempts=5,
            initial_backoff_seconds=0.5,
            max_backoff_seconds=8,
            max_concurrency=6,
        ),
    )
    create_experiment_design(tmp_path, design)
    create_personas(tmp_path, design.experiment_id, persona_design)
    captured: dict[str, object] = {}

    async def fake_run(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            runs=[SimpleNamespace(run_id="run-bfi10-test-20260603142709")]
        )

    monkeypatch.setattr(
        "llm_behavior_lab.main.run_persisted_persona_batch_async",
        fake_run,
    )
    monkeypatch.setattr(
        "llm_behavior_lab.main.AsyncOpenAiChatClient",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    result = main(
        [
            "scale-run",
            "--project-root",
            str(tmp_path),
            "--experiment-id",
            design.experiment_id,
            "--run-id",
            "run-bfi10-test-20260603142709",
            "--retry-failed",
        ]
    )

    settings = captured["settings"]
    assert isinstance(settings, ModelSettings)
    assert settings.max_attempts == 5
    assert settings.initial_backoff_seconds == 0.5
    assert settings.max_backoff_seconds == 8
    assert settings.max_concurrency == 6
    assert captured["run_id"] == "run-bfi10-test-20260603142709"
    assert captured["retry_failed"] is True
    assert isinstance(captured["cancel_event"], asyncio.Event)
    assert result == 0
    assert "run-bfi10-test-20260603142709" in capsys.readouterr().out


def test_scale_score_cli_override_takes_precedence_over_design(
    tmp_path: Path, monkeypatch
) -> None:
    design = ExperimentDesign(
        experiment_id="pilot-study-two",
        procedure=ScaleProcedureDesign(
            questionnaire_id="bfi_10",
            scoring_model_id="design-model",
        ),
        personas=PersonaDesign(count=1),
        provider=ProviderDesign(model="test", base_url="http://localhost"),
    )
    create_experiment_design(tmp_path, design)
    run_root = tmp_path / "experiments" / "pilot-study-two" / "run-bfi10-test-1"
    run_root.mkdir()
    captured: dict[str, object] = {}

    def fake_score_run(path: Path, scoring_model_id: str | None):
        captured["scoring_model_id"] = scoring_model_id
        return SimpleNamespace(output_root=path / "scoring" / "override-1.0")

    monkeypatch.setattr("llm_behavior_lab.main.score_run", fake_score_run)

    main(
        [
            "scale-score",
            "--project-root",
            str(tmp_path),
            "--experiment-id",
            "pilot-study-two",
            "--scoring-model-id",
            "override",
        ]
    )

    assert captured["scoring_model_id"] == "override"


def test_scale_design_rejects_unknown_scoring_model(tmp_path: Path) -> None:
    with pytest.raises(
        ValueError,
        match="unknown scoring model 'missing' for bfi_10",
    ):
        main(
            [
                "scale-design",
                "--project-root",
                str(tmp_path),
                "--experiment-id",
                "pilot-study-three",
                "--questionnaire",
                "bfi_10",
                "--scoring-model-id",
                "missing",
            ]
        )
