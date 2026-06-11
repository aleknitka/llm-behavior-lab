import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from llm_behavior_lab.experiments import PersonaDesign
from llm_behavior_lab.main import build_parser, main
from llm_behavior_lab.models import LlmQuestionResult, ModelSettings
from llm_behavior_lab.protocol_runs import (
    create_protocol_experiment,
    create_protocol_run,
    load_protocol_experiment,
)
from llm_behavior_lab.protocols import (
    ProtocolQuestionnaireStep,
    ProtocolTaskStep,
    UnifiedExperimentProtocol,
    load_compatible_protocol,
    protocol_fingerprint,
)


class RecordingClient:
    calls: list[list[dict[str, str]]] = []

    def __init__(self, api_key: str, base_url: str) -> None:
        self.api_key = api_key
        self.base_url = base_url

    def complete(
        self,
        messages: Sequence[dict[str, str]],
        settings: ModelSettings,
        allowed_answer_ids: Sequence[str],
    ) -> LlmQuestionResult:
        self.calls.append(list(messages))
        selected = allowed_answer_ids[0]
        return LlmQuestionResult(selected_answer_id=selected, raw_response=selected)


def _protocol(**updates: object) -> UnifiedExperimentProtocol:
    payload: dict[str, object] = {
        "version": "1.0",
        "experiment_id": "unified-study-one",
        "name": "Unified study",
        "persona_seed": 11,
        "run_seed": 21,
        "personas": {
            "count": 1,
            "requested_fields": ["age", "country"],
            "generation_config": {"field_values": {"country": "PL"}},
        },
        "provider": {
            "model": "test-model",
            "base_url": "http://localhost:1234/v1",
            "temperature": 0,
            "timeout_seconds": 10,
        },
        "steps": [
            {
                "id": "personality",
                "kind": "questionnaire",
                "questionnaire_id": "bfi_10",
                "scoring_model_id": "default",
                "history": "reset",
            },
            {
                "id": "decision-task",
                "kind": "task",
                "task_id": "four-deck-card-task",
                "task_config": {"trial_count": 1},
                "history": "inherit",
            },
        ],
    }
    payload.update(updates)
    return UnifiedExperimentProtocol.model_validate(payload)


def test_unified_protocol_round_trips_ordered_mixed_steps() -> None:
    protocol = _protocol()

    loaded = UnifiedExperimentProtocol.model_validate_json(protocol.model_dump_json())

    assert loaded == protocol
    assert isinstance(loaded.steps[0], ProtocolQuestionnaireStep)
    assert isinstance(loaded.steps[1], ProtocolTaskStep)
    assert [step.id for step in loaded.steps] == ["personality", "decision-task"]


def test_protocol_provider_round_trips_execution_policy() -> None:
    protocol = _protocol(
        provider={
            "model": "test-model",
            "base_url": "http://localhost:1234/v1",
            "max_attempts": 5,
            "initial_backoff_seconds": 0.5,
            "max_backoff_seconds": 8,
            "max_concurrency": 6,
        }
    )

    loaded = UnifiedExperimentProtocol.model_validate_json(protocol.model_dump_json())

    assert loaded.provider.max_attempts == 5
    assert loaded.provider.initial_backoff_seconds == 0.5
    assert loaded.provider.max_backoff_seconds == 8
    assert loaded.provider.max_concurrency == 6


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_attempts", 0),
        ("initial_backoff_seconds", -0.1),
        ("max_backoff_seconds", -0.1),
        ("max_concurrency", 0),
    ],
)
def test_protocol_provider_rejects_invalid_execution_policy(
    field: str,
    value: int | float,
) -> None:
    provider = {
        "model": "test-model",
        "base_url": "http://localhost:1234/v1",
        field: value,
    }

    with pytest.raises(ValueError):
        _protocol(provider=provider)


def test_protocol_identity_excludes_only_seed_defaults() -> None:
    original = _protocol()
    different_seeds = _protocol(persona_seed=99, run_seed=101)
    changed_provider = _protocol(
        provider={
            **original.provider.model_dump(mode="json"),
            "temperature": 0.5,
        }
    )

    assert protocol_fingerprint(original) == protocol_fingerprint(different_seeds)
    assert protocol_fingerprint(original) != protocol_fingerprint(changed_provider)


def test_create_protocol_experiment_writes_immutable_protocol_and_first_cohort(
    tmp_path: Path,
) -> None:
    created = create_protocol_experiment(tmp_path, _protocol())

    experiment_root = tmp_path / "experiments" / "unified-study-one"
    assert created.protocol_path == experiment_root / "protocol.json"
    assert created.cohort_id.startswith("cohort-")
    assert (experiment_root / "cohorts" / created.cohort_id / "personas.json").exists()
    assert (
        experiment_root / "cohorts" / created.cohort_id / "protocol-assignments.json"
    ).exists()
    metadata = json.loads(
        (experiment_root / "cohorts" / created.cohort_id / "metadata.json").read_text()
    )
    assert metadata["persona_seed"] == 11
    assert metadata["protocol_fingerprint"] == protocol_fingerprint(_protocol())


def test_same_protocol_reuses_exact_cohort_or_creates_new_seeded_cohort(
    tmp_path: Path,
) -> None:
    created = create_protocol_experiment(tmp_path, _protocol())

    reused = create_protocol_run(
        tmp_path,
        _protocol(run_seed=22),
        cohort_id=created.cohort_id,
        execute=False,
    )
    generated = create_protocol_run(
        tmp_path,
        _protocol(persona_seed=12, run_seed=23),
        persona_seed=12,
        execute=False,
    )

    assert reused.cohort_id == created.cohort_id
    assert generated.cohort_id != created.cohort_id
    cohorts = list((tmp_path / "experiments" / "unified-study-one" / "cohorts").iterdir())
    assert len(cohorts) == 2


def test_explicit_cohort_reuse_records_the_cohorts_actual_persona_seed(
    tmp_path: Path,
) -> None:
    create_protocol_experiment(tmp_path, _protocol())
    generated = create_protocol_run(
        tmp_path,
        _protocol(),
        persona_seed=12,
        execute=False,
    )

    reused = create_protocol_run(
        tmp_path,
        _protocol(),
        cohort_id=generated.cohort_id,
        execute=False,
    )

    row = json.loads((reused.run_root / "run.json").read_text())
    assert row["metadata"]["persona_seed"] == 12


def test_changed_non_seed_protocol_requires_new_experiment_id(tmp_path: Path) -> None:
    create_protocol_experiment(tmp_path, _protocol())
    changed = _protocol(
        steps=[
            {
                "id": "personality",
                "kind": "questionnaire",
                "questionnaire_id": "purchase_decision_making_inventory",
            }
        ]
    )

    with pytest.raises(ValueError, match="new experiment_id"):
        create_protocol_run(tmp_path, changed, execute=False)


def test_protocol_run_records_effective_seeds_cohort_and_step_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    RecordingClient.calls = []
    monkeypatch.setattr("llm_behavior_lab.protocol_runs.OpenAiChatClient", RecordingClient)
    created = create_protocol_experiment(tmp_path, _protocol())

    result = create_protocol_run(
        tmp_path,
        _protocol(),
        cohort_id=created.cohort_id,
        api_key="test-key",
    )

    row = json.loads((result.run_root / "run.json").read_text())
    assert row["metadata"]["protocol_fingerprint"] == protocol_fingerprint(_protocol())
    assert row["metadata"]["cohort_id"] == created.cohort_id
    assert row["metadata"]["persona_seed"] == 11
    assert row["metadata"]["run_seed"] == 21
    assert [step["step_id"] for step in row["metadata"]["step_results"]] == [
        "personality",
        "decision-task",
    ]
    assert (result.run_root / "steps" / "personality" / "scale.json").exists()
    assert (result.run_root / "steps" / "decision-task" / "task.json").exists()


def test_inherit_step_receives_prior_questionnaire_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    RecordingClient.calls = []
    monkeypatch.setattr("llm_behavior_lab.protocol_runs.OpenAiChatClient", RecordingClient)
    protocol = _protocol()
    created = create_protocol_experiment(tmp_path, protocol)

    create_protocol_run(
        tmp_path,
        protocol,
        cohort_id=created.cohort_id,
        api_key="test-key",
    )

    task_call = RecordingClient.calls[-1]
    assert any("I see myself as someone who" in message["content"] for message in task_call)
    assert sum(message["role"] == "system" for message in task_call) == 1


def test_cli_requires_new_run_for_existing_protocol_in_non_interactive_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "study.json"
    path.write_text(_protocol().model_dump_json(indent=2))
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    assert main(["protocol-create", "--project-root", str(tmp_path), "--file", str(path)]) == 0
    with pytest.raises(ValueError, match="--new-run"):
        main(["protocol-create", "--project-root", str(tmp_path), "--file", str(path)])


def test_interactive_rerun_prompts_for_optional_run_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "study.json"
    path.write_text(_protocol().model_dump_json(indent=2))
    monkeypatch.setattr("llm_behavior_lab.protocol_runs.OpenAiChatClient", RecordingClient)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    answers = iter(["yes", "yes", "33"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    assert main(["protocol-create", "--project-root", str(tmp_path), "--file", str(path)]) == 0

    assert main(["protocol-create", "--project-root", str(tmp_path), "--file", str(path)]) == 0

    experiment_root = tmp_path / "experiments" / "unified-study-one"
    run_root = next(experiment_root.glob("run-protocol-*"))
    row = json.loads((run_root / "run.json").read_text())
    assert row["metadata"]["run_seed"] == 33


def test_cli_rejects_cohort_id_with_persona_seed() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "protocol-create",
                "--file",
                "study.json",
                "--new-run",
                "--cohort-id",
                "cohort-00000000-0000-4000-8000-000000000000",
                "--persona-seed",
                "12",
            ]
        )


def test_analysis_commands_accept_protocol_step_id() -> None:
    parser = build_parser()

    score = parser.parse_args(
        [
            "scale-score",
            "--experiment-id",
            "unified-study-one",
            "--run-id",
            "run-protocol-test-model-20260610120000",
            "--step-id",
            "personality",
        ]
    )
    analyze = parser.parse_args(
        [
            "task-analyze",
            "--experiment-id",
            "unified-study-one",
            "--run-id",
            "run-protocol-test-model-20260610120000",
            "--step-id",
            "decision-task",
        ]
    )

    assert score.step_id == "personality"
    assert analyze.step_id == "decision-task"


def test_legacy_design_and_factor_protocol_remain_loadable(tmp_path: Path) -> None:
    experiment_root = tmp_path / "experiments" / "legacy-study-one"
    experiment_root.mkdir(parents=True)
    (experiment_root / "design.json").write_text(
        json.dumps(
            {
                "version": "2.0",
                "experiment_id": "legacy-study-one",
                "procedure": {"kind": "scale", "questionnaire_id": "bfi_10"},
                "personas": PersonaDesign(count=1, seed=7).model_dump(mode="json"),
                "provider": {
                    "model": "test-model",
                    "base_url": "http://localhost:1234/v1",
                },
            }
        )
    )

    loaded = load_protocol_experiment(tmp_path, "legacy-study-one")
    factor = load_compatible_protocol(
        Path("examples/ollama-bfi10-factorial/protocol.json")
    )

    assert loaded.source == "design.json"
    assert loaded.protocol.experiment_id == "legacy-study-one"
    assert factor.source == "factor_protocol"
