import json
from datetime import UTC, datetime

import pytest

from llm_behavior_lab.responses.base import ItemResponseRecord, LikertAnswerValue, ResponseStatus
from llm_behavior_lab.storage import (
    ExperimentPaths,
    append_jsonl_record,
    build_run_directory_name,
    generate_experiment_id,
    load_json_document,
    resolve_compatible_snapshot_path,
    resolve_experiment_paths,
    slugify_model_name,
    validate_experiment_id,
    write_json_document,
)


def test_append_jsonl_record_creates_parent_and_writes_line(tmp_path) -> None:
    path = tmp_path / "nested" / "records.jsonl"
    record = ItemResponseRecord(
        subject_id="subject-1",
        session_id="session-00000000-0000-4000-8000-000000000002",
        run_id="run-00000000-0000-4000-8000-000000000001",
        questionnaire_id="bfi_10",
        questionnaire_version="1.0",
        item_id="bfi10_01_reserved",
        item_order=1,
        item_text="Question",
        response_format_type="likert",
        messages=[],
        answer=LikertAnswerValue(value=1, label="Strongly agree"),
        raw_response="Strongly disagree",
        structured_response={"selected_answer_id": "1"},
        logprobs=None,
        status=ResponseStatus.COMPLETED,
        error=None,
    )

    append_jsonl_record(path, record)

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["answer"]["value"] == 1


def test_write_json_document_atomically_replaces_validated_snapshot(tmp_path) -> None:
    path = tmp_path / "nested" / "record.json"
    first = LikertAnswerValue(value=1, label="Low")
    second = LikertAnswerValue(value=5, label="High")

    write_json_document(path, first)
    write_json_document(path, second)

    assert load_json_document(path, LikertAnswerValue) == second
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []


def test_write_json_document_preserves_existing_snapshot_when_replace_fails(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "record.json"
    first = LikertAnswerValue(value=1, label="Low")
    write_json_document(path, first)

    def fail_replace(_source, _target) -> None:
        raise OSError("interrupted replace")

    monkeypatch.setattr("llm_behavior_lab.storage.replace", fail_replace)

    with pytest.raises(OSError, match="interrupted replace"):
        write_json_document(path, LikertAnswerValue(value=5, label="High"))

    assert load_json_document(path, LikertAnswerValue) == first
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []


def test_resolve_compatible_snapshot_path_uses_legacy_file(tmp_path) -> None:
    normalized = tmp_path / "personas.json"
    legacy = tmp_path / "personas.jsonl"
    legacy.write_text('{"experiment_id":"pilot-study-one"}\n', encoding="utf-8")

    assert resolve_compatible_snapshot_path(normalized, legacy) == legacy


def test_resolve_compatible_snapshot_path_rejects_conflicting_files(tmp_path) -> None:
    normalized = tmp_path / "personas.json"
    legacy = tmp_path / "personas.jsonl"
    normalized.write_text('{"experiment_id":"pilot-study-one"}\n', encoding="utf-8")
    legacy.write_text('{"experiment_id":"other-study-one"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="conflicting canonical snapshot files"):
        resolve_compatible_snapshot_path(normalized, legacy)


def test_generate_experiment_id_uses_three_hyphenated_items() -> None:
    experiment_id = generate_experiment_id(seed=31)

    assert experiment_id.count("-") == 2
    assert validate_experiment_id(experiment_id) == experiment_id


@pytest.mark.parametrize(
    "experiment_id",
    [
        "one",
        "two-items",
        "four-items-are-invalid",
        "has_underscore_item",
        "has.dot.item",
        "has space item",
        "../escape",
        "special-@-item",
    ],
)
def test_validate_experiment_id_rejects_invalid_names(experiment_id: str) -> None:
    with pytest.raises(ValueError):
        validate_experiment_id(experiment_id)


def test_slugify_model_name_replaces_separators_with_hyphens() -> None:
    assert slugify_model_name("openai/gpt-oss-20b") == "openai-gpt-oss-20b"
    assert slugify_model_name("LM Studio Model 1") == "lm-studio-model-1"


@pytest.mark.parametrize("model", ["", "///", "..."])
def test_slugify_model_name_rejects_empty_slugs(model: str) -> None:
    with pytest.raises(ValueError):
        slugify_model_name(model)


def test_build_run_directory_name_uses_scale_model_and_timestamp() -> None:
    started_at = datetime(2026, 6, 3, 14, 27, 9, tzinfo=UTC)

    run_id = build_run_directory_name(
        questionnaire_shorthand="bfi10",
        model="openai/gpt-oss-20b",
        started_at=started_at,
    )

    assert run_id == "run-bfi10-openai-gpt-oss-20b-20260603142709"


def test_resolve_experiment_paths_uses_todo_hierarchy(tmp_path) -> None:
    paths = resolve_experiment_paths(
        project_root=tmp_path,
        experiment_id="pilot-study-one",
        run_id="run-bfi10-openai-gpt-oss-20b-20260603142709",
    )

    assert isinstance(paths, ExperimentPaths)
    assert paths.experiment_root == tmp_path / "experiments" / "pilot-study-one"
    assert paths.personas_path == paths.experiment_root / "personas.json"
    assert paths.base_personas_path == paths.experiment_root / "base_personas.json"
    assert paths.protocol_assignments_path == (
        paths.experiment_root / "protocol_assignments.json"
    )
    assert paths.metadata_path == paths.experiment_root / "metadata.json"
    assert paths.run_root == (
        paths.experiment_root / "run-bfi10-openai-gpt-oss-20b-20260603142709"
    )
    assert paths.run_path == paths.run_root / "run.json"
    assert paths.responses_root == paths.run_root / "responses"
    assert paths.response_path_for_subject("subject-1") == (
        paths.responses_root / "subject-1.jsonl"
    )
    assert paths.scale_path == paths.run_root / "scale.json"
