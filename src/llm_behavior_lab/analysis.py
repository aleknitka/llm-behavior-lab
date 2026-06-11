from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from llm_behavior_lab.personas.factory import PersonaBatch
from llm_behavior_lab.responses.base import (
    ItemResponseRecord,
    MultipleChoiceAnswerValue,
    NumericAnswerValue,
    SingleChoiceAnswerValue,
    TextAnswerValue,
)
from llm_behavior_lab.responses.base.values import LikertAnswerValue
from llm_behavior_lab.responses.item_ledgers import latest_item_attempts
from llm_behavior_lab.storage import (
    load_json_document,
    resolve_compatible_snapshot_path,
)

TABLE_COLUMNS = [
    "subject_id",
    "session_id",
    "run_id",
    "questionnaire_id",
    "questionnaire_version",
    "item_id",
    "item_order",
    "item_text",
    "response_format_type",
    "answer_type",
    "answer_value",
    "answer_label",
    "raw_response",
    "status",
    "error",
    "logprobs_available",
    "experiment_id",
    "protocol_name",
    "base_subject_id",
    "condition_id",
    "iteration_index",
]


def load_response_table(path: Path) -> list[dict[str, Any]]:
    rows = [_row_from_record(record) for record in _load_records(path)]
    personas = _load_personas_for_responses(path)
    for row in rows:
        features = personas.get(str(row["subject_id"]))
        if features is not None:
            row.update({f"persona_{key}": value for key, value in features.items()})
    return sorted(rows, key=lambda row: (str(row["subject_id"]), int(row["item_order"])))


def write_response_table_csv(rows: Iterable[dict[str, Any]], path: Path) -> None:
    materialized = list(rows)
    extra_columns = sorted(
        {key for row in materialized for key in row if key not in TABLE_COLUMNS}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[*TABLE_COLUMNS, *extra_columns])
        writer.writeheader()
        writer.writerows(materialized)


def _load_records(path: Path) -> list[ItemResponseRecord]:
    if path.is_file():
        return latest_item_attempts(_load_jsonl_file(path))

    responses_root = path / "responses" if (path / "responses").is_dir() else path
    records: list[ItemResponseRecord] = []
    for response_path in sorted(responses_root.glob("*.jsonl")):
        records.extend(_load_jsonl_file(response_path))
    return latest_item_attempts(records)


def _load_jsonl_file(path: Path) -> list[ItemResponseRecord]:
    records: list[ItemResponseRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(ItemResponseRecord.model_validate(json.loads(line)))
    return records


def _load_personas_for_responses(path: Path) -> dict[str, dict[str, Any]]:
    run_root = _run_root_from_response_path(path)
    if run_root is None:
        return {}
    canonical_run_root = _canonical_run_root(run_root)
    experiment_root = canonical_run_root.parent
    cohort_id = _cohort_id_from_run(canonical_run_root)
    snapshot_root = (
        experiment_root / "cohorts" / cohort_id
        if cohort_id is not None
        else experiment_root
    )
    personas_path = resolve_compatible_snapshot_path(
        snapshot_root / "personas.json",
        snapshot_root / "personas.jsonl",
    )
    if not personas_path.exists():
        return {}
    batch = load_json_document(personas_path, PersonaBatch)
    return {
        str(persona.subject_id): persona.features.model_dump(
            mode="json",
            exclude_none=True,
        )
        for persona in batch.personas
    }


def _run_root_from_response_path(path: Path) -> Path | None:
    if path.is_file():
        return path.parent.parent if path.parent.name == "responses" else None
    if path.name == "responses":
        return path.parent
    if (path / "responses").is_dir():
        return path
    return None


def _cohort_id_from_run(run_root: Path) -> str | None:
    run_path = run_root / "run.json"
    if not run_path.exists():
        return None
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return None
    cohort_id = metadata.get("cohort_id")
    return cohort_id if isinstance(cohort_id, str) else None


def _canonical_run_root(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if (candidate.parent / "protocol.json").exists() or (
            candidate.parent / "design.json"
        ).exists():
            return candidate
    return path


def _row_from_record(record: ItemResponseRecord) -> dict[str, Any]:
    answer_value, answer_label = _answer_columns(record)
    row = {
        "subject_id": record.subject_id,
        "session_id": record.session_id,
        "run_id": record.run_id,
        "questionnaire_id": record.questionnaire_id,
        "questionnaire_version": record.questionnaire_version,
        "item_id": record.item_id,
        "item_order": record.item_order,
        "item_text": record.item_text,
        "response_format_type": record.response_format_type,
        "answer_type": record.answer.type if record.answer is not None else None,
        "answer_value": answer_value,
        "answer_label": answer_label,
        "raw_response": record.raw_response,
        "status": str(record.status),
        "error": record.error,
        "logprobs_available": record.logprobs is not None,
        "experiment_id": record.metadata.get("experiment_id"),
    }
    if "protocol_name" in record.metadata:
        row["protocol_name"] = record.metadata.get("protocol_name")
        row["base_subject_id"] = record.metadata.get("base_subject_id")
        row["condition_id"] = record.metadata.get("condition_id")
        row["iteration_index"] = record.metadata.get("iteration_index")
    factor_values = record.metadata.get("factor_values")
    if isinstance(factor_values, dict):
        for field, value in factor_values.items():
            row[f"factor_{field}"] = value
    return row


def _answer_columns(record: ItemResponseRecord) -> tuple[Any, str | None]:
    answer = record.answer
    if answer is None:
        return None, None
    if isinstance(answer, LikertAnswerValue | NumericAnswerValue):
        return answer.value, getattr(answer, "label", None)
    if isinstance(answer, SingleChoiceAnswerValue):
        return answer.option_id, answer.label
    if isinstance(answer, MultipleChoiceAnswerValue):
        return ",".join(answer.option_ids), ",".join(answer.labels)
    if isinstance(answer, TextAnswerValue):
        return answer.text, None
    return None, None
