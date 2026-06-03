from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from llm_psych_scales.responses.base import (
    ItemResponseRecord,
    MultipleChoiceAnswerValue,
    NumericAnswerValue,
    SingleChoiceAnswerValue,
    TextAnswerValue,
)
from llm_psych_scales.responses.base.values import LikertAnswerValue

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
]


def load_response_table(path: Path) -> list[dict[str, Any]]:
    rows = [_row_from_record(record) for record in _load_records(path)]
    return sorted(rows, key=lambda row: (str(row["subject_id"]), int(row["item_order"])))


def write_response_table_csv(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=TABLE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_records(path: Path) -> list[ItemResponseRecord]:
    if path.is_file():
        return _load_jsonl_file(path)

    responses_root = path / "responses" if (path / "responses").is_dir() else path
    records: list[ItemResponseRecord] = []
    for response_path in sorted(responses_root.glob("*.jsonl")):
        records.extend(_load_jsonl_file(response_path))
    return records


def _load_jsonl_file(path: Path) -> list[ItemResponseRecord]:
    records: list[ItemResponseRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(ItemResponseRecord.model_validate(json.loads(line)))
    return records


def _row_from_record(record: ItemResponseRecord) -> dict[str, Any]:
    answer_value, answer_label = _answer_columns(record)
    return {
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
