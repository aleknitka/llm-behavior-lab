from collections.abc import Sequence
from pathlib import Path

from llm_behavior_lab.questionnaires.base.scale import Questionnaire
from llm_behavior_lab.responses.base import ItemResponseRecord, ResponseStatus


def load_item_ledger(path: Path) -> list[ItemResponseRecord]:
    if not path.exists():
        return []
    return [
        ItemResponseRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def latest_item_attempts(
    records: Sequence[ItemResponseRecord],
) -> list[ItemResponseRecord]:
    latest = {(record.subject_id, record.item_id): record for record in records}
    return sorted(
        latest.values(),
        key=lambda record: (record.subject_id, record.item_order),
    )


def pending_item_ids(
    questionnaire: Questionnaire,
    records: Sequence[ItemResponseRecord],
    *,
    retry_failed: bool,
) -> list[str]:
    latest = {record.item_id: record for record in latest_item_attempts(records)}
    retryable = {ResponseStatus.FAILED, ResponseStatus.INVALID}
    return [
        item.id
        for item in questionnaire.items
        if item.id not in latest
        or (retry_failed and latest[item.id].status in retryable)
    ]


def validate_item_ledger(
    questionnaire: Questionnaire,
    subject_id: str,
    run_id: str,
    records: Sequence[ItemResponseRecord],
) -> None:
    items = {item.id: item for item in questionnaire.items}
    for record in records:
        if record.subject_id != subject_id:
            raise ValueError("response ledger subject does not match resume request")
        if record.run_id != run_id:
            raise ValueError("response ledger run does not match resume request")
        if (
            record.questionnaire_id != questionnaire.id
            or record.questionnaire_version != questionnaire.version
        ):
            raise ValueError("response ledger questionnaire does not match snapshot")
        item = items.get(record.item_id)
        if item is None:
            raise ValueError(f"response ledger contains unknown item {record.item_id!r}")
        if record.item_order != item.order:
            raise ValueError(f"response ledger item order differs for {item.id!r}")
        if record.item_text != item.text:
            raise ValueError(f"response ledger item text differs for {item.id!r}")
        if record.response_format_type != str(item.response_format.type):
            raise ValueError(f"response ledger response format differs for {item.id!r}")
