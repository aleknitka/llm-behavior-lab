from pathlib import Path

import pytest

from llm_behavior_lab.questionnaires.base.response_formats import LikertFormat
from llm_behavior_lab.questionnaires.base.scale import Item, Questionnaire, Section
from llm_behavior_lab.responses.base import (
    ChatMessage,
    ItemResponseRecord,
    ResponseStatus,
)
from llm_behavior_lab.responses.item_ledgers import (
    latest_item_attempts,
    load_item_ledger,
    pending_item_ids,
    validate_item_ledger,
)


def _questionnaire() -> Questionnaire:
    response_format = LikertFormat(min_value=1, max_value=5)
    return Questionnaire(
        id="example",
        shorthand="exam",
        name="Example",
        version="1.0",
        sections=[Section(id="main", item_ids=["item-1", "item-2", "item-3"])],
        items=[
            Item(id="item-1", order=1, text="First", response_format=response_format),
            Item(id="item-2", order=2, text="Second", response_format=response_format),
            Item(id="item-3", order=3, text="Third", response_format=response_format),
        ],
        reference="Reference",
        licence="Licence",
    )


def _response(
    item_id: str,
    status: ResponseStatus,
    *,
    item_order: int,
    item_text: str | None = None,
    subject_id: str = "subject-1",
    run_id: str = "run-1",
) -> ItemResponseRecord:
    return ItemResponseRecord(
        subject_id=subject_id,
        session_id="session-1",
        run_id=run_id,
        questionnaire_id="example",
        questionnaire_version="1.0",
        item_id=item_id,
        item_order=item_order,
        item_text=item_text or ["First", "Second", "Third"][item_order - 1],
        response_format_type="likert",
        messages=[ChatMessage(role="user", content="Question")],
        status=status,
    )


def test_latest_item_attempts_replace_prior_attempt_without_mutating_input() -> None:
    first = _response("item-1", ResponseStatus.FAILED, item_order=1)
    second = _response("item-2", ResponseStatus.COMPLETED, item_order=2)
    retried = _response("item-1", ResponseStatus.COMPLETED, item_order=1)
    records = [first, second, retried]

    assert latest_item_attempts(records) == [retried, second]
    assert records == [first, second, retried]


@pytest.mark.parametrize("status", [ResponseStatus.FAILED, ResponseStatus.INVALID])
def test_pending_item_ids_only_retry_unsuccessful_when_requested(
    status: ResponseStatus,
) -> None:
    records = [
        _response("item-1", ResponseStatus.COMPLETED, item_order=1),
        _response("item-2", status, item_order=2),
    ]

    assert pending_item_ids(_questionnaire(), records, retry_failed=False) == ["item-3"]
    assert pending_item_ids(_questionnaire(), records, retry_failed=True) == [
        "item-2",
        "item-3",
    ]


def test_load_item_ledger_returns_empty_for_missing_path(tmp_path: Path) -> None:
    assert load_item_ledger(tmp_path / "missing.jsonl") == []


def test_load_item_ledger_preserves_append_order(tmp_path: Path) -> None:
    path = tmp_path / "subject-1.jsonl"
    records = [
        _response("item-1", ResponseStatus.FAILED, item_order=1),
        _response("item-1", ResponseStatus.COMPLETED, item_order=1),
    ]
    path.write_text(
        "\n".join(record.model_dump_json() for record in records) + "\n",
        encoding="utf-8",
    )

    assert load_item_ledger(path) == records


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"subject_id": "subject-2"}, "subject"),
        ({"run_id": "run-2"}, "run"),
        ({"questionnaire_id": "changed"}, "questionnaire"),
        ({"questionnaire_version": "2.0"}, "questionnaire"),
        ({"item_id": "unknown"}, "unknown item"),
        ({"item_order": 9}, "item order"),
        ({"item_text": "Changed"}, "item text"),
        ({"response_format_type": "text"}, "response format"),
    ],
)
def test_validate_item_ledger_rejects_incompatible_records(
    update: dict[str, object],
    message: str,
) -> None:
    record = _response("item-1", ResponseStatus.COMPLETED, item_order=1).model_copy(
        update=update
    )

    with pytest.raises(ValueError, match=message):
        validate_item_ledger(_questionnaire(), "subject-1", "run-1", [record])


def test_validate_item_ledger_accepts_matching_records() -> None:
    records = [
        _response("item-1", ResponseStatus.COMPLETED, item_order=1),
        _response("item-2", ResponseStatus.FAILED, item_order=2),
    ]

    validate_item_ledger(_questionnaire(), "subject-1", "run-1", records)
