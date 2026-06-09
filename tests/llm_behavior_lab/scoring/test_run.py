import json
from pathlib import Path

from llm_behavior_lab.questionnaires.bfi10 import BFI_10
from llm_behavior_lab.responses.base import (
    ItemResponseRecord,
    LikertAnswerValue,
    ResponseStatus,
)
from llm_behavior_lab.scoring import score_run


def test_score_run_writes_scores_reliability_and_manifest(tmp_path: Path) -> None:
    run_root = tmp_path / "run-bfi10-model-20260608120000"
    responses_root = run_root / "responses"
    responses_root.mkdir(parents=True)
    (run_root / "scale.json").write_text(BFI_10.model_dump_json(indent=2))
    for subject_index, value in enumerate([1, 2, 4, 5], start=1):
        records = []
        for item in BFI_10.items:
            records.append(
                ItemResponseRecord(
                    subject_id=f"subject-{subject_index}",
                    session_id="session-1",
                    run_id=run_root.name,
                    questionnaire_id=BFI_10.id,
                    questionnaire_version=BFI_10.version,
                    item_id=item.id,
                    item_order=item.order,
                    item_text=item.text,
                    response_format_type="likert",
                    messages=[],
                    answer=LikertAnswerValue(value=value),
                    raw_response=str(value),
                    status=ResponseStatus.COMPLETED,
                    metadata={"experiment_id": "pilot-study-one"},
                )
            )
        (responses_root / f"subject-{subject_index}.jsonl").write_text(
            "".join(record.model_dump_json() + "\n" for record in records)
        )

    result = score_run(run_root)

    assert len(result.scores) == 20
    assert result.output_root.name == "default-1.0"
    assert (result.output_root / "scores.jsonl").exists()
    assert (result.output_root / "reliability.jsonl").exists()
    manifest = json.loads((result.output_root / "scoring.json").read_text())
    assert manifest["questionnaire_id"] == "bfi_10"
    assert manifest["scoring_model_id"] == "default"


def test_score_run_uses_validated_current_definition_for_legacy_snapshot(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run-bfi10-model-20260608120000"
    responses_root = run_root / "responses"
    responses_root.mkdir(parents=True)
    legacy = BFI_10.model_copy(update={"scoring_models": []})
    (run_root / "scale.json").write_text(legacy.model_dump_json(indent=2))
    records = [
        ItemResponseRecord(
            subject_id="subject-1",
            session_id="session-1",
            run_id=run_root.name,
            questionnaire_id=BFI_10.id,
            questionnaire_version=BFI_10.version,
            item_id=item.id,
            item_order=item.order,
            item_text=item.text,
            response_format_type="likert",
            messages=[],
            answer=LikertAnswerValue(value=3),
            raw_response="3",
            status=ResponseStatus.COMPLETED,
            metadata={},
        )
        for item in BFI_10.items
    ]
    (responses_root / "subject-1.jsonl").write_text(
        "".join(record.model_dump_json() + "\n" for record in records)
    )

    result = score_run(run_root)

    manifest = json.loads((result.output_root / "scoring.json").read_text())
    assert manifest["used_current_definition_fallback"] is True
    assert manifest["snapshot_definition_digest"] != manifest["scoring_definition_digest"]
