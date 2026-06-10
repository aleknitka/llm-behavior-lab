import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from llm_behavior_lab.questionnaires.base.scale import Questionnaire
from llm_behavior_lab.questionnaires.catalog import (
    describe_questionnaire,
    resolve_questionnaire,
)
from llm_behavior_lab.responses.base import ItemResponseRecord
from llm_behavior_lab.scoring.engine import score_records
from llm_behavior_lab.scoring.models import (
    ScaleReliabilityRecord,
    ScaleScoreRecord,
    ScaleScoreStatus,
)
from llm_behavior_lab.scoring.reliability import calculate_reliability
from llm_behavior_lab.storage import write_jsonl_records


@dataclass(frozen=True)
class ScoreRunResult:
    scores: list[ScaleScoreRecord]
    reliability: list[ScaleReliabilityRecord]
    output_root: Path


def score_run(run_root: Path, scoring_model_id: str | None = None) -> ScoreRunResult:
    """Score one persisted run and write versioned scoring artifacts."""
    snapshot = Questionnaire.model_validate_json(
        (run_root / "scale.json").read_text(encoding="utf-8")
    )
    questionnaire = snapshot
    used_fallback = False
    if not questionnaire.scoring_models:
        descriptor = describe_questionnaire(questionnaire.id)
        parameters = {
            parameter.name: str(questionnaire.metadata[parameter.name])
            for parameter in descriptor.parameters
            if parameter.name in questionnaire.metadata
        }
        current = resolve_questionnaire(questionnaire.id, parameters)
        _validate_legacy_compatibility(snapshot, current)
        questionnaire = current
        used_fallback = True
    if not questionnaire.scoring_models:
        raise ValueError(
            f"questionnaire {questionnaire.id} has no executable scoring models"
        )
    model_id = scoring_model_id or questionnaire.scoring_models[0].id
    scoring_model = next(
        (candidate for candidate in questionnaire.scoring_models if candidate.id == model_id),
        None,
    )
    if scoring_model is None:
        raise ValueError(f"unknown scoring model {model_id!r} for {questionnaire.id}")

    records = _load_response_records(run_root / "responses")
    scores = score_records(questionnaire, scoring_model, records)
    reliability = _reliability_records(scores)
    output_root = run_root / "scoring" / f"{scoring_model.id}-{scoring_model.version}"
    if output_root.exists():
        raise FileExistsError(f"scoring output already exists: {output_root}")
    output_root.mkdir(parents=True)
    write_jsonl_records(output_root / "scores.jsonl", scores)
    write_jsonl_records(output_root / "reliability.jsonl", reliability)
    (output_root / "scoring.json").write_text(
        json.dumps(
            {
                "questionnaire_id": questionnaire.id,
                "questionnaire_version": questionnaire.version,
                "scoring_model_id": scoring_model.id,
                "scoring_model_version": scoring_model.version,
                "scoring_model_provenance": scoring_model.provenance,
                "used_current_definition_fallback": used_fallback,
                "snapshot_definition_digest": _definition_digest(snapshot),
                "scoring_definition_digest": _definition_digest(questionnaire),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return ScoreRunResult(scores=scores, reliability=reliability, output_root=output_root)


def _validate_legacy_compatibility(snapshot: Questionnaire, current: Questionnaire) -> None:
    if snapshot.id != current.id or snapshot.version != current.version:
        raise ValueError("legacy questionnaire identity does not match current definition")
    snapshot_items = [
        (item.id, item.order, item.text, item.response_format.model_dump(mode="json"))
        for item in snapshot.items
    ]
    current_items = [
        (item.id, item.order, item.text, item.response_format.model_dump(mode="json"))
        for item in current.items
    ]
    if snapshot_items != current_items:
        raise ValueError("legacy questionnaire items do not match current definition")


def _definition_digest(questionnaire: Questionnaire) -> str:
    payload = questionnaire.model_dump_json(exclude_none=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _load_response_records(path: Path) -> list[ItemResponseRecord]:
    records: list[ItemResponseRecord] = []
    for response_path in sorted(path.glob("*.jsonl")):
        for line in response_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(ItemResponseRecord.model_validate_json(line))
    return records


def _reliability_records(scores: list[ScaleScoreRecord]) -> list[ScaleReliabilityRecord]:
    grouped: dict[tuple[str, str | None], dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    exemplars: dict[tuple[str, str | None], ScaleScoreRecord] = {}
    for score in scores:
        if score.status != ScaleScoreStatus.COMPLETED:
            continue
        condition = score.metadata.get("condition_id")
        condition_id = str(condition) if condition is not None else None
        keys = [(score.scale_id, None)]
        if condition_id is not None:
            keys.append((score.scale_id, condition_id))
        for key in keys:
            exemplars[key] = score
            for item in score.items:
                grouped[key][score.subject_id][item.item_id] = item.keyed_value

    output: list[ScaleReliabilityRecord] = []
    sorted_groups = sorted(
        grouped.items(),
        key=lambda entry: (
            entry[0][0],
            entry[0][1] is not None,
            entry[0][1] or "",
        ),
    )
    for key, item_values in sorted_groups:
        scale_id, condition_id = key
        exemplar = exemplars[key]
        metadata: dict[str, object] = (
            {"condition_id": condition_id} if condition_id is not None else {}
        )
        output.append(
            calculate_reliability(
                scale_id=scale_id,
                scoring_model_id=exemplar.scoring_model_id,
                scoring_model_version=exemplar.scoring_model_version,
                item_values=item_values,
                metadata=metadata,
            )
        )
    return output
