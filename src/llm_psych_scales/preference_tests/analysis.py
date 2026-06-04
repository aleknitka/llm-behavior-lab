from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_psych_scales.preference_tests.models import PairwisePreferenceRecord
from llm_psych_scales.responses.base import ResponseStatus


def load_pairwise_preference_rows(path: Path) -> list[dict[str, Any]]:
    records = _load_records(path)
    return [
        {
            "subject_id": record.subject_id,
            "session_id": record.session_id,
            "run_id": record.run_id,
            "preference_experiment_id": record.preference_experiment_id,
            "preference_experiment_version": record.preference_experiment_version,
            "trial_id": record.trial_id,
            "trial_order": record.trial_order,
            "stimulus_ids": list(record.stimulus_ids),
            "displayed_stimulus_ids": list(record.displayed_stimulus_ids),
            "selected_label": record.selected_label,
            "selected_stimulus_id": record.selected_stimulus_id,
            "rejected_stimulus_id": record.rejected_stimulus_id,
            "status": str(record.status),
            "error": record.error,
            "logprobs_available": record.logprobs is not None,
            "metadata": record.metadata,
        }
        for record in records
    ]


def summarize_pairwise_preferences(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stimulus_wins: dict[str, int] = {}
    pair_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        if row.get("status") != str(ResponseStatus.COMPLETED):
            continue
        selected_stimulus_id = row.get("selected_stimulus_id")
        stimulus_ids = row.get("stimulus_ids")
        if not isinstance(selected_stimulus_id, str) or not isinstance(stimulus_ids, list):
            continue
        pair_id = "__".join(sorted(str(stimulus_id) for stimulus_id in stimulus_ids))
        stimulus_wins[selected_stimulus_id] = stimulus_wins.get(selected_stimulus_id, 0) + 1
        pair_counts.setdefault(pair_id, {})
        pair_counts[pair_id][selected_stimulus_id] = (
            pair_counts[pair_id].get(selected_stimulus_id, 0) + 1
        )
    return {"stimulus_wins": stimulus_wins, "pair_counts": pair_counts}


def _load_records(path: Path) -> list[PairwisePreferenceRecord]:
    if path.is_file():
        return _load_jsonl_file(path)

    responses_root = path / "responses" if (path / "responses").is_dir() else path
    records: list[PairwisePreferenceRecord] = []
    for response_path in sorted(responses_root.glob("*.jsonl")):
        records.extend(_load_jsonl_file(response_path))
    return records


def _load_jsonl_file(path: Path) -> list[PairwisePreferenceRecord]:
    records: list[PairwisePreferenceRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(PairwisePreferenceRecord.model_validate(json.loads(line)))
    return records
