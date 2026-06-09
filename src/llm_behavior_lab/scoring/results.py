import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from llm_behavior_lab.analysis import load_response_table
from llm_behavior_lab.scoring.models import ScaleReliabilityRecord, ScaleScoreRecord


@dataclass(frozen=True)
class ResultsExport:
    output_root: Path
    summary: dict[str, object]


def export_results(run_root: Path, scoring_directory: str | None = None) -> ResultsExport:
    """Export tidy response, score, reliability, and descriptive result files."""
    scoring_roots = sorted((run_root / "scoring").iterdir())
    if scoring_directory is None:
        if len(scoring_roots) != 1:
            raise ValueError("scoring directory is required when a run has multiple score outputs")
        scoring_root = scoring_roots[0]
    else:
        scoring_root = run_root / "scoring" / scoring_directory

    scores = _load_models(scoring_root / "scores.jsonl", ScaleScoreRecord)
    reliability = _load_models(
        scoring_root / "reliability.jsonl", ScaleReliabilityRecord
    )
    output_root = run_root / "results" / scoring_root.name
    if output_root.exists():
        raise FileExistsError(f"results output already exists: {output_root}")
    output_root.mkdir(parents=True)

    response_frame = pd.DataFrame(load_response_table(run_root))
    score_rows = []
    for record in scores:
        row = record.model_dump(mode="json")
        metadata = row.pop("metadata")
        row["condition_id"] = metadata.get("condition_id")
        row["base_subject_id"] = metadata.get("base_subject_id")
        row["iteration_index"] = metadata.get("iteration_index")
        score_rows.append(row)
    score_frame = pd.DataFrame(score_rows)
    reliability_frame = pd.DataFrame(
        [record.model_dump(mode="json") for record in reliability]
    )
    response_frame.to_csv(output_root / "responses.csv", index=False)
    score_frame.to_csv(output_root / "scores.csv", index=False)
    reliability_frame.to_csv(output_root / "reliability.csv", index=False)

    completed = score_frame[score_frame["status"] == "completed"]
    summary_rows = []
    if not completed.empty:
        group_columns = ["scale_id"]
        include_condition = bool(completed["condition_id"].notna().any())
        if include_condition:
            group_columns.append("condition_id")
        grouped = completed.groupby(group_columns, dropna=False)["score"]
        for group_key, values in grouped:
            scale_id = group_key[0] if isinstance(group_key, tuple) else group_key
            row = {
                "scale_id": scale_id,
                "count": int(values.count()),
                "mean": float(values.mean()),
                "standard_deviation": (
                    float(values.std(ddof=1)) if values.count() > 1 else None
                ),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            }
            if include_condition and isinstance(group_key, tuple):
                row["condition_id"] = group_key[-1]
            summary_rows.append(row)
    summary: dict[str, object] = {
        "run_id": run_root.name,
        "scoring_directory": scoring_root.name,
        "scales": summary_rows,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return ResultsExport(output_root=output_root, summary=summary)


def _load_models(path: Path, model_type):
    return [
        model_type.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
