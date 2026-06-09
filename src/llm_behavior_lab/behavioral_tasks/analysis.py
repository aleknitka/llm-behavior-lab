from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from llm_behavior_lab.behavioral_tasks.base import TaskSummaryRecord, TaskTrialRecord
from llm_behavior_lab.behavioral_tasks.iowa_gambling import (
    IowaGamblingConfig,
    IowaGamblingTask,
)
from llm_behavior_lab.responses.base import ResponseStatus
from llm_behavior_lab.storage import write_jsonl_records


@dataclass(frozen=True)
class TaskAnalysisResult:
    summaries: list[TaskSummaryRecord]
    output_root: Path


@dataclass(frozen=True)
class TaskResultsExport:
    output_root: Path
    summary: dict[str, object]


def analyze_task_run(run_root: Path, block_size: int = 20) -> TaskAnalysisResult:
    """Derive subject-level behavioral metrics from completed trial ledgers."""
    config = IowaGamblingConfig.model_validate_json(
        (run_root / "task.json").read_text(encoding="utf-8")
    )
    task = IowaGamblingTask(config)
    summaries: list[TaskSummaryRecord] = []
    for response_path in sorted((run_root / "responses").glob("*.jsonl")):
        records = [
            TaskTrialRecord.model_validate_json(line)
            for line in response_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        completed = [
            record
            for record in records
            if record.status == ResponseStatus.COMPLETED and record.transition is not None
        ]
        if not completed:
            continue
        summary = task.summarize(
            [record.transition for record in completed if record.transition is not None],
            block_size=block_size,
        ).model_copy(
            update={
                "subject_id": completed[0].subject_id,
                "schedule_id": completed[0].schedule_id,
                "metadata": {
                    "invalid_attempt_count": sum(
                        attempt.status == ResponseStatus.INVALID
                        for record in records
                        for attempt in record.attempts
                    ),
                    "provider_failure_count": sum(
                        attempt.status == ResponseStatus.FAILED
                        for record in records
                        for attempt in record.attempts
                    ),
                },
            }
        )
        summaries.append(summary)

    output_root = run_root / "analysis" / f"{task.id}-{task.version}"
    if output_root.exists():
        raise FileExistsError(f"task analysis output already exists: {output_root}")
    output_root.mkdir(parents=True)
    write_jsonl_records(output_root / "summaries.jsonl", summaries)
    return TaskAnalysisResult(summaries=summaries, output_root=output_root)


def export_task_results(
    run_root: Path, analysis_directory: str | None = None
) -> TaskResultsExport:
    """Export task summaries and trial ledgers as flat CSV files."""
    analysis_roots = (
        sorted(path for path in (run_root / "analysis").iterdir() if path.is_dir())
        if (run_root / "analysis").exists()
        else []
    )
    if analysis_directory is None:
        if len(analysis_roots) != 1:
            raise ValueError(
                "analysis directory is required unless the run has exactly one analysis"
            )
        analysis_root = analysis_roots[0]
    else:
        analysis_root = run_root / "analysis" / analysis_directory

    summaries = [
        TaskSummaryRecord.model_validate_json(line)
        for line in (analysis_root / "summaries.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    output_root = run_root / "results" / analysis_root.name
    if output_root.exists():
        raise FileExistsError(f"task results output already exists: {output_root}")
    output_root.mkdir(parents=True)
    _write_summary_csv(output_root / "summaries.csv", summaries)
    _write_trials_csv(output_root / "trials.csv", run_root / "responses")
    summary: dict[str, object] = {
        "run_id": run_root.name,
        "analysis_directory": analysis_root.name,
        "subject_count": len(summaries),
        "completed_trial_count": sum(item.trial_count for item in summaries),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return TaskResultsExport(output_root=output_root, summary=summary)


def _write_summary_csv(path: Path, summaries: list[TaskSummaryRecord]) -> None:
    rows = []
    for summary in summaries:
        row = summary.model_dump(mode="json")
        row["internal_action_counts"] = json.dumps(row["internal_action_counts"])
        row["visible_action_counts"] = json.dumps(row["visible_action_counts"])
        row["blocks"] = json.dumps(row["blocks"])
        row["metadata"] = json.dumps(row["metadata"])
        rows.append(row)
    _write_csv(path, rows)


def _write_trials_csv(path: Path, responses_root: Path) -> None:
    rows: list[dict[str, object]] = []
    for response_path in sorted(responses_root.glob("*.jsonl")):
        for line in response_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = TaskTrialRecord.model_validate_json(line)
            transition = record.transition
            rows.append(
                {
                    "subject_id": record.subject_id,
                    "trial_index": record.trial_index,
                    "status": record.status,
                    "visible_action_id": (
                        transition.visible_action_id if transition else None
                    ),
                    "internal_action_id": (
                        transition.internal_action_id if transition else None
                    ),
                    "gain": transition.gain if transition else None,
                    "penalty": transition.penalty if transition else None,
                    "net": transition.net if transition else None,
                    "balance": transition.state.balance if transition else None,
                    "schedule_id": record.schedule_id,
                }
            )
    _write_csv(path, rows)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
