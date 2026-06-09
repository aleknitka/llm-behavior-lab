from pathlib import Path

from llm_behavior_lab.questionnaires.base.scale import Transformation
from llm_behavior_lab.scoring import export_results
from llm_behavior_lab.scoring.models import (
    ScaleReliabilityRecord,
    ScaleReliabilityStatus,
    ScaleScoreRecord,
    ScaleScoreStatus,
)
from llm_behavior_lab.storage import write_jsonl_records


def test_export_results_writes_csv_tables_and_summary(tmp_path: Path) -> None:
    run_root = tmp_path / "run-bfi10-model-20260608120000"
    scoring_root = run_root / "scoring" / "default-1.0"
    responses_root = run_root / "responses"
    responses_root.mkdir(parents=True)
    (responses_root / "empty.jsonl").write_text("")
    write_jsonl_records(
        scoring_root / "scores.jsonl",
        [
            ScaleScoreRecord(
                subject_id="subject-1",
                session_id="session-1",
                run_id=run_root.name,
                questionnaire_id="bfi_10",
                questionnaire_version="1.0",
                scoring_model_id="default",
                scoring_model_version="1.0",
                scale_id="extraversion",
                transformation=Transformation.MEAN,
                status=ScaleScoreStatus.COMPLETED,
                score=4,
            ),
            ScaleScoreRecord(
                subject_id="subject-2",
                session_id="session-1",
                run_id=run_root.name,
                questionnaire_id="bfi_10",
                questionnaire_version="1.0",
                scoring_model_id="default",
                scoring_model_version="1.0",
                scale_id="extraversion",
                transformation=Transformation.MEAN,
                status=ScaleScoreStatus.COMPLETED,
                score=2,
            ),
        ],
    )
    write_jsonl_records(
        scoring_root / "reliability.jsonl",
        [
            ScaleReliabilityRecord(
                scale_id="extraversion",
                scoring_model_id="default",
                scoring_model_version="1.0",
                status=ScaleReliabilityStatus.COMPLETED,
                subject_count=2,
                item_count=2,
                alpha=0.8,
                confidence_interval=(0.1, 0.95),
            )
        ],
    )

    result = export_results(run_root)

    assert (result.output_root / "responses.csv").exists()
    assert (result.output_root / "scores.csv").exists()
    assert (result.output_root / "reliability.csv").exists()
    assert (result.output_root / "summary.json").exists()
    assert result.summary["scales"] == [
        {
            "scale_id": "extraversion",
            "count": 2,
            "mean": 3.0,
            "standard_deviation": 1.4142135623730951,
            "minimum": 2.0,
            "maximum": 4.0,
        }
    ]
