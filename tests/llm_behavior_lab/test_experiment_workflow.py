from collections.abc import Sequence
from pathlib import Path

from llm_behavior_lab.main import main
from llm_behavior_lab.models import LlmQuestionResult, ModelSettings


class FirstAllowedAnswerClient:
    def __init__(self, api_key: str, base_url: str) -> None:
        self.api_key = api_key
        self.base_url = base_url

    def complete(
        self,
        messages: Sequence[dict[str, str]],
        settings: ModelSettings,
        allowed_answer_ids: Sequence[str],
    ) -> LlmQuestionResult:
        selected = allowed_answer_ids[0]
        return LlmQuestionResult(
            selected_answer_id=selected,
            raw_response=selected,
        )


def test_staged_scale_workflow_produces_analysis_ready_results(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "llm_behavior_lab.main.OpenAiChatClient",
        FirstAllowedAnswerClient,
    )
    common = ["--project-root", str(tmp_path), "--experiment-id", "workflow-test-one"]

    assert (
        main(
            [
                "scale-design",
                *common,
                "--questionnaire",
                "bfi_10",
                "--persona-count",
                "2",
                "--model",
                "test-model",
                "--base-url",
                "http://localhost:1234/v1",
                "--seed",
                "7",
                "--scoring-model-id",
                "default",
            ]
        )
        == 0
    )
    assert main(["personas", *common]) == 0
    assert main(["scale-run", *common, "--api-key", "test-key"]) == 0
    assert main(["scale-score", *common]) == 0
    assert main(["scale-results", *common]) == 0

    experiment_root = tmp_path / "experiments" / "workflow-test-one"
    run_root = next(path for path in experiment_root.glob("run-*") if path.is_dir())
    result_root = run_root / "results" / "default-1.0"

    assert (experiment_root / "design.json").exists()
    assert (experiment_root / "personas.jsonl").exists()
    assert (run_root / "scale.json").exists()
    assert len(list((run_root / "responses").glob("*.jsonl"))) == 2
    assert (run_root / "scoring" / "default-1.0" / "scores.jsonl").exists()
    assert (result_root / "responses.csv").exists()
    assert (result_root / "scores.csv").exists()
    assert (result_root / "reliability.csv").exists()
    assert (result_root / "summary.json").exists()
