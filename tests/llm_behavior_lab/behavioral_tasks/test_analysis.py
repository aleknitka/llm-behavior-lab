import json
from pathlib import Path

from llm_behavior_lab.behavioral_tasks.analysis import (
    analyze_task_run,
    export_task_results,
)
from llm_behavior_lab.behavioral_tasks.iowa_gambling import IowaGamblingConfig, IowaGamblingTask
from llm_behavior_lab.behavioral_tasks.runner import run_behavioral_task
from llm_behavior_lab.models import LlmQuestionResult, ModelSettings, Persona


class FirstChoiceClient:
    def complete(self, messages, settings, allowed_answer_ids):
        selected = allowed_answer_ids[0]
        return LlmQuestionResult(selected_answer_id=selected, raw_response=selected)


def test_analysis_and_results_export_subject_metrics(tmp_path: Path) -> None:
    task = IowaGamblingTask(IowaGamblingConfig(trial_count=4))
    schedule = task.resolve_schedule(seed=9, subject_id="subject-1")
    run_root = tmp_path / "experiments" / "card-task-one" / "run-task-test"
    run_behavioral_task(
        persona=Persona(persona_id="subject-1", features={}),
        task=task,
        settings=ModelSettings(
            model="test",
            provider_base_url="http://localhost",
            temperature=0,
            timeout_seconds=10,
            seed=9,
        ),
        client=FirstChoiceClient(),
        project_root=tmp_path,
        experiment_id="card-task-one",
        run_id="run-task-test",
        resolved_schedule=schedule,
    )

    analysis = analyze_task_run(run_root)
    exported = export_task_results(run_root, analysis.output_root.name)

    assert len(analysis.summaries) == 1
    assert analysis.summaries[0].trial_count == 4
    assert (analysis.output_root / "summaries.jsonl").exists()
    assert (exported.output_root / "summaries.csv").exists()
    summary = json.loads((exported.output_root / "summary.json").read_text())
    assert summary["subject_count"] == 1
