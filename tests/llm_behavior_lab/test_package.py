from llm_behavior_lab import (
    IowaGamblingConfig,
    IowaGamblingTask,
    __version__,
    analyze_task_run,
    create_experiment_design,
    export_results,
    export_task_results,
    run_behavioral_task,
    run_questionnaire,
    run_questionnaire_async,
    score_run,
)


def test_package_exports_version_and_runners() -> None:
    assert __version__ == "0.1.0"
    assert callable(run_questionnaire)
    assert callable(run_questionnaire_async)
    assert callable(create_experiment_design)
    assert callable(score_run)
    assert callable(export_results)
    assert callable(run_behavioral_task)
    assert callable(analyze_task_run)
    assert callable(export_task_results)
    assert IowaGamblingTask(IowaGamblingConfig()).id == "four-deck-card-task"
