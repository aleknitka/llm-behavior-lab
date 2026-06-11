from llm_behavior_lab import (
    IowaGamblingConfig,
    IowaGamblingTask,
    __version__,
    analyze_task_run,
    create_experiment_design,
    create_personas,
    create_protocol_experiment,
    create_protocol_run,
    export_results,
    export_task_results,
    list_persona_fields,
    preview_persona_creation,
    protocol_fingerprint,
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
    assert callable(create_personas)
    assert callable(list_persona_fields)
    assert callable(preview_persona_creation)
    assert callable(create_protocol_experiment)
    assert callable(create_protocol_run)
    assert callable(protocol_fingerprint)
    assert callable(score_run)
    assert callable(export_results)
    assert callable(run_behavioral_task)
    assert callable(analyze_task_run)
    assert callable(export_task_results)
    assert IowaGamblingTask(IowaGamblingConfig()).id == "four-deck-card-task"
