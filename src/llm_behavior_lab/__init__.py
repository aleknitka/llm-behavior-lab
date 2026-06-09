"""Persona-based LLM questionnaire and behavioral-task experimentation."""

from llm_behavior_lab.behavioral_tasks import IowaGamblingConfig, IowaGamblingTask
from llm_behavior_lab.behavioral_tasks.analysis import (
    analyze_task_run,
    export_task_results,
)
from llm_behavior_lab.behavioral_tasks.runner import run_behavioral_task
from llm_behavior_lab.experiments import (
    ExperimentDesign,
    create_experiment_design,
    materialize_personas,
)
from llm_behavior_lab.runner import (
    run_persisted_persona_batch,
    run_questionnaire,
    run_questionnaire_async,
)
from llm_behavior_lab.scoring import export_results, score_run

__all__ = [
    "ExperimentDesign",
    "IowaGamblingConfig",
    "IowaGamblingTask",
    "__version__",
    "analyze_task_run",
    "create_experiment_design",
    "export_results",
    "export_task_results",
    "materialize_personas",
    "run_persisted_persona_batch",
    "run_behavioral_task",
    "run_questionnaire",
    "run_questionnaire_async",
    "score_run",
]

__version__ = "0.1.0"
