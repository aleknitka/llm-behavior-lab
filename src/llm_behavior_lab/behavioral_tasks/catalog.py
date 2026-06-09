from pathlib import Path

from llm_behavior_lab.behavioral_tasks.iowa_gambling import (
    IowaGamblingConfig,
    IowaGamblingTask,
)


def load_task_config(path: Path | None) -> IowaGamblingConfig:
    """Load an optional task configuration, otherwise use classic defaults."""
    if path is None:
        return IowaGamblingConfig()
    return IowaGamblingConfig.model_validate_json(path.read_text(encoding="utf-8"))


def resolve_behavioral_task(
    task_id: str, config: dict[str, object] | IowaGamblingConfig | None = None
) -> IowaGamblingTask:
    """Resolve a built-in task from a stable identifier and validated config."""
    if task_id != IowaGamblingTask.id:
        raise ValueError(f"unknown task_id: {task_id}")
    if isinstance(config, IowaGamblingConfig):
        resolved = config
    else:
        resolved = IowaGamblingConfig.model_validate(config or {})
    return IowaGamblingTask(resolved)
