from __future__ import annotations

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID, uuid4

import friendlywords
from pydantic import BaseModel

if TYPE_CHECKING:
    from llm_behavior_lab.personas.factory import PersonaBatch


_EXPERIMENT_ID_PATTERN = re.compile(r"^[a-z0-9]+-[a-z0-9]+-[a-z0-9]+$")


@dataclass(frozen=True)
class ExperimentPaths:
    experiment_root: Path
    personas_path: Path
    base_personas_path: Path
    protocol_path: Path
    protocol_assignments_path: Path
    metadata_path: Path
    run_root: Path
    run_path: Path
    responses_root: Path
    scale_path: Path

    def response_path_for_subject(self, subject_id: str) -> Path:
        return self.responses_root / f"{subject_id}.jsonl"


def validate_experiment_id(experiment_id: str) -> str:
    if not _EXPERIMENT_ID_PATTERN.fullmatch(experiment_id):
        raise ValueError(
            "experiment_id must contain exactly three lowercase letter or digit items "
            "separated by hyphens"
        )
    return experiment_id


def generate_experiment_id(seed: int | None = None) -> str:
    if seed is None:
        return _friendlywords_generate()

    state = random.getstate()
    try:
        random.seed(seed)
        return _friendlywords_generate()
    finally:
        random.setstate(state)


def normalize_prefixed_uuid(prefix: str, value: str | None = None) -> str:
    if value is None:
        return f"{prefix}{uuid4()}"

    uuid_text = value.removeprefix(prefix)
    UUID(uuid_text)
    return f"{prefix}{uuid_text}"


def slugify_model_name(model: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")
    if not slug:
        raise ValueError("model name must contain at least one letter or digit")
    return slug


def build_run_directory_name(
    questionnaire_shorthand: str,
    model: str,
    started_at: datetime,
) -> str:
    timestamp = started_at.strftime("%Y%m%d%H%M%S")
    return f"run-{questionnaire_shorthand}-{slugify_model_name(model)}-{timestamp}"


def resolve_experiment_paths(
    project_root: Path,
    experiment_id: str,
    run_id: str,
) -> ExperimentPaths:
    experiment_id = validate_experiment_id(experiment_id)

    experiment_root = project_root / "experiments" / experiment_id
    run_root = experiment_root / run_id
    return ExperimentPaths(
        experiment_root=experiment_root,
        personas_path=experiment_root / "personas.jsonl",
        base_personas_path=experiment_root / "base_personas.jsonl",
        protocol_path=experiment_root / "protocol.json",
        protocol_assignments_path=experiment_root / "protocol_assignments.jsonl",
        metadata_path=experiment_root / "metadata.jsonl",
        run_root=run_root,
        run_path=run_root / "run.jsonl",
        responses_root=run_root / "responses",
        scale_path=run_root / "scale.json",
    )


def _friendlywords_generate() -> str:
    generated = cast(Any, friendlywords).generate("ppo", separator="-")
    if not isinstance(generated, str):
        raise TypeError("friendlywords.generate returned a non-string experiment name")
    return validate_experiment_id(generated)


def append_jsonl_record(path: Path, record: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(record.model_dump_json() + "\n")


def write_persona_batch_jsonl(project_root: Path, batch: PersonaBatch) -> Path:
    experiment_id = validate_experiment_id(batch.experiment_id)
    path = project_root / "experiments" / experiment_id / "personas.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write(
            batch.model_dump_json(
                include={
                    "experiment_id": True,
                    "metadata": True,
                    "personas": True,
                }
            )
            + "\n"
        )
    return path


def write_persona_batch_jsonl_at_path(path: Path, batch: PersonaBatch) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write(
            batch.model_dump_json(
                include={
                    "experiment_id": True,
                    "metadata": True,
                    "personas": True,
                }
            )
            + "\n"
        )
    return path


def write_jsonl_records(path: Path, records: Sequence[BaseModel]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(record.model_dump_json() + "\n")
    return path
