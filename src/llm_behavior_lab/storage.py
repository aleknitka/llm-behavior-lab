from __future__ import annotations

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from os import replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID, uuid4

import friendlywords
from pydantic import BaseModel

from llm_behavior_lab.responses.base.session import ExperimentMetadata, RunRecord

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
    run_root_override: Path | None = None,
) -> ExperimentPaths:
    experiment_id = validate_experiment_id(experiment_id)

    experiment_root = project_root / "experiments" / experiment_id
    run_root = run_root_override or experiment_root / run_id
    return ExperimentPaths(
        experiment_root=experiment_root,
        personas_path=experiment_root / "personas.json",
        base_personas_path=experiment_root / "base_personas.json",
        protocol_path=experiment_root / "protocol.json",
        protocol_assignments_path=experiment_root / "protocol_assignments.json",
        metadata_path=experiment_root / "metadata.json",
        run_root=run_root,
        run_path=run_root / "run.json",
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


def write_json_document(path: Path, document: BaseModel) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = document.model_dump_json(indent=2) + "\n"
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary.write(payload)
        temporary.flush()
        temporary_path = Path(temporary.name)
    try:
        replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def load_json_document[ModelT: BaseModel](
    path: Path,
    model: type[ModelT],
) -> ModelT:
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def resolve_compatible_snapshot_path(normalized: Path, legacy: Path) -> Path:
    if normalized.exists() and legacy.exists():
        normalized_payload = _load_snapshot_payload(normalized)
        legacy_payload = _load_snapshot_payload(legacy)
        if normalized_payload != legacy_payload:
            raise ValueError(
                "conflicting canonical snapshot files: "
                f"{normalized.name} and {legacy.name}"
            )
        return normalized
    if normalized.exists():
        return normalized
    if legacy.exists():
        return legacy
    return normalized


def update_experiment_metadata(path: Path, run_record: RunRecord) -> Path:
    legacy_path = path.with_suffix(".jsonl")
    legacy_runs = (
        [
            RunRecord.model_validate_json(line)
            for line in legacy_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if legacy_path.exists()
        else None
    )
    normalized_metadata = (
        load_json_document(path, ExperimentMetadata) if path.exists() else None
    )
    if normalized_metadata is not None and legacy_runs is not None:
        if normalized_metadata.runs != legacy_runs:
            raise ValueError(
                "conflicting canonical snapshot files: "
                f"{path.name} and {legacy_path.name}"
            )
        metadata = normalized_metadata
    elif normalized_metadata is not None:
        metadata = normalized_metadata
    elif legacy_runs is not None:
        metadata = ExperimentMetadata(
            experiment_id=run_record.experiment_id,
            runs=legacy_runs,
        )
    else:
        metadata = ExperimentMetadata(experiment_id=run_record.experiment_id)
    if metadata.experiment_id != run_record.experiment_id:
        raise ValueError("run experiment_id does not match metadata experiment_id")
    runs = [run for run in metadata.runs if run.run_id != run_record.run_id]
    updated = metadata.model_copy(update={"runs": [*runs, run_record]})
    written = write_json_document(path, updated)
    legacy_path.unlink(missing_ok=True)
    return written


def write_persona_batch(project_root: Path, batch: PersonaBatch) -> Path:
    experiment_id = validate_experiment_id(batch.experiment_id)
    path = project_root / "experiments" / experiment_id / "personas.json"
    return write_persona_batch_at_path(path, batch)


def write_persona_batch_at_path(path: Path, batch: PersonaBatch) -> Path:
    return write_json_document(path, batch)


def write_persona_batch_jsonl(project_root: Path, batch: PersonaBatch) -> Path:
    """Compatibility alias for callers using the former snapshot writer name."""
    return write_persona_batch(project_root, batch)


def write_persona_batch_jsonl_at_path(path: Path, batch: PersonaBatch) -> Path:
    """Compatibility alias that normalizes a legacy snapshot suffix to JSON."""
    normalized_path = path.with_suffix(".json") if path.suffix == ".jsonl" else path
    return write_persona_batch_at_path(normalized_path, batch)


def write_jsonl_records(path: Path, records: Sequence[BaseModel]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(record.model_dump_json() + "\n")
    return path


def _load_snapshot_payload(path: Path) -> Any:
    import json

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if path.suffix != ".jsonl":
            raise
        return [json.loads(line) for line in text.splitlines() if line.strip()]
