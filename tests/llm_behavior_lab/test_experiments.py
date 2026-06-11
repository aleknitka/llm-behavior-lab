import json
from pathlib import Path

import pytest

from llm_behavior_lab.experiments import (
    ExperimentDesign,
    PersonaDesign,
    ProviderDesign,
    ScaleProcedureDesign,
    TaskProcedureDesign,
    create_experiment_design,
    create_personas,
    list_persona_fields,
    load_experiment_design,
    load_personas,
    materialize_personas,
    preview_persona_creation,
)
from llm_behavior_lab.personas.factory import RequestedDemographicField
from llm_behavior_lab.protocols import ExperimentProtocol


def test_design_round_trip_and_persona_materialization(tmp_path: Path) -> None:
    design = ExperimentDesign(
        experiment_id="pilot-study-one",
        procedure=ScaleProcedureDesign(
            questionnaire_id="bfi_10",
            scoring_model_id="default",
        ),
        personas=PersonaDesign(count=2, seed=7),
        provider=ProviderDesign(
            model="test-model",
            base_url="http://localhost:1234/v1",
        ),
    )

    path = create_experiment_design(tmp_path, design)
    loaded = load_experiment_design(tmp_path, "pilot-study-one")
    batch = materialize_personas(tmp_path, loaded)

    assert path == tmp_path / "experiments" / "pilot-study-one" / "design.json"
    assert loaded == design
    assert len(batch.personas) == 2
    assert (path.parent / "personas.json").exists()
    assert load_personas(tmp_path, "pilot-study-one") == batch


def test_task_design_round_trip() -> None:
    design = ExperimentDesign(
        experiment_id="card-task-one",
        procedure=TaskProcedureDesign(
            task_id="four-deck-card-task",
            task_config={"trial_count": 20, "schedule_assignment": "per_subject"},
        ),
        personas=PersonaDesign(count=1, seed=7),
        provider=ProviderDesign(
            model="test-model",
            base_url="http://localhost:1234/v1",
        ),
    )

    assert design.procedure.kind == "task"
    assert design.procedure.task_config["trial_count"] == 20


def test_provider_design_round_trips_execution_policy() -> None:
    provider = ProviderDesign(
        model="test-model",
        base_url="http://localhost:1234/v1",
        max_attempts=5,
        initial_backoff_seconds=0.5,
        max_backoff_seconds=8,
        max_concurrency=6,
    )

    assert ProviderDesign.model_validate(provider.model_dump()) == provider


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_attempts", 0),
        ("initial_backoff_seconds", -0.1),
        ("max_backoff_seconds", -0.1),
        ("max_concurrency", 0),
    ],
)
def test_provider_design_rejects_invalid_execution_policy(
    field: str,
    value: int | float,
) -> None:
    with pytest.raises(ValueError):
        ProviderDesign(
            model="test-model",
            base_url="http://localhost:1234/v1",
            **{field: value},
        )


def test_create_personas_persists_and_returns_deterministic_batch(tmp_path: Path) -> None:
    design = PersonaDesign(
        count=2,
        seed=7,
        requested_fields={
            RequestedDemographicField.AGE,
            RequestedDemographicField.COUNTRY,
        },
    )

    first = create_personas(tmp_path, "persona-study-one", design)
    persisted = load_personas(tmp_path, "persona-study-one")

    assert first == persisted
    assert first.metadata.requested_fields == ["age", "country"]
    assert (tmp_path / "experiments" / "persona-study-one" / "personas.json").exists()


def test_preview_persona_creation_resolves_defaults_without_writing(tmp_path: Path) -> None:
    preview = preview_persona_creation(
        experiment_id="preview-study-one",
        design=PersonaDesign(
            count=3,
            seed=11,
            requested_fields={RequestedDemographicField.AGE},
        ),
    )

    assert preview.model_dump(mode="json") == {
        "count": 3,
        "requested_fields": ["age"],
        "seed": 11,
        "experiment_id": "preview-study-one",
        "generation_config": {
            "field_values": {},
            "field_probabilities": {},
        },
    }
    assert list(tmp_path.iterdir()) == []


def test_list_persona_fields_describes_supported_configuration() -> None:
    fields = {descriptor.id: descriptor for descriptor in list_persona_fields()}

    assert fields[RequestedDemographicField.AGE].value_type == "integer"
    assert fields[RequestedDemographicField.AGE].supports_range is True
    assert fields[RequestedDemographicField.GENDER].value_type == "enum"
    assert fields[RequestedDemographicField.GENDER].allowed_values == [
        "female",
        "male",
        "non_binary",
    ]
    assert fields[RequestedDemographicField.GENDER].supports_probabilities is True
    assert fields[RequestedDemographicField.REGION].supports_probabilities is False


def test_create_personas_refuses_overwrite_without_replace(tmp_path: Path) -> None:
    design = PersonaDesign(
        count=1,
        seed=7,
        requested_fields={RequestedDemographicField.AGE},
    )
    original = create_personas(tmp_path, "replace-study-one", design)

    with pytest.raises(FileExistsError, match="personas already exist"):
        create_personas(
            tmp_path,
            "replace-study-one",
            PersonaDesign(
                count=1,
                seed=9,
                requested_fields={RequestedDemographicField.AGE},
            ),
        )

    replacement = create_personas(
        tmp_path,
        "replace-study-one",
        PersonaDesign(
            count=1,
            seed=9,
            requested_fields={RequestedDemographicField.AGE},
        ),
        replace=True,
    )

    assert replacement != original
    assert load_personas(tmp_path, "replace-study-one") == replacement


def test_persona_design_rejects_configuration_for_unrequested_field() -> None:
    with pytest.raises(ValueError, match="configured fields must be requested"):
        PersonaDesign.model_validate(
            {
                "requested_fields": ["age"],
                "generation_config": {
                    "field_values": {"country": "PL"},
                },
            }
        )


def test_persona_range_declaration_stays_in_design_and_materializes_to_integer(
    tmp_path: Path,
) -> None:
    design = ExperimentDesign.model_validate(
        {
            "experiment_id": "range-study-one",
            "procedure": {"kind": "scale", "questionnaire_id": "bfi_10"},
            "personas": {
                "count": 1,
                "seed": 7,
                "requested_fields": ["age"],
                "generation_config": {
                    "field_values": {
                        "age": {
                            "type": "rand_uniform_range",
                            "left": 20,
                            "right": 25,
                        }
                    }
                },
            },
            "provider": {
                "model": "test-model",
                "base_url": "http://localhost:1234/v1",
            },
        }
    )

    design_path = create_experiment_design(tmp_path, design)
    batch = materialize_personas(tmp_path, design)
    persisted_design = json.loads(design_path.read_text(encoding="utf-8"))
    persisted_personas = json.loads(
        (design_path.parent / "personas.json").read_text(encoding="utf-8")
    )

    assert persisted_design["personas"]["generation_config"]["field_values"]["age"] == {
        "type": "rand_uniform_range",
        "left": 20,
        "right": 25,
    }
    assert isinstance(batch.personas[0].features.age, int)
    assert isinstance(persisted_personas["personas"][0]["features"]["age"], int)


def test_protocol_range_declaration_and_realized_assignments_are_persisted(
    tmp_path: Path,
) -> None:
    protocol = ExperimentProtocol.model_validate(
        {
            "name": "age-range",
            "base_persona_count": 1,
            "seed": 11,
            "iterations": 2,
            "requested_fields": ["age"],
            "factors": [
                {
                    "name": "age_group",
                    "field": "age",
                    "levels": [
                        {
                            "id": "adult",
                            "value": {
                                "type": "rand_uniform_range",
                                "left": 20,
                                "right": 25,
                            },
                        }
                    ],
                }
            ],
        }
    )
    design = ExperimentDesign(
        experiment_id="protocol-range-one",
        procedure=ScaleProcedureDesign(questionnaire_id="bfi_10"),
        protocol=protocol,
        provider=ProviderDesign(
            model="test-model",
            base_url="http://localhost:1234/v1",
        ),
    )

    batch = materialize_personas(tmp_path, design)
    experiment_root = tmp_path / "experiments" / "protocol-range-one"
    persisted_protocol = json.loads((experiment_root / "protocol.json").read_text(encoding="utf-8"))
    assignment_rows = json.loads(
        (experiment_root / "protocol_assignments.json").read_text(encoding="utf-8")
    )["assignments"]

    assert persisted_protocol["factors"][0]["levels"][0]["value"] == {
        "type": "rand_uniform_range",
        "left": 20,
        "right": 25,
    }
    assert all(isinstance(persona.features.age, int) for persona in batch.personas)
    assert all(isinstance(row["factor_values"]["age"], int) for row in assignment_rows)
    assert all(row["factor_level_ids"]["age"] == "adult" for row in assignment_rows)


def test_load_personas_supports_legacy_snapshot_and_rejects_conflict(
    tmp_path: Path,
) -> None:
    design = PersonaDesign(count=1, seed=7)
    batch = create_personas(tmp_path, "legacy-persona-one", design)
    experiment_root = tmp_path / "experiments" / "legacy-persona-one"
    normalized = experiment_root / "personas.json"
    legacy = experiment_root / "personas.jsonl"
    normalized.replace(legacy)

    assert load_personas(tmp_path, "legacy-persona-one") == batch

    normalized.write_text(
        batch.model_copy(update={"experiment_id": "conflict-study-one"}).model_dump_json(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="conflicting canonical snapshot files"):
        load_personas(tmp_path, "legacy-persona-one")
