import json
from pathlib import Path

from llm_behavior_lab.experiments import (
    ExperimentDesign,
    PersonaDesign,
    ProviderDesign,
    ScaleProcedureDesign,
    TaskProcedureDesign,
    create_experiment_design,
    load_experiment_design,
    load_personas,
    materialize_personas,
)
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
    assert (path.parent / "personas.jsonl").exists()
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
        (design_path.parent / "personas.jsonl").read_text(encoding="utf-8")
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
    assignment_rows = [
        json.loads(line)
        for line in (experiment_root / "protocol_assignments.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert persisted_protocol["factors"][0]["levels"][0]["value"] == {
        "type": "rand_uniform_range",
        "left": 20,
        "right": 25,
    }
    assert all(isinstance(persona.features.age, int) for persona in batch.personas)
    assert all(isinstance(row["factor_values"]["age"], int) for row in assignment_rows)
    assert all(row["factor_level_ids"]["age"] == "adult" for row in assignment_rows)
