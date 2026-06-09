from pathlib import Path

from llm_behavior_lab.experiments import (
    ExperimentDesign,
    PersonaDesign,
    ProviderDesign,
    ScaleProcedureDesign,
    TaskProcedureDesign,
    create_experiment_design,
    load_experiment_design,
    materialize_personas,
)


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
