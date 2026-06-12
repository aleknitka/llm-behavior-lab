import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo

    from llm_behavior_lab.main import main
    from llm_behavior_lab.protocols import (
        UnifiedExperimentProtocol,
        expand_protocol_personas,
    )

    return (
        Path,
        UnifiedExperimentProtocol,
        expand_protocol_personas,
        json,
        main,
        mo,
    )


@app.cell
def _(mo):
    mo.md("""
    # Ollama BFI-10 factorial example

    Preview a reproducible UK age x affluence x urbanicity design before
    explicitly starting its 120 local model calls.
    """)
    return


@app.cell
def _(Path, UnifiedExperimentProtocol, expand_protocol_personas, json):
    example_root = Path(__file__).resolve().parent
    project_root = example_root.parents[1]
    protocol_path = example_root / "protocol.json"
    protocol = UnifiedExperimentProtocol.model_validate(
        json.loads(protocol_path.read_text(encoding="utf-8"))
    )
    factorial = protocol.personas.factorial
    if factorial is None:
        raise RuntimeError("example protocol must contain a factorial persona design")
    expansion = expand_protocol_personas(factorial, protocol.experiment_id)
    return expansion, factorial, project_root, protocol, protocol_path


@app.cell
def _(expansion, factorial, mo):
    condition_rows_by_id = {}
    for assignment in expansion.assignments:
        condition_rows_by_id.setdefault(
            assignment.condition_id,
            {
                "condition_id": assignment.condition_id,
                "age": assignment.factor_values["age"],
                "affluence": assignment.factor_values["affluence_level"],
                "urbanicity": assignment.factor_values["urbanicity"],
            },
        )
    condition_rows = list(condition_rows_by_id.values())
    preview = {
        "base_personas": len(expansion.base_personas.personas),
        "conditions": len(condition_rows),
        "iterations": factorial.iterations,
        "expanded_personas": len(expansion.personas.personas),
        "questionnaire_calls": len(expansion.personas.personas) * 10,
    }
    mo.vstack(
        [
            mo.md(
                f"""
                ## Validated preview

                - Base personas: **{preview["base_personas"]}**
                - Factorial conditions: **{preview["conditions"]}**
                - Iterations per base-persona condition: **{preview["iterations"]}**
                - Expanded personas: **{preview["expanded_personas"]}**
                - BFI-10 calls: **{preview["questionnaire_calls"]}**
                """
            ),
            mo.ui.table(condition_rows, selection=None, pagination=False),
        ]
    )
    return


@app.cell
def _(mo, protocol):
    run_button = mo.ui.run_button(
        label="Run experiment",
        kind="danger",
        tooltip="Creates experiment artifacts and starts 120 Ollama calls.",
    )
    mo.vstack(
        [
            mo.md(
                f"""
                ## Live execution

                This is the only control that creates files or contacts Ollama.
                The immutable experiment ID is `{protocol.experiment_id}`.
                """
            ),
            run_button,
        ]
    )
    return (run_button,)


@app.cell
def _(
    json,
    main,
    mo,
    project_root,
    protocol,
    protocol_path,
    run_button,
):
    mo.stop(
        not run_button.value,
        mo.callout("Preview only. Press Run experiment to continue.", kind="info"),
    )
    experiment_root = project_root / "experiments" / protocol.experiment_id
    common_args = [
        "--project-root",
        str(project_root),
        "--file",
        str(protocol_path),
    ]

    if not experiment_root.exists():
        create_exit = main(["protocol-create", *common_args])
        if create_exit != 0:
            raise RuntimeError("protocol creation failed")

    existing_run_ids = {
        path.name for path in experiment_root.glob("run-protocol-*") if path.is_dir()
    }
    run_exit = main(
        ["protocol-create", *common_args, "--new-run", "--api-key", "ollama"]
    )
    if run_exit != 0:
        raise RuntimeError("protocol run failed")

    new_run_roots = sorted(
        path
        for path in experiment_root.glob("run-protocol-*")
        if path.is_dir() and path.name not in existing_run_ids
    )
    if len(new_run_roots) != 1:
        raise RuntimeError("expected exactly one new protocol run")
    run_root = new_run_roots[0]
    procedure_args = [
        "--project-root",
        str(project_root),
        "--experiment-id",
        protocol.experiment_id,
        "--run-id",
        run_root.name,
        "--step-id",
        "personality",
    ]

    score_exit = main(["scale-score", *procedure_args])
    if score_exit != 0:
        raise RuntimeError("scale scoring failed")

    results_exit = main(["scale-results", *procedure_args])
    if results_exit != 0:
        raise RuntimeError("result export failed")

    run_record = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
    cohort_root = experiment_root / "cohorts" / run_record["metadata"]["cohort_id"]
    step_root = run_root / "steps" / "personality"
    results_root = step_root / "results" / "default-1.0"
    return cohort_root, experiment_root, results_root, run_root, step_root


@app.cell
def _(cohort_root, experiment_root, json, mo, results_root, run_root, step_root):
    summary_path = results_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    scale_rows = summary["scales"]
    mo.vstack(
        [
            mo.callout("Experiment, scoring, and result export completed.", kind="success"),
            mo.md(
                f"""
                ## Artifacts

                - Protocol: `{experiment_root / "protocol.json"}`
                - Experiment run index: `{experiment_root / "metadata.json"}`
                - Cohort: `{cohort_root}`
                - Personas: `{cohort_root / "personas.json"}`
                - Assignments: `{cohort_root / "protocol-assignments.json"}`
                - Run: `{run_root}`
                - Run manifest: `{run_root / "run.json"}`
                - Questionnaire step: `{step_root}`
                - Item ledgers: `{step_root / "responses"}`
                - Scores: `{step_root / "scoring" / "default-1.0"}`
                - Results: `{results_root}`
                - Persona-enriched responses: `{results_root / "responses.csv"}`
                - Summary: `{summary_path}`
                """
            ),
            mo.ui.table(scale_rows, selection=None, page_size=12),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
