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
        ExperimentProtocol,
        expand_protocol_personas,
    )
    from llm_behavior_lab.storage import validate_experiment_id

    return (
        ExperimentProtocol,
        Path,
        expand_protocol_personas,
        json,
        main,
        mo,
        validate_experiment_id,
    )


@app.cell
def _(mo):
    mo.md("""
    # Ollama BFI-10 factorial example

    Preview a reproducible UK age x affluence x urbanicity design before
    explicitly starting its 3,000 local model calls.
    """)
    return


@app.cell
def _(ExperimentProtocol, Path, expand_protocol_personas, json):
    example_root = Path(__file__).resolve().parent
    project_root = example_root.parents[1]
    protocol_path = example_root / "protocol.json"
    protocol = ExperimentProtocol.model_validate(
        json.loads(protocol_path.read_text(encoding="utf-8"))
    )
    expansion = expand_protocol_personas(protocol, "ollama-bfi-ten")
    return expansion, project_root, protocol, protocol_path


@app.cell
def _(expansion, mo, protocol):
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
        "iterations": protocol.iterations,
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
def _(mo):
    experiment_id = mo.ui.text(
        value="ollama-bfi-ten",
        label="Experiment ID",
        full_width=True,
    )
    run_button = mo.ui.run_button(
        label="Run experiment",
        kind="danger",
        tooltip="Creates experiment artifacts and starts 3,000 Ollama calls.",
    )
    mo.vstack(
        [
            mo.md(
                """
                ## Live execution

                This is the only control that creates files or contacts Ollama.
                A complete run can take substantial time.
                """
            ),
            experiment_id,
            run_button,
        ]
    )
    return experiment_id, run_button


@app.cell
def _(
    experiment_id,
    main,
    mo,
    project_root,
    protocol_path,
    run_button,
    validate_experiment_id,
):
    mo.stop(
        not run_button.value,
        mo.callout("Preview only. Press Run experiment to continue.", kind="info"),
    )
    selected_experiment_id = validate_experiment_id(experiment_id.value.strip())
    common_args = [
        "--project-root",
        str(project_root),
        "--experiment-id",
        selected_experiment_id,
    ]

    design_exit = main(
        [
            "scale-design",
            *common_args,
            "--questionnaire",
            "bfi_10",
            "--protocol",
            str(protocol_path),
            "--model",
            "gemma4:12b",
            "--base-url",
            "http://localhost:11434/v1",
            "--temperature",
            "0",
            "--seed",
            "20260609",
            "--scoring-model-id",
            "default",
            "--no-logprobs",
        ]
    )
    if design_exit != 0:
        raise RuntimeError("scale-design failed")

    personas_exit = main(["personas", *common_args])
    if personas_exit != 0:
        raise RuntimeError("persona materialization failed")

    run_exit = main(["scale-run", *common_args, "--api-key", "ollama"])
    if run_exit != 0:
        raise RuntimeError("scale run failed")

    score_exit = main(["scale-score", *common_args])
    if score_exit != 0:
        raise RuntimeError("scale scoring failed")

    results_exit = main(["scale-results", *common_args])
    if results_exit != 0:
        raise RuntimeError("result export failed")

    experiment_root = project_root / "experiments" / selected_experiment_id
    run_roots = sorted(path for path in experiment_root.glob("run-*") if path.is_dir())
    if len(run_roots) != 1:
        raise RuntimeError("expected exactly one completed run")
    run_root = run_roots[0]
    results_root = run_root / "results" / "default-1.0"
    return experiment_root, results_root, run_root


@app.cell
def _(experiment_root, json, mo, results_root, run_root):
    summary_path = results_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    scale_rows = summary["scales"]
    mo.vstack(
        [
            mo.callout("Experiment, scoring, and result export completed.", kind="success"),
            mo.md(
                f"""
                ## Artifacts

                - Personas: `{experiment_root / "personas.json"}`
                - Base personas: `{experiment_root / "base_personas.json"}`
                - Assignments: `{experiment_root / "protocol_assignments.json"}`
                - Experiment run index: `{experiment_root / "metadata.json"}`
                - Run: `{run_root}`
                - Run manifest: `{run_root / "run.json"}`
                - Item ledgers: `{run_root / "responses"}`
                - Scores: `{run_root / "scoring" / "default-1.0"}`
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
