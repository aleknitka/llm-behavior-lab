# Stateful Behavioral Tasks and Project Rename

## Summary

Rename the project to **`llm-behavior-lab`**, including the Python package
`llm_behavior_lab` and CLI command. Add a reusable stateful behavioral-task
engine, with Iowa Gambling as its first implementation.

The design follows behavioral-task systems such as
[PEBL](https://pebl.sourceforge.net/) while adapting them for LLMs:
deterministic environment transitions, bounded action choices, persistent
conversation history, and auditable trial-level records.

## Core Architecture

- Introduce a `BehavioralTask` protocol with:
  - `initial_state()`
  - `observe(state)`
  - `allowed_actions(state)`
  - `apply_action(state, action)`
  - `is_complete(state)`
  - `summarize(records)`
- Keep task environment logic deterministic and separate from the LLM client.
- The LLM receives:
  1. One persona system message.
  2. One task instruction without the recognized task name.
  3. Prior assistant selections and user outcome feedback.
  4. The current trial observation.
- The LLM selects only an allowed action. It never receives schedules, internal
  deck IDs, advantageous classifications, or future outcomes.
- Preserve existing questionnaire and pairwise runners rather than forcing them
  into the task engine.

## Public Models and Configuration

Add Pydantic models for:

- `BehavioralTaskDefinition`
- `TaskObservation`
- `TaskAction`
- `TaskState`
- `TaskTransition`
- `TaskAttemptRecord`
- `TaskTrialRecord`
- `TaskSummaryRecord`
- `TaskProcedureDesign`

Experiment manifests become a validated union of `ScaleProcedureDesign` and
`TaskProcedureDesign`, with exactly one procedure per experiment.

Iowa Gambling configuration will expose:

- Trial count and starting balance.
- Four internal deck contingencies.
- Gain amount and editable penalty templates per deck.
- Fixed or template-generated schedules.
- Optional block-level shuffling.
- Shared-experiment or per-subject schedule assignment.
- Alternate visible labels.
- Deterministic per-subject counterbalancing between visible labels and hidden
  contingencies.
- Feedback fields shown after each selection.

Ship a versioned default configuration containing classic IGT contingencies,
based on the documented gain/loss structure, while allowing experiments to
override them through a task configuration file.

## Execution and Persistence

Use this trial loop:

```text
restore state -> render observation -> request action
-> validate action -> apply transition -> persist outcome
-> append action and feedback to conversation -> next trial
```

- A malformed action gets one corrective retry without advancing task state.
- A second invalid action marks that subject failed.
- Provider failures never advance the environment.
- Resume reconstructs state by replaying completed trial records against the
  resolved schedule.
- Completed subjects are skipped when resuming.
- Failed subjects require an explicit retry option.
- All schedules are resolved and persisted before the first model call.
- Subjects may run concurrently with configurable bounded async concurrency;
  trials within one subject remain sequential.

Avoid quadratic message duplication by storing:

- `task.json`: resolved task definition.
- `schedules/schedule.json` or `schedules/{subject_id}.json`: hidden resolved
  schedules and label mappings.
- `conversations/{subject_id}.jsonl`: each system, user, and assistant message
  once.
- `responses/{subject_id}.jsonl`: validated trial events referencing
  conversation message indexes.
- `run.jsonl`: generic run metadata using `procedure_kind`, `procedure_id`, and
  `procedure_version`.

Record operational request latency if useful, but never label it as human
reaction time.

## Iowa Gambling Analysis

The `task-analyze` stage will produce:

- Selection count and proportion for each hidden contingency and displayed
  label.
- Advantageous minus disadvantageous selection score.
- Scores for five 20-trial blocks by default.
- Final balance and net earnings.
- Gain, loss, and net outcome totals.
- Choice progression across trials.
- Invalid attempt and provider failure counts.
- Schedule ID, seed, assignment mode, and label mapping metadata.

The original task identity and hidden contingency mapping remain available in
artifacts but never appear in model-facing prompts.

## CLI Workflows

Rename the executable to `llm-behavior-lab`.

Scale workflow:

```text
scale-design -> personas -> scale-run -> scale-score -> scale-results
```

Task workflow:

```text
task-design -> personas -> task-run -> task-analyze -> task-results
```

`task-design` accepts a built-in task ID and optional task configuration file.
`task-run` supports concurrency, normal resume, and explicit retry of failed
subjects.

## Rename and Documentation

- Rename the package directory and every import to `llm_behavior_lab`.
- Rename project metadata, executable, tests, documentation, examples, and
  artifact metadata.
- Update `README.md` with both workflows.
- Add a behavioral-task authoring guide covering task design, configuration,
  execution, recovery, and analysis.
- Add an Iowa Gambling README documenting its scientific source, deviations,
  configurable contingencies, masking strategy, and interpretation.
- Add docstrings showing how to implement another stateful task such as Risky
  Gains.
- Treat this as a breaking release without compatibility import aliases.
- Rename the GitHub repository and update local Git remotes as an operational
  release step; existing experiment artifacts are not automatically migrated.

## Test Plan

- Validate task definitions, schedules, label mappings, and manifest unions.
- Verify hidden schedules and canonical labels never enter model messages.
- Verify persona text remains the system prompt throughout all trials.
- Test deterministic shared and per-subject schedules.
- Test counterbalanced visible-label mappings.
- Test state transitions and known Iowa Gambling balances.
- Test corrective retry without state advancement.
- Test interruption and exact state reconstruction from the ledger.
- Test concurrent subjects remain isolated.
- Test task summaries and block scores against hand-calculated fixtures.
- Test renamed imports, package metadata, and every CLI workflow.
- Run `ruff`, `ty`, and the complete `pytest` suite.

## Research Basis

- [Cognitive Atlas: Risky Gains Task](https://cognitiveatlas.org/task/id/tsk_4a57abb949d5b/)
- [Ahn et al. payoff distributions and bandit framing](https://bpb-us-w2.wpmucdn.com/u.osu.edu/dist/4/19514/files/2015/11/Ahn2008-1ex6c28.pdf)
- [PsyToolkit Iowa Gambling implementation](https://www.psytoolkit.org/library/igt.html)
- [PEBL experiment framework](https://pebl.sourceforge.net/)
- [Block-wise IGT analysis example](https://pmc.ncbi.nlm.nih.gov/articles/PMC2826566/)

## Assumptions

- Iowa Gambling is the reference task for v1.
- Task contingencies are configurable but fully resolved before execution.
- Visible labels are noncanonical and counterbalanced per subject.
- The model is told to maximize its balance but is not told the task name or
  deck properties.
- Multi-procedure studies combining scales and tasks are deferred.
