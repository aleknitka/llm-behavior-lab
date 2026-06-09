# Behavioral Task Authoring

Behavioral tasks differ from questionnaires because each model choice changes
the next environment state. The reusable task loop is:

```text
observe -> choose an allowed action -> apply a deterministic transition
-> persist the transition -> reveal feedback -> observe again
```

The model selects actions. Task code owns rewards, losses, state changes,
completion rules, and metrics.

## 1. Design the Experiment

Choose a task, persona design, provider, seed, and schedule policy before
execution. Keep hidden task properties out of model-facing instructions.

For the built-in card task:

```json
{
  "trial_count": 100,
  "starting_balance": 2000,
  "schedule_mode": "template",
  "schedule_assignment": "shared",
  "shuffle_template_blocks": false,
  "visible_labels": ["Circle", "Square", "Triangle", "Star"],
  "feedback_fields": ["gain", "penalty", "net", "balance"]
}
```

The four `decks` entries may also be supplied to replace the default gains,
penalty templates, and advantageous classifications. Configuration is validated
when `task-design` runs and snapshotted into `design.json`.

## 2. Materialize Personas

Run `personas` once. Task execution consumes that exact batch and does not
regenerate demographics. Persona protocol assignments are copied into trial
metadata.

## 3. Resolve Hidden Schedules

Before the first provider call, resolve:

- Every gain and penalty sequence needed by the configured trial count.
- Shared or per-subject schedule seeds.
- A deterministic per-subject mapping from visible labels to internal
  contingencies.

Persist resolved schedules under `schedules/`. Never include internal IDs,
advantage classifications, future outcomes, or the recognized task name in
model messages.

## 4. Execute Trials

Implementations satisfy the `BehavioralTask` protocol in
`behavioral_tasks/base.py`:

- `instruction()` gives task-neutral instructions.
- `initial_state()` creates the environment state.
- `observe()` returns the current prompt and allowed action IDs.
- `apply_action()` performs one deterministic transition.
- `is_complete()` controls termination.
- `summarize()` derives subject metrics.

The persona prompt remains the first system message. Instructions, observations,
selections, and feedback form a compact full ledger. An invalid selection gets
one corrective retry without advancing state. Provider failures and repeated
invalid selections stop that subject.

Trials for one subject must remain sequential. Batch execution may use bounded
concurrency across subjects.

## 5. Resume Safely

A completed `TaskTrialRecord` is the commit boundary. Resume loads completed
records, replays each action against the persisted schedule, and rejects the
ledger if any recorded transition differs from the deterministic replay.

Do not infer state from chat text. Chat files are an audit surface; trial
records and schedules are the canonical state.

## 6. Analyze and Export

`task-analyze` derives subject summaries from completed transitions. For the
four-deck task these include deck counts, advantageous-minus-disadvantageous
choice scores, block scores, outcomes, and final balance. `task-results` exports
flat summary and trial CSV files.

Provider latency may be retained as operational metadata, but it is not a human
reaction-time measure.

## Adding Another Task

For a sequential-offer task such as Risky Gains:

1. Define a validated configuration and fully resolved hidden schedule.
2. Define state fields for trial, current offer, accepted amount, and outcome.
3. Make `observe()` expose only the current offer and valid actions.
4. Make `apply_action()` advance within a trial or settle the trial.
5. Persist one record per experimental analysis unit.
6. Add deterministic transition, masking, retry, resume, and summary tests.

Keep task-specific logic in its task module. Extend the shared engine only when
another task demonstrates the same requirement.
