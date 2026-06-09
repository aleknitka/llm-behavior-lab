from llm_behavior_lab.behavioral_tasks.iowa_gambling import (
    DeckContingency,
    IowaGamblingConfig,
    IowaGamblingTask,
    ScheduleAssignment,
)


def small_config() -> IowaGamblingConfig:
    return IowaGamblingConfig(
        trial_count=4,
        starting_balance=2000,
        schedule_assignment=ScheduleAssignment.PER_SUBJECT,
        visible_labels=["Circle", "Square", "Triangle", "Star"],
        decks=[
            DeckContingency(
                id="deck_a",
                advantageous=False,
                gain=100,
                penalties=[0, -150],
            ),
            DeckContingency(
                id="deck_b",
                advantageous=False,
                gain=100,
                penalties=[0, -200],
            ),
            DeckContingency(
                id="deck_c",
                advantageous=True,
                gain=50,
                penalties=[0, -25],
            ),
            DeckContingency(
                id="deck_d",
                advantageous=True,
                gain=50,
                penalties=[0, -50],
            ),
        ],
    )


def test_task_masks_name_and_internal_deck_ids() -> None:
    task = IowaGamblingTask(small_config())
    schedule = task.resolve_schedule(seed=7, subject_id="subject-1")
    state = task.initial_state(schedule)

    observation = task.observe(state, schedule)
    prompt = f"{task.instruction()}\n{observation.prompt}"

    assert "Iowa" not in prompt
    assert "deck_a" not in prompt
    assert set(observation.allowed_action_ids) == set(schedule.label_mapping)


def test_schedule_and_label_mapping_are_deterministic_per_subject() -> None:
    task = IowaGamblingTask(small_config())

    first = task.resolve_schedule(seed=11, subject_id="subject-1")
    repeated = task.resolve_schedule(seed=11, subject_id="subject-1")
    second_subject = task.resolve_schedule(seed=11, subject_id="subject-2")

    assert first == repeated
    assert first.id != second_subject.id
    assert first.label_mapping != second_subject.label_mapping


def test_action_applies_hidden_contingency_and_updates_balance() -> None:
    task = IowaGamblingTask(small_config())
    schedule = task.resolve_schedule(seed=3, subject_id="subject-1")
    state = task.initial_state(schedule)
    label = next(
        visible
        for visible, internal in schedule.label_mapping.items()
        if internal == "deck_c"
    )

    transition = task.apply_action(state, label, schedule)

    assert transition.internal_action_id == "deck_c"
    assert transition.gain == 50
    assert transition.penalty in {0, -25}
    assert transition.state.balance == 2000 + transition.net
    assert transition.state.trial_index == 1
    assert "deck_c" not in transition.feedback


def test_summary_reports_advantageous_choice_score_by_block() -> None:
    task = IowaGamblingTask(small_config())
    schedule = task.resolve_schedule(seed=5, subject_id="subject-1")
    state = task.initial_state(schedule)
    internal_choices = ["deck_a", "deck_c", "deck_d", "deck_b"]
    transitions = []
    for internal_id in internal_choices:
        label = next(
            visible
            for visible, internal in schedule.label_mapping.items()
            if internal == internal_id
        )
        transition = task.apply_action(state, label, schedule)
        transitions.append(transition)
        state = transition.state

    summary = task.summarize(transitions, block_size=2)

    assert summary.trial_count == 4
    assert summary.advantageous_choice_score == 0
    assert [block.advantageous_choice_score for block in summary.blocks] == [0, 0]
    assert summary.final_balance == state.balance
