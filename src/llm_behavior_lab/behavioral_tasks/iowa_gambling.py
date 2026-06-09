from __future__ import annotations

import hashlib
import itertools
import random
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llm_behavior_lab.behavioral_tasks.base import (
    TaskBlockSummary,
    TaskObservation,
    TaskState,
    TaskSummaryRecord,
    TaskTransition,
)


class ScheduleAssignment(StrEnum):
    SHARED = "shared"
    PER_SUBJECT = "per_subject"


class ScheduleMode(StrEnum):
    TEMPLATE = "template"
    FIXED = "fixed"


class DeckContingency(BaseModel):
    """Hidden gain and penalty sequence for one internal deck."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    advantageous: bool
    gain: int | float
    penalties: list[int | float] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_penalties(self) -> DeckContingency:
        if any(value > 0 for value in self.penalties):
            raise ValueError("penalties must be zero or negative")
        return self


def _classic_contingencies() -> list[DeckContingency]:
    return [
        DeckContingency(
            id="deck_a",
            advantageous=False,
            gain=100,
            penalties=[0, 0, -150, 0, -300, 0, -200, 0, -250, -350],
        ),
        DeckContingency(
            id="deck_b",
            advantageous=False,
            gain=100,
            penalties=[0, 0, 0, 0, 0, 0, 0, 0, 0, -1250],
        ),
        DeckContingency(
            id="deck_c",
            advantageous=True,
            gain=50,
            penalties=[0, -50, 0, -50, 0, -50, 0, -50, 0, -50],
        ),
        DeckContingency(
            id="deck_d",
            advantageous=True,
            gain=50,
            penalties=[0, 0, 0, 0, 0, 0, 0, 0, 0, -250],
        ),
    ]


class IowaGamblingConfig(BaseModel):
    """Editable configuration for a four-contingency card-choice task."""

    model_config = ConfigDict(extra="forbid")

    trial_count: int = Field(default=100, ge=1)
    starting_balance: int | float = 2000
    schedule_mode: ScheduleMode = ScheduleMode.TEMPLATE
    schedule_assignment: ScheduleAssignment = ScheduleAssignment.SHARED
    shuffle_template_blocks: bool = False
    visible_labels: list[str] = Field(
        default_factory=lambda: ["Circle", "Square", "Triangle", "Star"],
        min_length=4,
        max_length=4,
    )
    decks: list[DeckContingency] = Field(default_factory=_classic_contingencies)
    feedback_fields: set[str] = Field(
        default_factory=lambda: {"gain", "penalty", "net", "balance"}
    )

    @model_validator(mode="after")
    def validate_config(self) -> IowaGamblingConfig:
        if len(self.decks) != 4:
            raise ValueError("exactly four deck contingencies are required")
        if len({deck.id for deck in self.decks}) != 4:
            raise ValueError("deck ids must be unique")
        if len(set(self.visible_labels)) != 4:
            raise ValueError("visible labels must be unique")
        valid_feedback_fields = {"gain", "penalty", "net", "balance"}
        if not self.feedback_fields <= valid_feedback_fields:
            raise ValueError("feedback_fields contains an unsupported value")
        if sum(deck.advantageous for deck in self.decks) != 2:
            raise ValueError("exactly two contingencies must be advantageous")
        if self.schedule_mode == ScheduleMode.FIXED:
            short = [deck.id for deck in self.decks if len(deck.penalties) < self.trial_count]
            if short:
                raise ValueError(
                    "fixed schedules need at least trial_count penalties for every deck: "
                    + ", ".join(short)
                )
        return self


class ResolvedIowaSchedule(BaseModel):
    """Hidden, fully resolved schedule persisted before execution."""

    model_config = ConfigDict(extra="forbid")

    id: str
    task_id: str
    task_version: str
    seed: int | None
    subject_id: str | None
    assignment: ScheduleAssignment
    trial_count: int
    starting_balance: int | float
    label_mapping: dict[str, str]
    gains: dict[str, int | float]
    penalties: dict[str, list[int | float]]
    advantageous: dict[str, bool]
    feedback_fields: set[str]


class IowaGamblingTask:
    """Stateful four-choice task with hidden, configurable contingencies."""

    id = "four-deck-card-task"
    version = "1.0"

    def __init__(self, config: IowaGamblingConfig | None = None) -> None:
        self.config = config or IowaGamblingConfig()

    def instruction(self) -> str:
        labels = ", ".join(self.config.visible_labels)
        return (
            "You will repeatedly choose one of four card options. Your goal is to "
            f"maximize your final balance. Reply with exactly one option label: {labels}. "
            "After each choice you will be told its monetary outcome."
        )

    def resolve_schedule(
        self, *, seed: int | None, subject_id: str
    ) -> ResolvedIowaSchedule:
        schedule_subject = (
            subject_id
            if self.config.schedule_assignment == ScheduleAssignment.PER_SUBJECT
            else None
        )
        schedule_seed = _derived_seed(seed, "schedule", schedule_subject or "shared")
        penalties = {
            deck.id: self._resolve_penalties(deck, schedule_seed)
            for deck in self.config.decks
        }
        label_mapping = self._label_mapping(seed, subject_id)
        digest = hashlib.sha256(
            repr(
                (
                    self.version,
                    schedule_seed,
                    subject_id,
                    label_mapping,
                    penalties,
                )
            ).encode()
        ).hexdigest()[:16]
        return ResolvedIowaSchedule(
            id=f"schedule-{digest}",
            task_id=self.id,
            task_version=self.version,
            seed=schedule_seed,
            subject_id=schedule_subject,
            assignment=self.config.schedule_assignment,
            trial_count=self.config.trial_count,
            starting_balance=self.config.starting_balance,
            label_mapping=label_mapping,
            gains={deck.id: deck.gain for deck in self.config.decks},
            penalties=penalties,
            advantageous={deck.id: deck.advantageous for deck in self.config.decks},
            feedback_fields=set(self.config.feedback_fields),
        )

    def _resolve_penalties(
        self, deck: DeckContingency, schedule_seed: int
    ) -> list[int | float]:
        if self.config.schedule_mode == ScheduleMode.FIXED:
            return list(deck.penalties[: self.config.trial_count])

        output: list[int | float] = []
        block_index = 0
        while len(output) < self.config.trial_count:
            block = list(deck.penalties)
            if self.config.shuffle_template_blocks:
                random.Random(  # nosec B311 - reproducible experiment scheduling.
                    _derived_seed(schedule_seed, deck.id, str(block_index))
                ).shuffle(block)
            output.extend(block)
            block_index += 1
        return output[: self.config.trial_count]

    def _label_mapping(self, seed: int | None, subject_id: str) -> dict[str, str]:
        deck_ids = [deck.id for deck in self.config.decks]
        permutations = list(itertools.permutations(deck_ids))
        index = _derived_seed(seed, "labels", subject_id) % len(permutations)
        return dict(zip(self.config.visible_labels, permutations[index], strict=True))

    def initial_state(self, schedule: ResolvedIowaSchedule) -> TaskState:
        return TaskState(
            balance=schedule.starting_balance,
            action_counts={deck_id: 0 for deck_id in schedule.gains},
        )

    def observe(
        self, state: TaskState, schedule: ResolvedIowaSchedule
    ) -> TaskObservation:
        next_trial = state.trial_index + 1
        labels = ", ".join(schedule.label_mapping)
        return TaskObservation(
            trial_index=next_trial,
            prompt=(
                f"Trial {next_trial} of {schedule.trial_count}. "
                f"Current balance: {state.balance}. Choose one option: {labels}."
            ),
            allowed_action_ids=list(schedule.label_mapping),
        )

    def apply_action(
        self,
        state: TaskState,
        action_id: str,
        schedule: ResolvedIowaSchedule,
    ) -> TaskTransition:
        if action_id not in schedule.label_mapping:
            raise ValueError(f"unknown action id: {action_id}")
        if self.is_complete(state):
            raise ValueError("task is already complete")

        internal_id = schedule.label_mapping[action_id]
        draw_index = state.action_counts.get(internal_id, 0)
        if draw_index >= len(schedule.penalties[internal_id]):
            raise ValueError(f"schedule exhausted for {internal_id}")
        gain = schedule.gains[internal_id]
        penalty = schedule.penalties[internal_id][draw_index]
        net = gain + penalty
        counts = dict(state.action_counts)
        counts[internal_id] = draw_index + 1
        next_state = TaskState(
            trial_index=state.trial_index + 1,
            balance=state.balance + net,
            action_counts=counts,
            metadata=dict(state.metadata),
        )
        feedback = self._feedback(action_id, next_state, gain, penalty, net, schedule)
        return TaskTransition(
            trial_index=next_state.trial_index,
            visible_action_id=action_id,
            internal_action_id=internal_id,
            gain=gain,
            penalty=penalty,
            net=net,
            feedback=feedback,
            state=next_state,
            metadata={"advantageous": schedule.advantageous[internal_id]},
        )

    def _feedback(
        self,
        action_id: str,
        state: TaskState,
        gain: int | float,
        penalty: int | float,
        net: int | float,
        schedule: ResolvedIowaSchedule,
    ) -> str:
        parts = [f"Trial {state.trial_index}: you selected {action_id}."]
        if "gain" in schedule.feedback_fields:
            parts.append(f"Gain: {gain}.")
        if "penalty" in schedule.feedback_fields:
            parts.append(f"Loss: {abs(penalty)}.")
        if "net" in schedule.feedback_fields:
            parts.append(f"Net outcome: {net}.")
        if "balance" in schedule.feedback_fields:
            parts.append(f"Running balance: {state.balance}.")
        return " ".join(parts)

    def is_complete(self, state: TaskState) -> bool:
        return state.trial_index >= self.config.trial_count

    def summarize(
        self, transitions: list[TaskTransition], block_size: int = 20
    ) -> TaskSummaryRecord:
        if block_size < 1:
            raise ValueError("block_size must be at least 1")
        internal_counts = {deck.id: 0 for deck in self.config.decks}
        visible_counts = {label: 0 for label in self.config.visible_labels}
        advantageous_ids = {deck.id for deck in self.config.decks if deck.advantageous}
        for transition in transitions:
            internal_counts[transition.internal_action_id] += 1
            visible_counts[transition.visible_action_id] += 1

        blocks: list[TaskBlockSummary] = []
        for start in range(0, len(transitions), block_size):
            chunk = transitions[start : start + block_size]
            advantageous = sum(
                item.internal_action_id in advantageous_ids for item in chunk
            )
            disadvantageous = len(chunk) - advantageous
            blocks.append(
                TaskBlockSummary(
                    start_trial=start + 1,
                    end_trial=start + len(chunk),
                    advantageous_choices=advantageous,
                    disadvantageous_choices=disadvantageous,
                    advantageous_choice_score=advantageous - disadvantageous,
                )
            )

        advantageous_total = sum(
            count for deck_id, count in internal_counts.items() if deck_id in advantageous_ids
        )
        disadvantageous_total = len(transitions) - advantageous_total
        final_balance = (
            transitions[-1].state.balance
            if transitions
            else self.config.starting_balance
        )
        return TaskSummaryRecord(
            task_id=self.id,
            task_version=self.version,
            trial_count=len(transitions),
            final_balance=final_balance,
            net_earnings=final_balance - self.config.starting_balance,
            total_gain=sum(item.gain for item in transitions),
            total_penalty=sum(item.penalty for item in transitions),
            advantageous_choice_score=advantageous_total - disadvantageous_total,
            internal_action_counts=internal_counts,
            visible_action_counts=visible_counts,
            blocks=blocks,
        )


def _derived_seed(seed: int | None, *parts: str) -> int:
    digest = hashlib.sha256(
        ":".join([str(seed), *parts]).encode()
    ).hexdigest()
    return int(digest[:16], 16)
