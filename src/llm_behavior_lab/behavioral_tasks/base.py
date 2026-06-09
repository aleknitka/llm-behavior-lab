from __future__ import annotations

from typing import Any, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field

from llm_behavior_lab.responses.base import ResponseStatus


class TaskObservation(BaseModel):
    """Model-facing description of the current decision."""

    model_config = ConfigDict(extra="forbid")

    trial_index: int = Field(ge=1)
    prompt: str
    allowed_action_ids: list[str] = Field(min_length=1)


class TaskState(BaseModel):
    """Generic state shared by repeated-choice monetary tasks."""

    model_config = ConfigDict(extra="forbid")

    trial_index: int = Field(default=0, ge=0)
    balance: int | float
    action_counts: dict[str, int] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskTransition(BaseModel):
    """Deterministic result of applying one valid action."""

    model_config = ConfigDict(extra="forbid")

    trial_index: int = Field(ge=1)
    visible_action_id: str
    internal_action_id: str
    gain: int | float = 0
    penalty: int | float = 0
    net: int | float
    feedback: str
    state: TaskState
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskAttemptRecord(BaseModel):
    """One provider attempt to select an action for a trial."""

    model_config = ConfigDict(extra="forbid")

    attempt: int = Field(ge=1)
    status: ResponseStatus
    selected_action_id: str | None = None
    raw_response: str | None = None
    structured_response: dict[str, Any] | None = None
    logprobs: dict[str, Any] | None = None
    error: str | None = None


class TaskTrialRecord(BaseModel):
    """Append-only analysis unit for one task trial."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    session_id: str
    run_id: str
    subject_id: str
    task_id: str
    task_version: str
    schedule_id: str
    trial_index: int = Field(ge=1)
    observation: TaskObservation
    attempts: list[TaskAttemptRecord] = Field(min_length=1)
    transition: TaskTransition | None = None
    message_start_index: int = Field(ge=0)
    message_end_index: int = Field(ge=0)
    status: ResponseStatus
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskBlockSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_trial: int
    end_trial: int
    advantageous_choices: int
    disadvantageous_choices: int
    advantageous_choice_score: int


class TaskSummaryRecord(BaseModel):
    """Subject-level metrics derived from completed task transitions."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    task_version: str
    subject_id: str | None = None
    schedule_id: str | None = None
    trial_count: int
    final_balance: int | float
    net_earnings: int | float
    total_gain: int | float
    total_penalty: int | float
    advantageous_choice_score: int
    internal_action_counts: dict[str, int]
    visible_action_counts: dict[str, int]
    blocks: list[TaskBlockSummary]
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskRunResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    records: list[TaskTrialRecord]
    summary: TaskSummaryRecord | None = None
    status: ResponseStatus


class BehavioralTask(Protocol):
    """Minimal interface for a deterministic stateful behavioral task.

    Implementations own the environment. The LLM receives an observation and
    selects an allowed action; it never computes rewards or mutates state.
    """

    id: str
    version: str

    def instruction(self) -> str: ...

    def initial_state(self, schedule: BaseModel) -> TaskState: ...

    def observe(self, state: TaskState, schedule: BaseModel) -> TaskObservation: ...

    def apply_action(
        self, state: TaskState, action_id: str, schedule: BaseModel
    ) -> TaskTransition: ...

    def is_complete(self, state: TaskState) -> bool: ...

    def summarize(
        self, transitions: list[TaskTransition], block_size: int = 20
    ) -> TaskSummaryRecord: ...

    def model_copy(self, *args: Any, **kwargs: Any) -> Self: ...
