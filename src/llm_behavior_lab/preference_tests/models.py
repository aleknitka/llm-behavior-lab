from __future__ import annotations

import itertools
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from llm_behavior_lab.responses.base import ChatMessage, ResponseStatus


class Stimulus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    label: str | None = None
    text: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PairwiseTrial(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    stimulus_ids: tuple[str, str]
    order: int = Field(default=1, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("stimulus_ids")
    @classmethod
    def validate_stimulus_ids(cls, value: tuple[str, str]) -> tuple[str, str]:
        if len(set(value)) != 2:
            raise ValueError("pairwise trial must compare two unique stimuli")
        return value


class PairwisePreferenceExperiment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    stimuli: list[Stimulus] = Field(min_length=2)
    trials: list[PairwiseTrial] = Field(default_factory=list)
    instruction: str = "Choose the version you prefer more."
    reference: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_experiment(self) -> PairwisePreferenceExperiment:
        stimulus_ids = [stimulus.id for stimulus in self.stimuli]
        if len(set(stimulus_ids)) != len(stimulus_ids):
            raise ValueError("stimulus ids must be unique")

        if not self.trials:
            self.trials = generate_pairwise_trials(stimulus_ids)

        known_ids = set(stimulus_ids)
        unknown_ids = sorted(
            {
                stimulus_id
                for trial in self.trials
                for stimulus_id in trial.stimulus_ids
                if stimulus_id not in known_ids
            }
        )
        if unknown_ids:
            raise ValueError(f"trials reference unknown stimulus ids: {', '.join(unknown_ids)}")

        trial_ids = [trial.id for trial in self.trials]
        if len(set(trial_ids)) != len(trial_ids):
            raise ValueError("trial ids must be unique")
        return self

    def stimulus_by_id(self, stimulus_id: str) -> Stimulus:
        for stimulus in self.stimuli:
            if stimulus.id == stimulus_id:
                return stimulus
        raise KeyError(stimulus_id)


class PairwisePreferenceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject_id: str
    session_id: str
    run_id: str
    preference_experiment_id: str
    preference_experiment_version: str
    trial_id: str
    trial_order: int
    stimulus_ids: tuple[str, str]
    displayed_stimulus_ids: tuple[str, str]
    selected_label: str | None = None
    selected_stimulus_id: str | None = None
    rejected_stimulus_id: str | None = None
    messages: list[ChatMessage]
    raw_response: str | None = None
    structured_response: dict[str, Any] | None = None
    logprobs: dict[str, Any] | None = None
    status: ResponseStatus
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def generate_pairwise_trials(stimulus_ids: list[str]) -> list[PairwiseTrial]:
    if len(set(stimulus_ids)) != len(stimulus_ids):
        raise ValueError("stimulus ids must be unique")
    if len(stimulus_ids) < 2:
        raise ValueError("at least two stimulus ids are required")
    return [
        PairwiseTrial(id=f"{left}__{right}", stimulus_ids=(left, right), order=index)
        for index, (left, right) in enumerate(itertools.combinations(stimulus_ids, 2), start=1)
    ]
