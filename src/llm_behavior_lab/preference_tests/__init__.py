"""Pairwise preference-test tooling."""

from llm_behavior_lab.preference_tests.analysis import (
    load_pairwise_preference_rows,
    summarize_pairwise_preferences,
)
from llm_behavior_lab.preference_tests.models import (
    PairwisePreferenceExperiment,
    PairwisePreferenceRecord,
    PairwiseTrial,
    Stimulus,
    generate_pairwise_trials,
)
from llm_behavior_lab.preference_tests.prompting import render_pairwise_preference_prompt
from llm_behavior_lab.preference_tests.runner import (
    PreferenceBatchRunResult,
    run_pairwise_preference_batch,
    run_pairwise_preference_test,
)

__all__ = [
    "PairwisePreferenceExperiment",
    "PairwisePreferenceRecord",
    "PairwiseTrial",
    "PreferenceBatchRunResult",
    "Stimulus",
    "generate_pairwise_trials",
    "load_pairwise_preference_rows",
    "render_pairwise_preference_prompt",
    "run_pairwise_preference_batch",
    "run_pairwise_preference_test",
    "summarize_pairwise_preferences",
]
