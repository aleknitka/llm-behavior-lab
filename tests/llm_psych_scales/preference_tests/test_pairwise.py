import json
from collections.abc import Sequence

from llm_psych_scales.models import LlmQuestionResult, ModelSettings, Persona
from llm_psych_scales.preference_tests import (
    PairwisePreferenceExperiment,
    PairwisePreferenceRecord,
    PairwiseTrial,
    Stimulus,
    generate_pairwise_trials,
    load_pairwise_preference_rows,
    render_pairwise_preference_prompt,
    summarize_pairwise_preferences,
)
from llm_psych_scales.preference_tests.runner import (
    run_pairwise_preference_batch,
    run_pairwise_preference_test,
)
from llm_psych_scales.responses.base import ResponseStatus

TEST_PERSONA = Persona(
    persona_id="persona-1",
    features={"age": "35", "country": "Poland"},
)


def _experiment() -> PairwisePreferenceExperiment:
    return PairwisePreferenceExperiment(
        id="landing-copy-test",
        name="Landing copy test",
        version="1.0",
        instruction="Choose the message you would be more likely to click.",
        stimuli=[
            Stimulus(id="direct", label="Direct", text="Start saving time today."),
            Stimulus(id="social", label="Social proof", text="Join 10,000 teams saving time."),
            Stimulus(id="calm", label="Calm", text="A quieter way to manage your work."),
        ],
        trials=generate_pairwise_trials(["direct", "social", "calm"]),
    )


def test_generate_pairwise_trials_creates_unordered_pairs() -> None:
    trials = generate_pairwise_trials(["a", "b", "c"])

    assert [trial.id for trial in trials] == ["a__b", "a__c", "b__c"]
    assert [trial.stimulus_ids for trial in trials] == [
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
    ]


def test_experiment_rejects_trials_with_unknown_stimulus() -> None:
    try:
        PairwisePreferenceExperiment(
            id="bad-test",
            name="Bad test",
            version="1.0",
            stimuli=[
                Stimulus(id="a", text="A"),
                Stimulus(id="b", text="B"),
            ],
            trials=[PairwiseTrial(id="bad", stimulus_ids=("a", "missing"), order=1)],
        )
    except ValueError as exc:
        assert "unknown stimulus ids" in str(exc)
    else:
        raise AssertionError("Expected unknown stimulus validation error")


def test_render_pairwise_preference_prompt_blinds_stimuli() -> None:
    experiment = _experiment()

    prompt = render_pairwise_preference_prompt(
        experiment=experiment,
        trial=experiment.trials[0],
        displayed_stimulus_ids=("social", "direct"),
    )

    assert "Choose the message you would be more likely to click." in prompt
    assert "A. Join 10,000 teams saving time." in prompt
    assert "B. Start saving time today." in prompt
    assert "Reply with exactly one answer id: A or B." in prompt
    assert "Direct" not in prompt
    assert "Social proof" not in prompt


class FakePreferenceClient:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []
        self.allowed_answer_ids: list[list[str]] = []
        self.seeds: list[int | None] = []

    def complete(
        self,
        messages: Sequence[dict[str, str]],
        settings: ModelSettings,
        allowed_answer_ids: Sequence[str],
    ) -> LlmQuestionResult:
        self.calls.append(list(messages))
        self.allowed_answer_ids.append(list(allowed_answer_ids))
        self.seeds.append(settings.seed)
        return LlmQuestionResult(selected_answer_id="A", raw_response="A")


def test_run_pairwise_preference_test_writes_records_and_maps_choice(tmp_path) -> None:
    experiment = _experiment()
    settings = ModelSettings(
        model="llama3.1",
        provider_base_url="http://localhost:11434/v1",
        temperature=0.2,
        timeout_seconds=60.0,
        seed=123,
    )
    client = FakePreferenceClient()

    records = run_pairwise_preference_test(
        persona=TEST_PERSONA,
        experiment=experiment,
        settings=settings,
        client=client,
        project_root=tmp_path,
        experiment_id="pref-study-one",
    )

    assert len(records) == 3
    assert {tuple(ids) for ids in client.allowed_answer_ids} == {("A", "B")}
    assert len(set(client.seeds)) == len(experiment.trials)
    assert records[0].run_id.startswith("run-pref-landing-copy-test-llama3-1-")
    assert records[0].status == ResponseStatus.COMPLETED
    assert records[0].selected_label == "A"
    assert records[0].selected_stimulus_id == records[0].displayed_stimulus_ids[0]
    assert records[0].rejected_stimulus_id == records[0].displayed_stimulus_ids[1]
    assert records[0].metadata["seed"] == client.seeds[0]

    run_root = tmp_path / "experiments" / "pref-study-one" / records[0].run_id
    response_path = run_root / "responses" / "persona-1.jsonl"
    rows = [json.loads(line) for line in response_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    assert rows[0]["selected_stimulus_id"] == records[0].selected_stimulus_id
    assert (run_root / "experiment.json").exists()
    assert (run_root / "run.jsonl").exists()


def test_run_pairwise_preference_test_marks_invalid_answers(tmp_path) -> None:
    class InvalidClient(FakePreferenceClient):
        def complete(
            self,
            messages: Sequence[dict[str, str]],
            settings: ModelSettings,
            allowed_answer_ids: Sequence[str],
        ) -> LlmQuestionResult:
            return LlmQuestionResult(selected_answer_id="C", raw_response="C")

    records = run_pairwise_preference_test(
        persona=TEST_PERSONA,
        experiment=_experiment(),
        settings=ModelSettings(
            model="llama3.1",
            provider_base_url="http://localhost:11434/v1",
            temperature=0.2,
            timeout_seconds=60.0,
        ),
        client=InvalidClient(),
        project_root=tmp_path,
        experiment_id="pref-study-one",
    )

    assert {record.status for record in records} == {ResponseStatus.INVALID}
    assert {record.selected_stimulus_id for record in records} == {None}


def test_run_pairwise_preference_batch_writes_one_run_for_generated_personas(tmp_path) -> None:
    experiment = _experiment()
    settings = ModelSettings(
        model="llama3.1",
        provider_base_url="http://localhost:11434/v1",
        temperature=0.2,
        timeout_seconds=60.0,
        seed=123,
    )

    result = run_pairwise_preference_batch(
        experiment=experiment,
        settings=settings,
        client=FakePreferenceClient(),
        project_root=tmp_path,
        experiment_id="pref-study-one",
        persona_count=2,
        seed=123,
    )

    assert result.experiment_id == "pref-study-one"
    assert len(result.personas.personas) == 2
    assert len(result.runs) == 1
    assert result.runs[0].item_count == len(experiment.trials) * 2
    run_root = tmp_path / "experiments" / "pref-study-one" / result.runs[0].run_id
    assert (run_root / "run.jsonl").read_text(encoding="utf-8").count("\n") == 1
    assert len(list((run_root / "responses").glob("*.jsonl"))) == 2
    assert (tmp_path / "experiments" / "pref-study-one" / "personas.jsonl").exists()


def test_summarize_pairwise_preferences_counts_wins(tmp_path) -> None:
    record_a = PairwisePreferenceRecord(
        subject_id="p1",
        session_id="session-1",
        run_id="run-pref-demo",
        preference_experiment_id="demo",
        preference_experiment_version="1.0",
        trial_id="a__b",
        trial_order=1,
        stimulus_ids=("a", "b"),
        displayed_stimulus_ids=("a", "b"),
        selected_label="A",
        selected_stimulus_id="a",
        rejected_stimulus_id="b",
        messages=[],
        raw_response="A",
        status=ResponseStatus.COMPLETED,
    )
    record_b = record_a.model_copy(
        update={
            "subject_id": "p2",
            "selected_label": "B",
            "selected_stimulus_id": "b",
            "rejected_stimulus_id": "a",
        }
    )
    path = tmp_path / "responses" / "records.jsonl"
    path.parent.mkdir()
    path.write_text(
        record_a.model_dump_json() + "\n" + record_b.model_dump_json() + "\n",
        encoding="utf-8",
    )

    rows = load_pairwise_preference_rows(path)
    summary = summarize_pairwise_preferences(rows)

    assert summary["stimulus_wins"] == {"a": 1, "b": 1}
    assert summary["pair_counts"] == {"a__b": {"a": 1, "b": 1}}
