from llm_psych_scales.client import _is_selected_answer_allowed


def test_selected_answer_allows_exact_option_id() -> None:
    assert _is_selected_answer_allowed("2", ["1", "2", "3"])


def test_selected_answer_allows_empty_constraints_for_free_response() -> None:
    assert _is_selected_answer_allowed("open text", [])


def test_selected_answer_allows_comma_separated_multiple_choice_ids() -> None:
    assert _is_selected_answer_allowed("a, c", ["a", "b", "c"])


def test_selected_answer_rejects_unknown_option_id() -> None:
    assert not _is_selected_answer_allowed("z", ["a", "b", "c"])
