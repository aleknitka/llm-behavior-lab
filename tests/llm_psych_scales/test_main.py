from pathlib import Path

import pytest

from llm_psych_scales.main import build_parser, parse_features


def test_parse_features_converts_key_value_pairs() -> None:
    features = parse_features(["age=35", "country=Poland"])

    assert features == {"age": "35", "country": "Poland"}


def test_parse_features_rejects_missing_separator() -> None:
    with pytest.raises(ValueError, match="Expected KEY=VALUE"):
        parse_features(["age"])


def test_parser_defaults_to_bfi10_and_local_model() -> None:
    args = build_parser().parse_args([])

    assert args.model == "openai/gpt-oss-20b"
    assert args.base_url == "http://localhost:1234/v1"
    assert args.api_key == "lm-studio"
    assert args.project_root == Path(".")
    assert args.experiment_id is None
    assert args.persona_count == 100
    assert args.seed is None
    assert args.log_level == "INFO"


def test_parser_accepts_log_level() -> None:
    args = build_parser().parse_args(["--log-level", "DEBUG"])

    assert args.log_level == "DEBUG"


def test_parser_rejects_old_output_argument() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--output", "runs/questionnaire_responses.jsonl"])
