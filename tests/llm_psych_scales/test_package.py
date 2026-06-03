from llm_psych_scales import __version__, run_questionnaire, run_questionnaire_async


def test_package_exports_version_and_runners() -> None:
    assert __version__ == "0.1.0"
    assert callable(run_questionnaire)
    assert callable(run_questionnaire_async)
