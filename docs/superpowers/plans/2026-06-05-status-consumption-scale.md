# Status Consumption Scale Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Status Consumption Scale from Eastman, Goldsmith, and Flynn (1999) as a validated questionnaire definition.

**Architecture:** Add SCS as one static questionnaire module using the existing Pydantic questionnaire models. Preserve source item wording, factor loadings, reverse scoring, response bounds, citation, DOI, and psychometric metadata in the questionnaire definition; do not add executable scoring behavior beyond scale mappings.

**Tech Stack:** Python, Pydantic questionnaire models, pytest, ruff, ty, uv.

---

## Source

Use `.local_sources/Status Consumption in Consumer Behavior  Scale Development and Validation.pdf`.

Source article:

Eastman, J. K., Goldsmith, R. E., & Flynn, L. R. (1999). Status Consumption in Consumer Behavior: Scale Development and Validation. Journal of Marketing Theory and Practice, 7(3), 41-52. DOI: `10.1080/10696679.1999.11501839`.

The source PDF provides:

- A five-item final Status Consumption Scale in Table 1.
- One unidimensional status consumption construct.
- Study Two factor loadings for the five final items.
- A seven-point Likert-type response format.
- One negatively worded item: "The status of a product is irrelevant to me".
- Reliability and validation metadata across six studies.

The extracted PDF text does not show explicit seven-point anchor labels. Implement the response format with `min_value=1`, `max_value=7`, and no labels; store a metadata note that the source PDF states "seven-point, Likert-type" but does not specify anchors in the available text.

## File Structure

- Create: `src/llm_behavior_lab/questionnaires/status_consumption/__init__.py`
  - Export `STATUS_CONSUMPTION_SCALE` and alias `SCS`.
- Create: `src/llm_behavior_lab/questionnaires/status_consumption/definition.py`
  - Define source constants, response format, item definitions, scale mapping, and the `Questionnaire`.
- Create: `src/llm_behavior_lab/questionnaires/status_consumption/README.md`
  - Document the source, response format, construct, reverse-scored item, validation summary, and scoring-model deferral.
- Modify: `src/llm_behavior_lab/questionnaires/__init__.py`
  - Export `STATUS_CONSUMPTION_SCALE` and `SCS`.
- Modify: `README.md`
  - Add a link to the SCS README in the "Questionnaire Definitions" list.
- Create: `tests/llm_behavior_lab/questionnaires/test_status_consumption.py`
  - Cover identity, source wording, response format, scale mappings, metadata, exports, README link, and Pydantic validation.

Do not commit `.local_sources/`, PDFs, `Zone.Identifier` files, or temporary extraction artifacts.

## Implementation Tasks

### Task 1: Add Failing Tests

**Files:**

- Create: `tests/llm_behavior_lab/questionnaires/test_status_consumption.py`

- [ ] Write tests for questionnaire identity and validation.

Expected assertions:

```python
from pathlib import Path

from llm_behavior_lab.questionnaires import SCS
from llm_behavior_lab.questionnaires.base.response_formats import LikertFormat
from llm_behavior_lab.questionnaires.base.scale import Questionnaire
from llm_behavior_lab.questionnaires.status_consumption import STATUS_CONSUMPTION_SCALE


def test_status_consumption_validates_and_exports_alias() -> None:
    questionnaire = STATUS_CONSUMPTION_SCALE

    assert isinstance(questionnaire, Questionnaire)
    assert SCS is questionnaire
    assert Questionnaire.model_validate(questionnaire.model_dump()) == questionnaire
    assert questionnaire.id == "status_consumption_scale"
    assert questionnaire.shorthand == "scs"
    assert questionnaire.name == "Status Consumption Scale"
    assert questionnaire.version == "1.0"
    assert questionnaire.language == "en"
```

- [ ] Write tests for item order, wording, source codes, and factor loadings.

Expected item data:

```python
SOURCE_ITEMS = [
    ("1", "I would buy a product just because it has status.", 0.78),
    ("2", "I am interested in new products with status.", 0.80),
    ("3", "I would pay more for a product if it had status.", 0.83),
    ("4", "The status of a product is irrelevant to me.", -0.70),
    ("5", "A product is more valuable to me if it has some snob appeal.", 0.59),
]
```

Expected assertions:

```python
def test_status_consumption_uses_source_order_wording_and_codes() -> None:
    questionnaire = STATUS_CONSUMPTION_SCALE

    assert len(questionnaire.items) == 5
    assert questionnaire.sections[0].item_ids == [item.id for item in questionnaire.items]
    assert [item.order for item in questionnaire.items] == [1, 2, 3, 4, 5]
    assert [(item.code, item.text, item.metadata["factor_loading"]) for item in questionnaire.items] == SOURCE_ITEMS
```

- [ ] Write tests for the seven-point Likert response format.

Expected assertions:

```python
def test_status_consumption_uses_source_likert_format() -> None:
    response_format = STATUS_CONSUMPTION_SCALE.items[0].response_format

    assert isinstance(response_format, LikertFormat)
    assert response_format.min_value == 1
    assert response_format.max_value == 7
    assert response_format.labels == {}
    assert all(item.response_format == response_format for item in STATUS_CONSUMPTION_SCALE.items)
    assert STATUS_CONSUMPTION_SCALE.metadata["source_response_format"] == (
        "Seven-point Likert-type response format; source PDF text does not specify anchor labels."
    )
```

- [ ] Write tests for scale mapping and reverse scoring.

Expected assertions:

```python
def test_status_consumption_scale_mapping_preserves_reverse_scoring() -> None:
    questionnaire = STATUS_CONSUMPTION_SCALE

    assert [scale.id for scale in questionnaire.scales] == ["status_consumption"]
    mappings = questionnaire.scales[0].item_mappings
    assert [mapping.item_id for mapping in mappings] == [item.id for item in questionnaire.items]
    assert [mapping.reverse_scored for mapping in mappings] == [False, False, False, True, False]
    assert mappings[3].metadata["source_note"] == "Negatively worded item in Table 1."
```

- [ ] Write tests for source metadata and README integration.

Expected assertions:

```python
def test_status_consumption_source_metadata_and_readme_link() -> None:
    questionnaire = STATUS_CONSUMPTION_SCALE
    questionnaire_readme = Path("src/llm_behavior_lab/questionnaires/status_consumption/README.md")
    root_readme = Path("README.md")

    assert questionnaire.metadata["source_doi"] == "10.1080/10696679.1999.11501839"
    assert questionnaire.metadata["source_table"] == "Table 1"
    assert questionnaire.metadata["study_two_cronbach_alpha"] == 0.86
    assert questionnaire.metadata["study_four_test_retest_reliability"] == 0.78
    assert questionnaire.metadata["scoring_models_status"] == "deferred"
    assert "Eastman" in questionnaire.reference
    assert questionnaire.licence == "See source article terms."
    assert questionnaire_readme.exists()
    assert "10.1080/10696679.1999.11501839" in questionnaire_readme.read_text()
    assert "questionnaires/status_consumption/README.md" in root_readme.read_text()
```

- [ ] Run the new test file and verify it fails because the module does not exist yet.

```bash
uv run pytest tests/llm_behavior_lab/questionnaires/test_status_consumption.py -q
```

Expected result: import failure for `llm_behavior_lab.questionnaires.status_consumption`.

### Task 2: Implement the Questionnaire Module

**Files:**

- Create: `src/llm_behavior_lab/questionnaires/status_consumption/__init__.py`
- Create: `src/llm_behavior_lab/questionnaires/status_consumption/definition.py`

- [ ] Create `definition.py` following the static questionnaire style in `bfi10/definition.py` and `pdmi/definition.py`.

Use these IDs and constants:

```python
STATUS_CONSUMPTION_SOURCE_DOI = "10.1080/10696679.1999.11501839"
STATUS_CONSUMPTION_SOURCE_URL = "https://doi.org/10.1080/10696679.1999.11501839"
STATUS_CONSUMPTION_CITATION = (
    "Eastman, J. K., Goldsmith, R. E., & Flynn, L. R. (1999). Status Consumption "
    "in Consumer Behavior: Scale Development and Validation. Journal of Marketing "
    "Theory and Practice, 7(3), 41-52."
)
STATUS_CONSUMPTION_LICENCE = "See source article terms."
```

Use a shared response format:

```python
STATUS_CONSUMPTION_RESPONSE_FORMAT = LikertFormat(min_value=1, max_value=7)
```

Use stable item IDs:

```python
_ITEM_DEFINITIONS = [
    ("scs_01_product_status", "1", "I would buy a product just because it has status.", 0.78, False),
    ("scs_02_new_products", "2", "I am interested in new products with status.", 0.80, False),
    ("scs_03_pay_more", "3", "I would pay more for a product if it had status.", 0.83, False),
    ("scs_04_status_irrelevant", "4", "The status of a product is irrelevant to me.", -0.70, True),
    ("scs_05_snob_appeal", "5", "A product is more valuable to me if it has some snob appeal.", 0.59, False),
]
```

- [ ] Build `STATUS_CONSUMPTION_ITEMS` from `_ITEM_DEFINITIONS`.

Each item should include:

- `id`, `code`, `order`, `text`, and `response_format`
- metadata:
  - `source_table: "Table 1"`
  - `factor_loading`
  - `reverse_scored`
  - `source_construct: "status_consumption"`

- [ ] Define one `Scale`.

Use:

- `id="status_consumption"`
- `name="Status consumption"`
- `construct="Motivational process by which individuals strive to improve their social standing through conspicuous consumption of consumer products that confer or symbolize status."`
- `item_mappings` for all five items
- `reverse_scored=True` only for `scs_04_status_irrelevant`
- mapping metadata with factor loading, source table, and the source note for item 4

- [ ] Define `STATUS_CONSUMPTION_SCALE`.

Use:

- `id="status_consumption_scale"`
- `shorthand="scs"`
- `name="Status Consumption Scale"`
- `version="1.0"`
- `language="en"`
- one section:
  - `id="scs_items"`
  - `title="Status Consumption Scale items"`
  - item IDs in source order
- `items=STATUS_CONSUMPTION_ITEMS`
- `scales=[...]`
- `scoring_models=[]`
- metadata:
  - `retain_history: True`
  - `source_doi`
  - `source_url`
  - `source_table: "Table 1"`
  - `source_article`
  - `source_response_format`
  - `source_construct_definition`
  - `study_two_cronbach_alpha: 0.86`
  - `study_two_average_inter_item_correlation: 0.55`
  - `study_two_single_factor_eigenvalue: 2.78`
  - `study_two_variance_explained: 55.6`
  - `study_four_test_retest_reliability: 0.78`
  - `validation_summary`
  - `scoring_models_status: "deferred"`
  - `scoring_models_note`

- [ ] Create `__init__.py`.

Export:

```python
from llm_behavior_lab.questionnaires.status_consumption.definition import (
    SCS,
    STATUS_CONSUMPTION_SCALE,
)

__all__ = ["SCS", "STATUS_CONSUMPTION_SCALE"]
```

Also define `SCS = STATUS_CONSUMPTION_SCALE` in `definition.py`.

- [ ] Run the focused tests.

```bash
uv run pytest tests/llm_behavior_lab/questionnaires/test_status_consumption.py -q
```

Expected result: tests still fail only for missing package-level exports and README links.

### Task 3: Export the Questionnaire and Document the Source

**Files:**

- Modify: `src/llm_behavior_lab/questionnaires/__init__.py`
- Create: `src/llm_behavior_lab/questionnaires/status_consumption/README.md`
- Modify: `README.md`

- [ ] Update `src/llm_behavior_lab/questionnaires/__init__.py`.

Add imports:

```python
from llm_behavior_lab.questionnaires.status_consumption import (
    SCS,
    STATUS_CONSUMPTION_SCALE,
)
```

Add both names to `__all__`.

- [ ] Create `src/llm_behavior_lab/questionnaires/status_consumption/README.md`.

Include:

- Source citation and DOI.
- Statement that this is the final five-item scale from Table 1.
- Seven-point Likert-type response format with unspecified anchors in available source text.
- One reverse-scored item.
- One status consumption scale mapping.
- Validation summary: unidimensionality, alpha range across studies, criterion validity, discriminant validity, nomological validity, test-retest reliability, social desirability and yea-saying checks.
- Scoring note: scale mappings are preserved, but executable scoring models are deferred until project scoring behavior is implemented.

- [ ] Update the root `README.md` questionnaire list.

Add:

```markdown
- [Status Consumption Scale (SCS)](src/llm_behavior_lab/questionnaires/status_consumption/README.md)
```

- [ ] Run the focused tests again.

```bash
uv run pytest tests/llm_behavior_lab/questionnaires/test_status_consumption.py -q
```

Expected result: all SCS tests pass.

### Task 4: Run Full Verification

**Files:**

- No additional file changes expected unless verification reveals issues.

- [ ] Run all questionnaire tests.

```bash
uv run pytest tests/llm_behavior_lab/questionnaires -q
```

Expected result: all questionnaire tests pass.

- [ ] Run the full test suite.

```bash
uv run pytest -q
```

Expected result: all tests pass.

- [ ] Run lint.

```bash
uv run ruff check .
```

Expected result: no lint findings.

- [ ] Run type checking.

```bash
uv run ty check
```

Expected result: no type errors.

## Scope Boundaries

- Do not implement questionnaire scoring output yet. Store reverse scoring and mappings in `Scale.item_mappings`.
- Do not add a CLI selector for SCS unless a later task asks for questionnaire selection behavior.
- Do not add the source PDF or extracted text to version control.
- Do not invent response anchors that are not visible in the source PDF text.
- Do not change existing questionnaire models unless implementation exposes a validation gap.

## Risks and Decisions

- **Response anchors:** The article states seven-point Likert-type but the available text extraction does not expose anchor wording. Use numeric bounds only and record this explicitly in metadata and README.
- **Reverse scoring:** Preserve the negatively worded item at the scale-mapping level with `reverse_scored=True`; do not duplicate scoring outputs into response records.
- **Factor loading sign:** Preserve item 4 factor loading as `-0.70` because Table 1 reports it as negative for the negatively worded item.
- **Scoring model deferral:** The source scale is commonly summed after reversing item 4, but this project has not implemented questionnaire scoring behavior. Keep `scoring_models=[]` and document the deferral.

## Final Verification Checklist

- [ ] `uv run pytest tests/llm_behavior_lab/questionnaires/test_status_consumption.py -q`
- [ ] `uv run pytest tests/llm_behavior_lab/questionnaires -q`
- [ ] `uv run pytest -q`
- [ ] `uv run ruff check .`
- [ ] `uv run ty check`

