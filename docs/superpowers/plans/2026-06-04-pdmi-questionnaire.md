# PDMI Questionnaire Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Purchase Decision-Making Inventory (PDMI) from the first local source paper in `.local_sources`.

**Architecture:** Add PDMI as one questionnaire definition with two sections, emotional and reasoned purchase decision-making, using the existing questionnaire Pydantic models. Keep source PDFs local and untracked.

**Tech Stack:** Python, Pydantic questionnaire models, pytest, ruff, ty, prek.

---

## Summary

Implement the first local source paper in `.local_sources`: Soler Anguiano et al. (2019), “Development and validation of the Inventory of Emotional and Reasoned Purchases Decision-Making Styles (PDMI).”

Use one `Questionnaire` named `PURCHASE_DECISION_MAKING_INVENTORY`, with two sections:

- `pdmi_emotional_items`: 30 items from Table 2
- `pdmi_reasoned_items`: 20 items from Table 3

Do not commit `.local_sources/`, PDFs, `Zone.Identifier` files, or extracted temporary artifacts.

## Key Changes

- Add `src/llm_psych_scales/questionnaires/pdmi/`.
- Export `PURCHASE_DECISION_MAKING_INVENTORY`.
- Optionally export alias `PDMI`.
- Use a shared `LikertFormat(min_value=1, max_value=5)` with labels:
  - `1: "Never"`
  - `2: "Rarely"`
  - `3: "Sometimes"`
  - `4: "Often"`
  - `5: "Always"`
- Preserve source item order, source codes like `C56`, and exact English wording from Tables 2 and 3.
- Add eight `Scale` definitions:
  - Emotional: `impulsivity`, `indebtedness`, `negative_emotions`, `frustration`, `hedonism`
  - Reasoned: `saving`, `reasoning`, `search_of_information`
- Store factor loadings, source table, source scale, Cronbach alpha where available, variance explained where available, DOI, source URL, article citation, and licence in metadata.
- Add `src/llm_psych_scales/questionnaires/pdmi/README.md` documenting:
  - source paper and DOI
  - response format
  - emotional/reasoned sections
  - subscales
  - cultural/sample context from the article
- Update root `README.md` to link to the PDMI README from the questionnaire definitions section.

## Test Plan

Add `tests/llm_psych_scales/questionnaires/test_pdmi.py` covering:

- `PURCHASE_DECISION_MAKING_INVENTORY` validates as a `Questionnaire`.
- Questionnaire identity:
  - `id == "purchase_decision_making_inventory"`
  - `shorthand == "pdmi"`
  - `name == "Purchase Decision-Making Inventory"`
  - `version == "1.0"`
  - `language == "en"`
- It has exactly 50 items.
- Section item IDs match item order:
  - 30 emotional items
  - 20 reasoned items
- Response format is the source 1-never to 5-always frequency scale.
- Item wording and source codes match Tables 2 and 3.
- Scale mappings match the eight PDMI subscales.
- Factor loading metadata is preserved for representative and boundary items from each subscale.
- README file exists and root README links to it.

Run verification:

```bash
uv run pytest tests/llm_psych_scales/questionnaires/test_pdmi.py -q
uv run pytest -q
uv run ruff check .
uv run ty check
uv run prek run --all-files
```

## Assumptions

- “First paper” means `.local_sources/74a41f48-42bb-4e6f-8ac4-92dd223b1bab.pdf`, identified as the PDMI article.
- Implement the full PDMI as one questionnaire.
- Use the EFA table item sets from Tables 2 and 3 because they provide complete item wording, subscales, and loadings.
- Do not add scoring output behavior yet; define scale mappings only, consistent with the existing questionnaire model.
- Keep existing uncommitted work untouched except for README integration where necessary.
