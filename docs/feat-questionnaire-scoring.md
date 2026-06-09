# Questionnaire Scoring and Psychometrics

## Problem

Questionnaire definitions already describe scales, item mappings, reverse scoring,
weights, transformations, output ranges, and interpretation bands. Runtime analysis,
however, stops at item-level response tables. Researchers must implement scoring
separately, which risks inconsistent treatment of missing responses, reverse-coded
items, and questionnaire versions.

## Proposed Feature

Add a questionnaire-aware analysis workflow that validates completed responses against
the selected questionnaire and scoring model, then produces scale-level scores and
psychometric summaries without changing the item-level JSONL schema.

## User and Research Value

- Converts collected responses into analysis-ready scale scores.
- Applies scoring rules consistently across experiments and models.
- Makes protocol conditions and model configurations directly comparable.
- Surfaces reliability and data-completeness problems before downstream analysis.

## Core Capabilities

- Select a questionnaire scoring model by stable ID.
- Apply answer mappings, reverse scoring, item weights, and scoring roles.
- Calculate `sum`, `mean`, and `weighted_mean` transformations.
- Enforce output ranges and assign interpretation bands when defined.
- Report missing, invalid, and unscorable items for each subject and scale.
- Calculate suitable reliability summaries, beginning with Cronbach's alpha.
- Export scale-level results and summaries as validated JSONL and CSV.

## In Scope

- Likert, numeric, and explicitly mapped choice answers.
- Existing `Questionnaire`, `Scale`, `ItemMapping`, and `ScoringModel` definitions.
- Subject-level scale scores and experiment-level descriptive summaries.
- Protocol identifiers and factor values carried into analysis outputs.
- Explicit scoring metadata identifying questionnaire and scoring-model versions.

## Out of Scope

- Changing original item-response records after collection.
- Clinical diagnosis or automated psychological conclusions.
- Item-response theory, norming, or population percentile estimation.
- Inventing scoring rules for questionnaires that do not define them.

## Acceptance Criteria

- A caller can score a run by supplying its response path, questionnaire, and scoring
  model ID.
- Reverse-scored, weighted, and mapped items produce deterministic expected values.
- Every output identifies its questionnaire, scale, and scoring-model versions.
- Missing or invalid responses follow a documented policy and are reported explicitly.
- Unsupported answer types or incomplete scoring definitions fail with actionable
  validation errors.
- Scored protocol runs retain base subject, condition, iteration, and factor metadata.
- Unit tests cover every supported transformation and representative failure cases.

## Dependencies and Risks

- Questionnaire scoring definitions must be complete and internally consistent.
- Missing-data policy affects scientific interpretation and must be explicit.
- Reliability statistics can be misleading for very small samples or short scales.
- Future scoring output schemas should be versioned independently from raw responses.

